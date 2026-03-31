import torch
import torch.nn.functional as F
import comfy.model_management as mm

from .utils import (
    IMAGENET_MEAN, IMAGENET_STD, DEFAULT_PATCH_SIZE,
    format_camera_params, process_tensor_to_image, process_tensor_to_mask,
    resize_to_patch_multiple, check_model_capabilities,
    imagenet_normalize, save_gaussians_to_ply,
)
from .normalization import (
    apply_edge_antialiasing,
    apply_standard_normalization,
    apply_v2_style_normalization,
    apply_raw_normalization,
)


class DepthAnything_V3():
    @classmethod
    def execute(cls, da3_model, images, normalization_mode="V2-Style", camera_params=None, resize_method="resize", invert_depth=False, keep_model_size=False):
        device = mm.get_torch_device()

        mm.load_models_gpu([da3_model])
        model = da3_model.model
        capabilities = da3_model.model_options.get("da3_capabilities", check_model_capabilities(model))
        dtype = da3_model.model_options.get("da3_dtype", torch.float16)
        B, H, W, C = images.shape
        images_pt = images.permute(0, 3, 1, 2)
        images_pt, orig_H, orig_W = resize_to_patch_multiple(images_pt, DEFAULT_PATCH_SIZE, resize_method)
        normalized_images = imagenet_normalize(images_pt)
        normalized_images = normalized_images.unsqueeze(1)
        extrinsics_input = None
        intrinsics_input = None
        if camera_params is not None:
            if capabilities["has_camera_conditioning"]:
                extrinsics_input = camera_params["extrinsics"].to(device).to(dtype)
                intrinsics_input = camera_params["intrinsics"].to(device).to(dtype)
                if extrinsics_input.shape[0] == 1 and B > 1:
                    extrinsics_input = extrinsics_input.expand(B, -1, -1, -1)
                    intrinsics_input = intrinsics_input.expand(B, -1, -1, -1)

        depth_out = []
        conf_out = []
        sky_out = []
        ray_origin_out = []
        ray_dir_out = []
        extrinsics_list = []
        intrinsics_list = []
        gaussians_list = []

        infer_gs = capabilities["has_3d_gaussians"]
        for i in range(B):
            img = normalized_images[i:i+1].to(device, dtype=dtype)
            ext_i = extrinsics_input[i:i+1] if extrinsics_input is not None else None
            int_i = intrinsics_input[i:i+1] if intrinsics_input is not None else None
            output = model(img, extrinsics=ext_i, intrinsics=int_i, infer_gs=infer_gs)

            depth = None
            if hasattr(output, 'depth'):
                depth = output.depth
            elif isinstance(output, dict) and 'depth' in output:
                depth = output['depth']

            if depth is None or not torch.is_tensor(depth):
                raise ValueError("Model output does not contain valid depth tensor")

            conf = None
            if hasattr(output, 'depth_conf'):
                conf = output.depth_conf
            elif isinstance(output, dict) and 'depth_conf' in output:
                conf = output['depth_conf']
            if conf is None or not torch.is_tensor(conf):
                conf = torch.ones_like(depth)
            sky = None
            if hasattr(output, 'sky'):
                sky = output.sky
            elif isinstance(output, dict) and 'sky' in output:
                sky = output['sky']

            if sky is None or not torch.is_tensor(sky):
                sky = torch.zeros_like(depth)
            else:
                sky_min, sky_max = sky.min(), sky.max()
                if sky_max > sky_min:
                    sky = (sky - sky_min) / (sky_max - sky_min)

            if normalization_mode == "Raw":
                depth_processed = apply_raw_normalization(depth, invert_depth)
            elif normalization_mode == "V2-Style":
                depth_processed = apply_v2_style_normalization(depth, sky, device, invert_depth)
            else:  # "Standard"
                depth_processed = apply_standard_normalization(depth, invert_depth)

            conf_range = conf.max() - conf.min()
            if conf_range > 1e-8:
                conf = (conf - conf.min()) / conf_range
            else:
                conf = torch.ones_like(conf)

            depth_out.append(depth_processed.cpu())
            conf_out.append(conf.cpu())
            sky_out.append(sky.cpu())

            ray = None
            if hasattr(output, 'ray'):
                ray = output.ray
            elif isinstance(output, dict) and 'ray' in output:
                ray = output['ray']

            if ray is not None and torch.is_tensor(ray):
                ray = ray.squeeze(0).squeeze(0)  # [6, H, W]
                ray_origin = ray[:3]
                ray_dir = ray[3:6]
                ray_origin_out.append(ray_origin.cpu())
                ray_dir_out.append(ray_dir.cpu())
            else:
                ray_origin_out.append(torch.zeros(3, depth.shape[-2], depth.shape[-1]))
                ray_dir_out.append(torch.zeros(3, depth.shape[-2], depth.shape[-1]))

            extr = None
            if hasattr(output, 'extrinsics'):
                extr = output.extrinsics
            elif isinstance(output, dict) and 'extrinsics' in output:
                extr = output['extrinsics']

            if extr is not None and torch.is_tensor(extr):
                extrinsics_list.append(extr.cpu())
            else:
                extrinsics_list.append(None)

            intr = None
            if hasattr(output, 'intrinsics'):
                intr = output.intrinsics
            elif isinstance(output, dict) and 'intrinsics' in output:
                intr = output['intrinsics']

            if intr is not None and torch.is_tensor(intr):
                intrinsics_list.append(intr.cpu())
            else:
                intrinsics_list.append(None)

            if infer_gs:
                gs = None
                if hasattr(output, 'gaussians'):
                    gs = output.gaussians
                elif isinstance(output, dict) and 'gaussians' in output:
                    gs = output['gaussians']

                if gs is not None and hasattr(gs, 'means') and torch.is_tensor(gs.means):
                    gaussians_list.append((gs, depth))

        normalize_depth_output = (normalization_mode != "Raw")
        depth_final = process_tensor_to_image(depth_out, orig_H, orig_W, normalize_output=normalize_depth_output, skip_resize=keep_model_size)

        if not keep_model_size:
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2
            model_H, model_W = images_pt.shape[2], images_pt.shape[3]

            if final_H != model_H or final_W != model_W:
                scale_h = final_H / model_H
                scale_w = final_W / model_W

                for i, intr in enumerate(intrinsics_list):
                    if intr is not None and torch.is_tensor(intr):
                        intr_scaled = intr.squeeze().clone()
                        intr_scaled[0, 0] *= scale_w  # fx
                        intr_scaled[1, 1] *= scale_h  # fy
                        intr_scaled[0, 2] *= scale_w  # cx
                        intr_scaled[1, 2] *= scale_h  # cy
                        intrinsics_list[i] = intr_scaled

        return depth_final

    @staticmethod
    def _process_ray_to_image(ray_list, orig_H, orig_W, normalize=True, skip_resize=False):
        out = torch.cat([r.unsqueeze(0) for r in ray_list], dim=0)

        if normalize:
            for i in range(out.shape[0]):
                ray_batch = out[i]
                ray_min = ray_batch.min()
                ray_max = ray_batch.max()
                if ray_max > ray_min:
                    out[i] = (ray_batch - ray_min) / (ray_max - ray_min)
                else:
                    out[i] = torch.zeros_like(ray_batch)

        out = out.permute(0, 2, 3, 1).float()

        if not skip_resize:
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2
            if out.shape[1] != final_H or out.shape[2] != final_W:
                out = F.interpolate(out.permute(0, 3, 1, 2), size=(final_H, final_W), mode="bilinear").permute(0, 2, 3, 1)

        if normalize:
            return torch.clamp(out, 0, 1)
        else:
            return out
