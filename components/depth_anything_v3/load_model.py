import logging
import torch
import os

import comfy.ops
import comfy.model_management as mm
import comfy.model_patcher
from comfy.utils import load_torch_file
import folder_paths

from .depth_anything_v3.configs import MODEL_CONFIGS, MODEL_REPOS
from .depth_anything_v3.model import (DepthAnything3Net, DinoV2, DualDPT, DPT,)
from .depth_anything_v3.camera import CameraEnc, CameraDec
from .depth_anything_v3.gs import GSDPT, GaussianAdapter
from .utils import DEFAULT_PATCH_SIZE, check_model_capabilities

_da3_model_dir = os.path.join(folder_paths.models_dir, "depthanything3")
os.makedirs(_da3_model_dir, exist_ok=True)
folder_paths.add_model_folder_path("depth_anything_v3", _da3_model_dir)


def _get_da3_model_list():
    local_models = folder_paths.get_filename_list("depth_anything_v3")
    known_models = list(MODEL_REPOS.keys())
    return list(dict.fromkeys(known_models + local_models))

ENCODER_EMBED_DIMS = {
    'vits': 384,
    'vitb': 768,
    'vitl': 1024,
    'vitg': 1536,
}

def detect_da3_variant(state_dict):
    keys = set(state_dict.keys())
    stripped_keys = set()
    for k in keys:
        stripped_keys.add(k[6:] if k.startswith('model.') else k)

    has_da3_metric = any(k.startswith('da3_metric.') for k in stripped_keys)
    if has_da3_metric:
        return 'da3nested-giant-large'

    has_da3_prefix = any(k.startswith('da3.') for k in stripped_keys)
    prefix = 'da3.' if has_da3_prefix else ''
    patch_key = f'{prefix}net.patch_embed.proj.weight'
    if patch_key in stripped_keys:
        embed_dim = state_dict.get(patch_key, state_dict.get(f'model.{patch_key}')).shape[0]
    else:
        embed_dim = None
        for k in stripped_keys:
            if 'patch_embed.proj.weight' in k:
                sd_key = k if k in state_dict else f'model.{k}'
                if sd_key in state_dict:
                    embed_dim = state_dict[sd_key].shape[0]
                break

    dim_to_encoder = {384: 'vits', 768: 'vitb', 1024: 'vitl', 1536: 'vitg'}
    encoder = dim_to_encoder.get(embed_dim)
    has_cam = any('cam_enc' in k for k in stripped_keys)
    has_gs = any('gs_head' in k for k in stripped_keys)
    head_output_key = f'{prefix}head.output_conv1.2.weight'
    is_dual_head = False
    if head_output_key in stripped_keys:
        sd_key = head_output_key if head_output_key in state_dict else f'model.{head_output_key}'
        if sd_key in state_dict:
            out_channels = state_dict[sd_key].shape[0]
            is_dual_head = (out_channels == 2)

    if encoder == 'vitg' and has_cam:
        if has_gs:
            return 'da3-giant'
        return 'da3-giant'  # giant always has GS potential
    elif encoder == 'vitl':
        if has_cam and is_dual_head:
            return 'da3-large'
        elif not has_cam and not is_dual_head:
            return 'da3mono-large'
        elif has_cam:
            return 'da3-large'
        else:
            return 'da3mono-large'
    elif encoder == 'vitb':
        return 'da3-base'
    elif encoder == 'vits':
        return 'da3-small'
    return 'da3-large'


def detect_da3_variant_with_filename_hint(state_dict, filename):
    variant = detect_da3_variant(state_dict)
    if variant == 'da3mono-large' and filename:
        fname_lower = filename.lower()
        if 'metric' in fname_lower:
            return 'da3metric-large'

    return variant

def _build_gs_modules(config, operations):
    gs_head = GSDPT(
        dim_in=config['dim_in'],
        output_dim=38,
        features=config['features'],
        out_channels=config['out_channels'],
        operations=operations,
    )

    gs_adapter = GaussianAdapter(
        sh_degree=2,
        pred_color=False,
        pred_offset_depth=True,
        pred_offset_xy=True,
        gaussian_scale_min=1e-5,
        gaussian_scale_max=30.0,
    )

    return gs_head, gs_adapter


class DA3ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.da3 = model

    def forward(self, *args, **kwargs):
        return self.da3(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.da3 = self.da3.to(*args, **kwargs)
        return self

    @property
    def cam_enc(self):
        return self.da3.cam_enc if hasattr(self.da3, 'cam_enc') else None

    @property
    def cam_dec(self):
        return self.da3.cam_dec if hasattr(self.da3, 'cam_dec') else None

    @property
    def gs_head(self):
        return self.da3.gs_head if hasattr(self.da3, 'gs_head') else None

    @property
    def gs_adapter(self):
        return self.da3.gs_adapter if hasattr(self.da3, 'gs_adapter') else None


class NestedModelWrapper(torch.nn.Module):
    def __init__(self, da3_main, da3_metric):
        super().__init__()
        self.da3 = da3_main
        self.da3_metric = da3_metric

    def forward(self, *args, **kwargs):
        from .depth_anything_v3.alignment import (
            apply_metric_scaling, compute_sky_mask, compute_alignment_mask,
            sample_tensor_for_quantile, least_squares_scale_scalar
        )

        output = self.da3(*args, **kwargs)
        x = args[0] if args else kwargs.get('x')
        infer_gs = kwargs.get('infer_gs', False)
        metric_output = self.da3_metric(x, infer_gs=infer_gs)

        metric_output.depth = apply_metric_scaling(
            metric_output.depth,
            output.intrinsics,
        )

        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        if non_sky_mask.sum() > 10:
            depth_conf_ns = output.depth_conf[non_sky_mask]
            depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
            median_conf = torch.quantile(depth_conf_sampled, 0.5)

            align_mask = compute_alignment_mask(
                output.depth_conf, non_sky_mask, output.depth, metric_output.depth, median_conf
            )

            valid_depth = output.depth[align_mask]
            valid_metric_depth = metric_output.depth[align_mask]
            scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

            output.depth *= scale_factor
            if hasattr(output, 'extrinsics') and output.extrinsics is not None:
                output.extrinsics[:, :, :3, 3] *= scale_factor
            output.is_metric = 1
            output.scale_factor = scale_factor.item()

            non_sky_depth = output.depth[non_sky_mask]
            if non_sky_depth.numel() > 100000:
                idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
                sampled_depth = non_sky_depth.flatten()[idx]
            else:
                sampled_depth = non_sky_depth
            max_depth = torch.quantile(sampled_depth, 0.99)
            sky_depth = max(200.0, max_depth.item() * 2.0)
            output.depth[~non_sky_mask] = sky_depth
            output.sky = metric_output.sky
        else:
            output.sky = metric_output.sky

        return output

    def to(self, *args, **kwargs):
        self.da3 = self.da3.to(*args, **kwargs)
        self.da3_metric = self.da3_metric.to(*args, **kwargs)
        return self

    @property
    def cam_enc(self):
        return self.da3.cam_enc if hasattr(self.da3, 'cam_enc') else None

    @property
    def cam_dec(self):
        return self.da3.cam_dec if hasattr(self.da3, 'cam_dec') else None

    @property
    def gs_head(self):
        return self.da3.gs_head if hasattr(self.da3, 'gs_head') else None

    @property
    def gs_adapter(self):
        return self.da3.gs_adapter if hasattr(self.da3, 'gs_adapter') else None


def _build_da3_model(model_path, model_key, dtype, attention):
    config = MODEL_CONFIGS[model_key]
    is_nested = config.get('is_nested', False)
    operations = comfy.ops.manual_cast

    with torch.device("meta"):
        if is_nested:
            backbone_main = DinoV2(
                name=config['encoder'],
                out_layers=config.get('out_layers', [19, 27, 33, 39]),
                alt_start=config.get('alt_start', 13),
                qknorm_start=config.get('qknorm_start', 13),
                rope_start=config.get('rope_start', 13),
                cat_token=config.get('cat_token', True),
                operations=operations,
            )
            head_main = DualDPT(
                dim_in=config['dim_in'],
                output_dim=2,
                features=config['features'],
                out_channels=config['out_channels'],
                operations=operations,
            )
            embed_dim = ENCODER_EMBED_DIMS.get(config['encoder'], 1536)
            cam_enc_main = CameraEnc(
                dim_out=embed_dim,
                dim_in=9,
                trunk_depth=4,
                num_heads=embed_dim // 64,
                mlp_ratio=4,
                init_values=0.01,
                operations=operations,
            )
            cam_dec_main = CameraDec(dim_in=config['dim_in'], operations=operations)
            gs_head_main, gs_adapter_main = _build_gs_modules(config, operations)

            da3_main = DepthAnything3Net(
                net=backbone_main,
                head=head_main,
                cam_dec=cam_dec_main,
                cam_enc=cam_enc_main,
                gs_head=gs_head_main,
                gs_adapter=gs_adapter_main,
            )

            metric_config = MODEL_CONFIGS.get('da3metric-large', {
                'encoder': 'vitl',
                'features': 256,
                'out_channels': [256, 512, 1024, 1024],
                'dim_in': 1024,
                'out_layers': [4, 11, 17, 23],
            })
            backbone_metric = DinoV2(
                name=metric_config.get('encoder', 'vitl'),
                out_layers=metric_config.get('out_layers', [4, 11, 17, 23]),
                alt_start=-1,
                qknorm_start=-1,
                rope_start=-1,
                cat_token=False,
                operations=operations,
            )
            head_metric = DPT(
                dim_in=metric_config.get('dim_in', 1024),
                output_dim=1,
                features=metric_config.get('features', 256),
                out_channels=metric_config.get('out_channels', [256, 512, 1024, 1024]),
                operations=operations,
            )
            da3_metric = DepthAnything3Net(
                net=backbone_metric,
                head=head_metric,
                cam_dec=None,
                cam_enc=None,
                gs_head=None,
                gs_adapter=None,
            )

            inner_model = NestedModelWrapper(da3_main, da3_metric)
        else:
            backbone = DinoV2(
                name=config['encoder'],
                out_layers=config.get('out_layers', [4, 11, 17, 23]),
                alt_start=config.get('alt_start', -1),
                qknorm_start=config.get('qknorm_start', -1),
                rope_start=config.get('rope_start', -1),
                cat_token=config.get('cat_token', False),
                operations=operations,
            )

            if config.get('is_mono', False) or config.get('is_metric', False):
                head = DPT(
                    dim_in=config['dim_in'],
                    output_dim=1,
                    features=config['features'],
                    out_channels=config['out_channels'],
                    operations=operations,
                )
            else:
                head = DualDPT(
                    dim_in=config['dim_in'],
                    output_dim=2,
                    features=config['features'],
                    out_channels=config['out_channels'],
                    operations=operations,
                )

            cam_enc = None
            cam_dec = None
            if config.get('has_cam', False) and config.get('alt_start', -1) != -1:
                embed_dim = ENCODER_EMBED_DIMS.get(config['encoder'], 1024)
                cam_enc = CameraEnc(
                    dim_out=embed_dim,
                    dim_in=9,
                    trunk_depth=4,
                    num_heads=embed_dim // 64,
                    mlp_ratio=4,
                    init_values=0.01,
                    operations=operations,
                )
                cam_dec = CameraDec(dim_in=config['dim_in'], operations=operations)

            gs_head = None
            gs_adapter = None
            if config.get('has_3d_gaussians', model_key == 'da3-giant'):
                gs_head, gs_adapter = _build_gs_modules(config, operations)

            inner_model = DepthAnything3Net(
                net=backbone,
                head=head,
                cam_dec=cam_dec,
                cam_enc=cam_enc,
                gs_head=gs_head,
                gs_adapter=gs_adapter,
            )

    state_dict = load_torch_file(model_path)
    new_state_dict = {}
    stripped_count = 0
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('model.'):
            new_key = new_key[6:]
            stripped_count += 1
        new_state_dict[new_key] = value

    has_da3_prefix = any(k.startswith('da3.') for k in new_state_dict.keys())

    if is_nested:
        model = inner_model
    elif has_da3_prefix:
        model = DA3ModelWrapper(inner_model)
    else:
        model = inner_model

    expanded = {}
    for key in list(new_state_dict.keys()):
        if 'output_conv2_aux.0.' in key:
            prefix, suffix = key.split('output_conv2_aux.0.')
            for idx in range(1, 4):
                clone_key = f"{prefix}output_conv2_aux.{idx}.{suffix}"
                if clone_key not in new_state_dict:
                    expanded[clone_key] = new_state_dict[key].clone()
    if expanded:
        new_state_dict.update(expanded)

    result = model.load_state_dict(new_state_dict, strict=False, assign=True)
    model.to(dtype=dtype)

    for name, param in list(model.named_parameters()):
        if param.device.type == 'meta':
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], torch.nn.Parameter(
                torch.zeros(param.shape, dtype=dtype), requires_grad=False
            ))

    for name, buf in list(model.named_buffers()):
        if buf.device.type == 'meta':
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            parent.register_buffer(parts[-1], torch.zeros(buf.shape, dtype=buf.dtype))

    model.eval()
    return model


class DownloadAndLoadDepthAnythingV3Model():
    @classmethod
    def execute(cls, model, precision="auto", attention="auto"):
        device = mm.get_torch_device()
        if precision == "auto":
            if mm.should_use_bf16(device):
                dtype = torch.bfloat16
            elif mm.should_use_fp16(device):
                dtype = torch.float16
            else:
                dtype = torch.float32
        elif precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        model_path = folder_paths.get_full_path("depth_anything_v3", model)

        if model_path is None and model in MODEL_REPOS:
            download_dir = os.path.join(folder_paths.models_dir, "depthanything3")
            os.makedirs(download_dir, exist_ok=True)
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to auto-download models. "
                    "Install with: pip install huggingface_hub\n"
                    "Or manually download and place in ComfyUI/models/depthanything3/"
                )

            snapshot_download(
                repo_id=MODEL_REPOS[model],
                allow_patterns=["*.safetensors"],
                local_dir=download_dir,
                local_dir_use_symlinks=False,
            )

            hf_default = os.path.join(download_dir, "model.safetensors")
            target_path = os.path.join(download_dir, model)
            if os.path.exists(hf_default) and not os.path.exists(target_path):
                os.rename(hf_default, target_path)
            model_path = target_path

        if model_path is None:
            raise FileNotFoundError(
                f"Model '{model}' not found in ComfyUI/models/depthanything3/ and not a known HuggingFace model."
            )

        sd = load_torch_file(model_path)
        model_key = detect_da3_variant_with_filename_hint(sd, model)
        del sd  # Free memory before building model
        config = MODEL_CONFIGS[model_key]
        loaded_model = _build_da3_model(model_path, model_key, dtype, attention)

        patcher = comfy.model_patcher.ModelPatcher(
            loaded_model,
            load_device=mm.get_torch_device(),
            offload_device=mm.unet_offload_device(),
        )

        patcher.model_options["da3_capabilities"] = check_model_capabilities(loaded_model)
        patcher.model_options["da3_config"] = config
        patcher.model_options["da3_dtype"] = dtype
        patcher.model_options["da3_model_key"] = model_key

        return patcher
