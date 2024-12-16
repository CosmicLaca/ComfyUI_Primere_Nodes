# from dataclasses import dataclass, field
from typing import Tuple
import torch
import torch.nn as nn
from ..diffusion import DPMS, FlowEuler, SASolverSampler
from ..diffusion.model.utils import resize_and_crop_tensor
from diffusers.image_processor import PixArtImageProcessor
from ..diffusion.data.datasets.utils import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST, ASPECT_RATIO_2048_TEST
from comfy.model_management import total_vram, vae_dtype, get_torch_device, vae_offload_device, unet_offload_device, soft_empty_cache
# from ..nodes import is_lowvram, vae_dtype

device = get_torch_device()
vae_dtype = vae_dtype(device, [torch.float16, torch.bfloat16, torch.float32])
is_lowvram = (torch.cuda.is_available() and total_vram < 8888)

def guidance_type_select(default_guidance_type, pag_scale, attn_type):
    guidance_type = default_guidance_type
    if not (pag_scale > 1.0 and attn_type == "linear"):
        guidance_type = "classifier-free"
    return guidance_type


def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
    """Returns binned height and width."""
    ar = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return int(default_hw[0]), int(default_hw[1])

class SanaPipeline(nn.Module):
    def __init__(
        self,
        config: dict = {},
        vae=None,
        weight_dtype=torch.float16,
        unet=None,
        **kargs,
    ):
        super().__init__()
        self.config = config
        # set some hyper-parameters
        self.image_size = self.config.model.image_size
        self.flow_shift = config.scheduler.flow_shift
        self.weight_dtype = weight_dtype
        self.base_ratios = eval(f"ASPECT_RATIO_{self.image_size}_TEST")
        self.model = unet
        self.vae = vae
        self.vae_scale_factor = 32 if vae == None else 2 ** (len(self.vae.cfg.encoder.width_list) - 1) # upscale
        self.vae_scaling_factor = 0.41407 if vae == None else self.vae.cfg.scaling_factor # down scale
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @torch.inference_mode()
    def forward(
        self,
        cond,
        uncond,
        num_inference_steps=20,
        guidance_scale=5,
        pag_guidance_scale=2.5,
        num_images_per_prompt=1,
        generator=torch.Generator().manual_seed(42),
        latents=None,
        noise_scheduler='flow_dpm-solver',
        output_type=True,
        pag_applied_layers=None,
    ):
        guidance_type = "classifier-free_PAG"
        self.device = device
        if latents.shape[1] == 32:
            width, height = latents.shape[3] * self.vae_scale_factor, latents.shape[2] * self.vae_scale_factor
        else: # comfy empty latent
            width, height = latents.shape[3] * 8, latents.shape[2] * 8
        self.ori_height, self.ori_width = height, width
        self.height, self.width = classify_height_width_bin(height, width, ratios=self.base_ratios)
        self.latent_size_h, self.latent_size_w = (
            self.height // self.config.vae.vae_downsample_rate,
            self.width // self.config.vae.vae_downsample_rate,
        )
        self.guidance_type = guidance_type_select(guidance_type, pag_guidance_scale, self.config.model.attn_type)
        
        pag_applied_layers = pag_applied_layers if pag_applied_layers != None else self.config.model.pag_applied_layers

        hw, ar = (
            torch.tensor([[self.image_size, self.image_size]], dtype=torch.float, device=self.device).repeat(num_images_per_prompt, 1),
            torch.tensor([[1.0]], device=self.device).repeat(num_images_per_prompt, 1),
        )
        for _ in range(num_images_per_prompt):
            # with torch.no_grad():
            with torch.autocast(self.device.type):
            # with torch.inference_mode():
                n = latents.shape[0]
                caption_embs, null_y, emb_masks = cond[0][0].to(self.weight_dtype), uncond[0][0].to(self.weight_dtype), cond[0][1]['emb_masks']
                
                if latents.shape[1] != 32: 
                    # resize comfy empty latent to DCAE latent size
                    # resized_latents = latents.clone().resize_(latents.shape[0], self.config.vae.vae_latent_dim, self.latent_size_h, self.latent_size_w)
                    # resized_latents = torch.randn_like(resized_latents)
                    # z = resized_latents.to(self.device, self.weight_dtype)
                    z = torch.randn(
                        n,
                        self.config.vae.vae_latent_dim,
                        self.latent_size_h,
                        self.latent_size_w,
                        generator=generator,
                        device=self.device,
                    )
                else:
                    z = latents.to(self.device, self.weight_dtype)
                    
                try:
                    self.model.to(device)
                except torch.cuda.OutOfMemoryError as e:
                    raise e
                
                model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)
                if noise_scheduler == "flow_euler":
                    flow_solver = FlowEuler(
                        self.model,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=guidance_scale,
                        model_kwargs=model_kwargs,
                    )
                    sample = flow_solver.sample(
                        z,
                        steps=num_inference_steps,
                    )
                elif noise_scheduler == "sa-solver":
                    sa_solver = SASolverSampler(self.model, device=device)
                    sample = sa_solver.sample(
                        S=num_inference_steps,
                        batch_size=n,
                        shape=(self.config.vae.vae_latent_dim, self.latent_size_h, self.latent_size_w),
                        eta=1,
                        conditioning=caption_embs,
                        unconditional_conditioning=null_y,
                        unconditional_guidance_scale=guidance_scale,
                        model_kwargs=model_kwargs,
                    )[0]
                elif noise_scheduler == "dpm-solver":
                    dpm_solver = DPMS(
                        self.model,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=guidance_scale,
                        model_kwargs=model_kwargs,
                    )
                    sample = dpm_solver.sample(
                        z,
                        steps=num_inference_steps,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )
                elif noise_scheduler == "flow_dpm-solver":
                    scheduler = DPMS(
                        self.model,
                        condition=caption_embs,
                        uncondition=null_y,
                        guidance_type=self.guidance_type,
                        cfg_scale=guidance_scale,
                        pag_scale=pag_guidance_scale,
                        pag_applied_layers=pag_applied_layers, #self.config.model.pag_applied_layers, # 0.6b: [14] 1.6b: [8]
                        model_type="flow",
                        model_kwargs=model_kwargs,
                        schedule="FLOW",
                    )
                    sample = scheduler.sample(
                        z,
                        steps=num_inference_steps,
                        order=2,
                        skip_type="time_uniform_flow",
                        method="multistep",
                        flow_shift=self.flow_shift,
                    )

            # sample = sample.to(self.weight_dtype)
            
            if not output_type:
                if is_lowvram:
                    self.model.to(unet_offload_device())
                soft_empty_cache(True)
                self.vae.to(device, vae_dtype)
                
                with torch.no_grad():
                # with torch.inference_mode():
                    sample = self.vae.decode(sample.to(vae_dtype).detach() / self.vae.cfg.scaling_factor)
                sample = resize_and_crop_tensor(sample.float(), self.ori_width, self.ori_height)
                samples = self.image_processor.postprocess(sample.cpu().float())
                
                self.vae.to(vae_offload_device())
            else:
                samples = {
                    'samples': sample,
                    'width': width,
                    'height': height,
                    }
                
        return samples

text_encoder_dict_custom = {
        "T5": "DeepFloyd/t5-v1_1-xxl",
        "T5-small": "google/t5-v1_1-small",
        "T5-base": "google/t5-v1_1-base",
        "T5-large": "google/t5-v1_1-large",
        "T5-xl": "google/t5-v1_1-xl",
        "T5-xxl": "google/t5-v1_1-xxl",
        "gemma-2b": "google/gemma-2b",
        "gemma-2b-it": "google/gemma-2b-it",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "google/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
        "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    }