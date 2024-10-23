import random
import nodes
import torch
from ..components import latentnoise
from ..components import utility
import comfy_extras.nodes_align_your_steps as nodes_align_your_steps
import comfy.samplers
import comfy_extras.nodes_custom_sampler as nodes_custom_sampler
import comfy_extras.nodes_stable_cascade as nodes_stable_cascade
import comfy_extras.nodes_flux as nodes_flux
import comfy_extras.nodes_model_advanced as nodes_model_advanced
from comfy import model_management
import gc
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler
)

def PKSampler(self, device, seed, model,
              steps, cfg, sampler_name, scheduler_name,
              positive, negative,
              latent_image, denoise,
              variation_extender, variation_batch_step_original, batch_counter, variation_extender_original, variation_batch_step, variation_level, variation_limit, align_your_steps, noise_extender):

    if variation_level == True:
        samples = latentnoise.noisy_samples(model, device, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise, seed, noise_extender)
    else:
        if variation_extender_original > 0 or device != 'DEFAULT' or variation_batch_step_original > 0:
            samples = latentnoise.noisy_samples(model, device, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise, seed, noise_extender)
        else:
            if align_your_steps == True:
                modelname_only = model
                model_version = utility.get_value_from_cache('model_version', modelname_only)
                match model_version:
                    case 'SDXL_2048':
                        model_type = 'SDXL'
                    case _:
                        model_type = 'SD1'

                sigmas = nodes_align_your_steps.AlignYourStepsScheduler.get_sigmas(self, model_type, steps, denoise)
                sampler = comfy.samplers.sampler_object(sampler_name)
                AYS_samples = nodes_custom_sampler.SamplerCustom().sample(model, True, seed, cfg, positive, negative, sampler, sigmas[0], latent_image)
                samples = (AYS_samples[0],)
            else:
                samples = nodes.KSampler.sample(self, model, seed, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise=denoise)

    return samples

def PTurboSampler(model, seed, cfg, positive, negative, latent_image, steps, denoise, sampler_name):
    sigmas = nodes_custom_sampler.SDTurboScheduler().get_sigmas(model, steps, denoise)
    sampler = comfy.samplers.sampler_object(sampler_name)
    turbo_samples = nodes_custom_sampler.SamplerCustom().sample(model, True, seed, cfg, positive, negative, sampler, sigmas[0], latent_image)
    samples = (turbo_samples[0],)
    return samples

def PCascadeSampler(self, model, seed, steps, cfg, sampler_name, scheduler_name,
                    positive, negative,
                    latent_image, denoise, device,
                    variation_level, variation_limit, variation_extender_original, variation_batch_step_original, variation_extender, variation_batch_step, batch_counter, noise_extender):

    samples = latent_image
    if type(model).__name__ == 'list':
        latent_size = utility.getLatentSize(latent_image)
        if (latent_size[0] < latent_size[1]):
            orientation = 'Vertical'
        else:
            orientation = 'Horizontal'

        dimensions = utility.get_dimensions_by_shape(self, 'Square [1:1]', 1024, orientation, True, True, latent_size[0], latent_size[1], 'CASCADE')
        dimension_x = dimensions[0]
        dimension_y = dimensions[1]

        height = dimension_y
        width = dimension_x
        compression = 42
        if type(model[0]).__name__ == 'ModelPatcher' and type(model[1]).__name__ == 'ModelPatcher':
            c_latent = {"samples": torch.zeros([1, 16, height // compression, width // compression])}
            b_latent = {"samples": torch.zeros([1, 4, height // 4, width // 4])}

            if variation_level == True:
                samples_c = latentnoise.noisy_samples(model[1], device, steps, cfg, sampler_name, scheduler_name, positive, negative, c_latent, denoise, seed, noise_extender)[0]
            else:
                if variation_extender_original > 0 or device != 'DEFAULT' or variation_batch_step_original > 0:
                    samples_c = latentnoise.noisy_samples(model[1], device, steps, cfg, sampler_name, scheduler_name, positive, negative, c_latent, denoise, seed, noise_extender)[0]
                else:
                    samples_c = nodes.KSampler.sample(self, model[1], seed, steps, cfg, sampler_name, scheduler_name, positive, negative, c_latent, denoise=denoise)[0]
            conditining_c = nodes_stable_cascade.StableCascade_StageB_Conditioning.set_prior(self, positive, samples_c)[0]
            samples = nodes.KSampler.sample(self, model[0], seed, 10, 1.00, sampler_name, scheduler_name, conditining_c, negative, b_latent, denoise=denoise)

    return samples

def PSamplerHyper(self, extra_pnginfo, model, seed, steps, cfg, positive, negative, sampler_name, scheduler_name, latent_image, denoise, prompt):
    WORKFLOWDATA = extra_pnginfo['workflow']['nodes']
    HyperSDSelector = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereModelConceptSelector', 'hypersd_selector', prompt)
    if (HyperSDSelector == 'UNET'):
        sigmas = utility.get_hypersd_sigmas(model)
        sampler = comfy.samplers.sampler_object(sampler_name)
        hyper_samples = nodes_custom_sampler.SamplerCustom().sample(model, True, seed, cfg, positive, negative, sampler, sigmas[0], latent_image)
        samples = (hyper_samples[0],)
    else:
        SamplingDiscreteResults = utility.TCDModelSamplingDiscrete(self, model, steps, scheduler_name, denoise, eta=0.8)
        model = SamplingDiscreteResults[0]
        sampler = SamplingDiscreteResults[1]
        sigmas = SamplingDiscreteResults[2]
        hyper_lora_samples = nodes_custom_sampler.SamplerCustom().sample(model, True, seed, cfg, positive, negative, sampler, sigmas, latent_image)
        samples = (hyper_lora_samples[0],)

    return samples

def PSamplerAdvanced(self, model, seed, WORKFLOWDATA, positive, scheduler_name, sampler_name, steps, denoise, latent_image, prompt):
    FLUX_GUIDANCE = float(utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereModelConceptSelector', 'flux_clip_guidance', prompt))
    if FLUX_GUIDANCE is None:
        FLUX_GUIDANCE = 3.5
    if FLUX_GUIDANCE <= 0:
        CONDITIONING_POS = positive
    else:
        CONDITIONING_POS = nodes_flux.FluxGuidance.append(self, positive, FLUX_GUIDANCE)[0]
    FLUX_GUIDER = nodes_custom_sampler.BasicGuider.get_guider(self, model, CONDITIONING_POS)[0]
    FLUX_SIGMAS = nodes_custom_sampler.BasicScheduler.get_sigmas(self, model, scheduler_name, steps, denoise=denoise)[0]
    FLUX_NOISE = nodes_custom_sampler.RandomNoise.get_noise(self, seed)[0]
    sampler_object = comfy.samplers.sampler_object(sampler_name)
    samples = (nodes_custom_sampler.SamplerCustomAdvanced.sample(self, FLUX_NOISE, FLUX_GUIDER, sampler_object, FLUX_SIGMAS, latent_image)[0],)
    return samples

def PSamplerSD3(self, model, seed, cfg, positive, negative, latent_image, steps, denoise, sampler_name, scheduler_name, model_sampling = 2.5, multiplier = 1000):
    sd3sampling = nodes_model_advanced.ModelSamplingSD3.patch(self, model, model_sampling, multiplier)[0]
    samples = nodes.KSampler.sample(self, sd3sampling, seed, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise=denoise)
    return samples

def PSamplerKOROLS(self, model, seed, cfg, positive, negative, latent_image, steps, denoise, sampler_name, scheduler_name, model_sampling = 0, multiplier = 1000):
    device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()
    vae_scaling_factor = 0.13025
    model_management.soft_empty_cache()
    gc.collect()
    pipeline = model['pipeline']
    scheduler_config = {
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "beta_end": 0.014,
        "dynamic_thresholding_ratio": 0.995,
        "num_train_timesteps": 1100,
        "prediction_type": "epsilon",
        "rescale_betas_zero_snr": False,
        "steps_offset": 1,
        "timestep_spacing": "leading",
        "trained_betas": None,
    }

    noise_scheduler = None
    if scheduler_name == "DPMSolverMultistepScheduler":
        noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
    elif scheduler_name == "DPMSolverMultistepScheduler_SDE_karras":
        scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
        scheduler_config.update({"use_karras_sigmas": True})
        noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
    elif scheduler_name == "DEISMultistepScheduler":
        scheduler_config.pop("rescale_betas_zero_snr")
        noise_scheduler = DEISMultistepScheduler(**scheduler_config)
    elif scheduler_name == "EulerDiscreteScheduler":
        scheduler_config.update({"interpolation_type": "linear"})
        scheduler_config.pop("dynamic_thresholding_ratio")
        noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
    elif scheduler_name == "EulerAncestralDiscreteScheduler":
        scheduler_config.pop("dynamic_thresholding_ratio")
        noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
    elif scheduler_name == "UniPCMultistepScheduler":
        scheduler_config.pop("rescale_betas_zero_snr")
        noise_scheduler = UniPCMultistepScheduler(**scheduler_config)

    if noise_scheduler == None:
        scheduler_config.update({"interpolation_type": "linear"})
        scheduler_config.pop("dynamic_thresholding_ratio")
        noise_scheduler = EulerDiscreteScheduler(**scheduler_config)

    pipeline.scheduler = noise_scheduler
    generator = torch.Generator(device).manual_seed(seed)
    pipeline.unet.to(device)

    if latent_image is not None:
        samples_in = latent_image['samples']
        samples_in = samples_in * vae_scaling_factor
        samples_in = samples_in.to(pipeline.unet.dtype).to(device)

    latentWidth, latentHeigth = utility.getLatentSize(latent_image)

    latent_out = pipeline(
        prompt = None,
        latents = samples_in if latent_image is not None else None,
        prompt_embeds = positive['prompt_embeds'],
        pooled_prompt_embeds = positive['pooled_prompt_embeds'],
        negative_prompt_embeds = positive['negative_prompt_embeds'],
        negative_pooled_prompt_embeds = positive['negative_pooled_prompt_embeds'],
        height = latentHeigth * 8,
        width = latentWidth * 8,
        num_inference_steps = steps,
        guidance_scale = cfg,
        num_images_per_prompt = 1,
        generator = generator,
        strength = denoise,
    ).images

    pipeline.unet.to(offload_device)
    latent_out = latent_out / vae_scaling_factor
    return ({'samples': latent_out},)