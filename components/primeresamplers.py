import random
import nodes
import torch
import folder_paths
from ..components import latentnoise
from ..components import utility
from ..components import models
import comfy_extras.nodes_align_your_steps as nodes_align_your_steps
import comfy.samplers
import comfy_extras.nodes_custom_sampler as nodes_custom_sampler
import comfy_extras.nodes_stable_cascade as nodes_stable_cascade
import comfy_extras.nodes_flux as nodes_flux
import comfy_extras.nodes_model_advanced as nodes_model_advanced
import comfy_extras.nodes_hunyuan as nodes_hunyuan
from comfy import model_management
import gc
import os
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
              variation_extender, variation_batch_step_original, batch_counter, variation_extender_original, variation_batch_step, variation_level, variation_limit, align_your_steps, noise_extender, model_sampling=None, control_data=None):

    if model_sampling is not None and model_sampling > 0:
        model = nodes_model_advanced.ModelSamplingSD3.patch(self, model, model_sampling, 1.0)[0]

    samples = None

    if scheduler_name == 'beta' and control_data is not None:
        beta_alpha = float(control_data.get('beta_alpha', 0.5))
        beta_beta = float(control_data.get('beta_beta', 0.5))
        try:
            sigmas = comfy.samplers.beta_scheduler(model.get_model_object("model_sampling"), steps, alpha=beta_alpha, beta=beta_beta)
            sampler = comfy.samplers.sampler_object(sampler_name)
            samples = (nodes_custom_sampler.SamplerCustom.execute(model, True, seed, cfg, positive, negative, sampler, sigmas, latent_image)[0],)
        except Exception as e:
            print(f"Primere: BetaSamplingScheduler failed: {e}")

    if samples is None:
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
                        case 'SDXL':
                            model_type = 'SDXL'
                        case _:
                            model_type = 'SD1'
                    sigmas = nodes_align_your_steps.AlignYourStepsScheduler.get_sigmas(self, model_type, steps, denoise)
                    sampler = comfy.samplers.sampler_object(sampler_name)
                    AYS_samples = nodes_custom_sampler.SamplerCustom.execute(model, True, seed, cfg, positive, negative, sampler, sigmas[0], latent_image)
                    samples = (AYS_samples[0],)
                else:
                    samples = nodes.KSampler.sample(self, model, seed, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise=denoise)

    # if control_data and control_data.get('refiner') == True:
    #    samples = run_refiner_pass(self, None, positive, negative, samples[0], control_data, seed)

    return samples

def PTurboSampler(model, seed, cfg, positive, negative, latent_image, steps, denoise, sampler_name):
    sigmas = nodes_custom_sampler.SDTurboScheduler.execute(model, steps, denoise)
    sampler = comfy.samplers.sampler_object(sampler_name)
    turbo_samples = nodes_custom_sampler.SamplerCustom.execute(model, True, seed, cfg, positive, negative, sampler, sigmas[0], latent_image)
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
        if type(model[0]).__name__ == 'ModelPatcherDynamic' and type(model[1]).__name__ == 'ModelPatcherDynamic':
            c_latent = {"samples": torch.zeros([1, 16, height // compression, width // compression])}
            b_latent = {"samples": torch.zeros([1, 4, height // 4, width // 4])}

            if variation_level == True:
                samples_c = latentnoise.noisy_samples(model[1], device, steps, cfg, sampler_name, scheduler_name, positive, negative, c_latent, denoise, seed, noise_extender)[0]
            else:
                if variation_extender_original > 0 or device != 'DEFAULT' or variation_batch_step_original > 0:
                    samples_c = latentnoise.noisy_samples(model[1], device, steps, cfg, sampler_name, scheduler_name, positive, negative, c_latent, denoise, seed, noise_extender)[0]
                else:
                    samples_c = nodes.KSampler.sample(self, model[1], seed, steps, cfg, sampler_name, scheduler_name, positive, negative, c_latent, denoise=denoise)[0]
            conditining_c = nodes_stable_cascade.StableCascade_StageB_Conditioning.execute(positive, samples_c)[0]
            samples = nodes.KSampler.sample(self, model[0], seed, int(steps/2), 1.1, sampler_name, scheduler_name, conditining_c, negative, b_latent, denoise=denoise)

    return samples

def PSamplerHyper(self, extra_pnginfo, model, seed, steps, cfg, positive, negative, sampler_name, scheduler_name, latent_image, denoise, prompt, control_data):
    # WORKFLOWDATA = extra_pnginfo['workflow']['nodes']
    OriginalBaseModel = control_data['model_name'] # OriginalBaseModel = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualCKPT', 'base_model', prompt)
    fullpathFile = folder_paths.get_full_path('checkpoints', OriginalBaseModel)
    is_link = os.path.islink(str(fullpathFile))
    HyperSDSelector = None
    if is_link == True:
        HyperSDSelector = 'UNET'
    if HyperSDSelector == 'UNET':
        sigmas = utility.get_hypersd_sigmas(model)
        sampler = comfy.samplers.sampler_object(sampler_name)
        hyper_samples = nodes_custom_sampler.SamplerCustom.execute(model, True, seed, cfg, positive, negative, sampler, sigmas[0], latent_image)
        samples = (hyper_samples[0],)
    else:
        SamplingDiscreteResults = utility.TCDModelSamplingDiscrete(self, model, steps, scheduler_name, denoise, eta=0.8)
        model = SamplingDiscreteResults[0]
        sampler = SamplingDiscreteResults[1]
        sigmas = SamplingDiscreteResults[2]
        hyper_lora_samples = nodes_custom_sampler.SamplerCustom.execute(model, True, seed, cfg, positive, negative, sampler, sigmas, latent_image)
        samples = (hyper_lora_samples[0],)

    return samples


def PSamplerSana(self, device, seed, model,
                 steps, cfg, sampler_name, scheduler_name,
                 positive, negative,
                 latent_image, denoise,
                 variation_extender, variation_batch_step_original, batch_counter, variation_extender_original, variation_batch_step, variation_level, variation_limit, align_your_steps, noise_extender, WORKFLOWDATA, prompt):

    if device == 'DEFAULT':
        device = model_management.get_torch_device()

    pag_applied_layers = None if 'pag_applied_layers' not in model.keys() else model['pag_applied_layers']

    latent_out = model['pipe'](
        cond = positive,
        uncond = negative,
        guidance_scale = cfg,
        pag_guidance_scale = 2,
        num_inference_steps = (steps + 1),
        generator = torch.Generator(device=device).manual_seed(seed),
        latents = latent_image['samples'],
        noise_scheduler = scheduler_name,
        output_type = True,
        pag_applied_layers = pag_applied_layers,
    )

    model['unet'].to(comfy.model_management.unet_offload_device())
    comfy.model_management.soft_empty_cache(True)

    return (latent_out,)

def run_refiner_pass(self, refiner_model, refiner_cond_pos, refiner_cond_neg, samples_main, control_data, seed):
    has_direct_refiner_model = refiner_model is not None
    if control_data is not None and control_data.get('_refiner_applied') == True:
        return (samples_main,)

    refiner_enabled = control_data is not None and control_data.get('refiner') == True
    if not refiner_enabled and not has_direct_refiner_model:
        return (samples_main,)

    if control_data is None:
        return (samples_main,)

    refiner_model_name = control_data.get('refiner_model', None)
    if refiner_model_name in (None, 'None'):
        if not has_direct_refiner_model:
            return (samples_main,)

    model_concept = control_data.get('model_concept', None)
    refiner_sampler = control_data.get('refiner_sampler', 'dpmpp_2m')
    refiner_scheduler = control_data.get('refiner_scheduler', 'normal')
    refiner_cfg = float(control_data.get('refiner_cfg', 2.0))
    refiner_steps = int(control_data.get('refiner_steps', 22))
    refiner_denoise = float(control_data.get('refiner_denoise', 0.9))
    refiner_start = int(control_data.get('refiner_start', 12))
    refiner_ignore_prompt = control_data.get('refiner_ignore_prompt', True)

    try:
        main_vae = utility.vae_loader_class.load_vae(control_data.get('vae'))[0]
        raw_image = nodes.VAEDecode.decode(self, main_vae, samples_main)[0]

        if model_concept == 'HunyuanV2':
            if refiner_model is None:
                file_link, linked_filename, _ = models.resolve_symlink(refiner_model_name)
                unet_name = linked_filename if file_link else refiner_model_name
                refiner_model = nodes.UNETLoader.load_unet(self, unet_name, 'default')[0]

            refiner_vae_name = control_data.get('refiner_vae', 'Baked')
            refiner_vae = utility.vae_loader_class.load_vae(refiner_vae_name)[0] if refiner_vae_name and refiner_vae_name != 'Baked' else main_vae
            encoded_image = nodes.VAEEncode.encode(self, refiner_vae, raw_image)[0]

            ref_pos_prompt = '''<|start_header_id|>system<|end_header_id|>
    Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Make the image high quality
    <|eot_id|>'''

            ref_neg_prompt = '''<|start_header_id|>system<|end_header_id|>
    Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    <|eot_id|>'''

            if refiner_ignore_prompt:
                hunyuan_clip = control_data.get('loaded_clip', None)
                if hunyuan_clip is None:
                    raise ValueError("HunyuanV2 refiner requires 'loaded_clip' in control_data when 'refiner_ignore_prompt' is enabled.")
                ref_cond_pos = nodes.CLIPTextEncode.encode(self, hunyuan_clip, ref_pos_prompt)[0]
                ref_cond_neg = nodes.CLIPTextEncode.encode(self, hunyuan_clip, ref_neg_prompt)[0]
            else:
                ref_cond_pos = refiner_cond_pos
                ref_cond_neg = refiner_cond_neg

            noise_augmentation = float(control_data.get('noise_augmentation', 0.10))
            refiner_pos, refiner_neg, refiner_latent = nodes_hunyuan.HunyuanRefinerLatent.execute(ref_cond_pos, ref_cond_neg, encoded_image, noise_augmentation)
            refiner_output = nodes.KSampler.sample(self, refiner_model, seed, refiner_steps, refiner_cfg, refiner_sampler, refiner_scheduler, refiner_pos, refiner_neg, refiner_latent, denoise=refiner_denoise)
            control_data['_refiner_applied'] = True
            return (refiner_output[0],)

        refiner_ckpt = nodes.CheckpointLoaderSimple.load_checkpoint(self, refiner_model_name)
        if refiner_model is None:
            refiner_model = refiner_ckpt[0]

        refiner_vae_name = control_data.get('refiner_vae', 'Baked')
        if refiner_vae_name and refiner_vae_name != 'Baked':
            refiner_vae = utility.vae_loader_class.load_vae(refiner_vae_name)[0]
        else:
            refiner_vae = refiner_ckpt[2]

        encoded_image = nodes.VAEEncode.encode(self, refiner_vae, raw_image)[0]
        sigmas = nodes_custom_sampler.BasicScheduler.execute(refiner_model, refiner_scheduler, refiner_steps, refiner_denoise)[0]
        low_sigmas = nodes_custom_sampler.SplitSigmas.execute(sigmas, refiner_start)[1]
        sampler = comfy.samplers.sampler_object(refiner_sampler)
        if refiner_ignore_prompt:
            empty_cond = nodes.CLIPTextEncode.encode(self, refiner_ckpt[1], "")[0]
            pos_cond = empty_cond
            neg_cond = empty_cond
        else:
            pos_cond = refiner_cond_pos if refiner_cond_pos is not None else nodes.CLIPTextEncode.encode(self, refiner_ckpt[1], "")[0]
            neg_cond = refiner_cond_neg if refiner_cond_neg is not None else nodes.CLIPTextEncode.encode(self, refiner_ckpt[1], "")[0]

        refiner_output = nodes_custom_sampler.SamplerCustom.execute(refiner_model, True, seed, refiner_cfg, pos_cond, neg_cond, sampler, low_sigmas, encoded_image)
        control_data['_refiner_applied'] = True
        return (refiner_output[0],)
    except Exception as e:
        print(f"Primere: Refiner sampling failed: {e}")
        return (samples_main,)


def PSamplerPixart(self, device, seed, model,
                   steps, cfg, sampler_name, scheduler_name,
                   positive, negative,
                   latent_image, denoise,
                   variation_extender, variation_batch_step_original, batch_counter, variation_extender_original, variation_batch_step, variation_level, variation_limit, align_your_steps, noise_extender, control_data):

    if control_data:
        sampler_name = control_data.get('sampler_name', sampler_name)
        scheduler_name = control_data.get('scheduler_name', scheduler_name)
        steps = control_data.get('steps', steps)
        cfg = control_data.get('cfg', cfg)
    if control_data and control_data.get('refiner') == True:
        denoise = float(control_data.get('refiner_sampling_denoise', denoise))

    sigmas_main = nodes_custom_sampler.BasicScheduler.execute(model, scheduler_name, steps, denoise=denoise)[0]
    sampler = comfy.samplers.sampler_object(sampler_name)
    if variation_level == True:
        samples_main = latentnoise.noisy_samples(model, device, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise, seed, noise_extender)[0]
    else:
        if variation_extender_original > 0 or device != 'DEFAULT' or variation_batch_step_original > 0:
            samples_main = latentnoise.noisy_samples(model, device, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise, seed, noise_extender)[0]
        else:
            if align_your_steps == True:
                model_type = 'SDXL'
                sigmas = nodes_align_your_steps.AlignYourStepsScheduler.get_sigmas(self, model_type, steps, denoise)
                sampler = comfy.samplers.sampler_object(sampler_name)
                AYS_samples = nodes_custom_sampler.SamplerCustom.execute(model, True, seed, cfg, positive, negative, sampler, sigmas[0], latent_image)
                samples_main = AYS_samples[0]
            else:
                samples_main = nodes_custom_sampler.SamplerCustom.execute(model, True, seed, cfg, positive, negative, sampler, sigmas_main, latent_image)[0]

    return (samples_main,)

def PSamplerAdvanced(self, model, seed, guidance, positive, scheduler_name, sampler_name, steps, denoise, latent_image):
    if guidance <= 0:
        CONDITIONING_POS = positive
    else:
        CONDITIONING_POS = nodes_flux.FluxGuidance.execute(positive, guidance)[0]
    FLUX_GUIDER = nodes_custom_sampler.BasicGuider.execute(model, CONDITIONING_POS)[0]
    FLUX_SIGMAS = nodes_custom_sampler.BasicScheduler.execute(model, scheduler_name, steps, denoise=denoise)[0]
    FLUX_NOISE = nodes_custom_sampler.RandomNoise.execute(seed)[0]
    sampler_object = comfy.samplers.sampler_object(sampler_name)
    samples = (nodes_custom_sampler.SamplerCustomAdvanced.execute(FLUX_NOISE, FLUX_GUIDER, sampler_object, FLUX_SIGMAS, latent_image)[0],)
    return samples

def PSamplerChroma(self, model, seed, cfg, positive, negative, scheduler_name, sampler_name, steps, denoise, latent_image):
    guider = nodes_custom_sampler.CFGGuider.execute(model, positive, negative, cfg)[0]
    sigmas = nodes_custom_sampler.BasicScheduler.execute(model, scheduler_name, steps, denoise=denoise)[0]
    sampler = nodes_custom_sampler.KSamplerSelect.execute(sampler_name)[0]
    noise = nodes_custom_sampler.RandomNoise.execute(seed)[0]
    samples = (nodes_custom_sampler.SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image)[0],)
    return samples

def PSamplerSD3(self, model, seed, cfg, positive, negative, latent_image, steps, denoise, sampler_name, scheduler_name, model_sampling = 2.5, multiplier = 1000):
    sd3sampling = nodes_model_advanced.ModelSamplingSD3.patch(self, model, model_sampling, multiplier)[0]
    samples = nodes.KSampler.sample(self, sd3sampling, seed, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise=denoise)
    return samples

def PSamplerKOROLS(self, model, seed, cfg, positive, negative, latent_image, steps, denoise, sampler_name, scheduler_name, model_sampling = 0, multiplier = 1000):
    device = model_management.get_torch_device()    #"cuda" if torch.cuda.is_available() else "cpu"
    offload_device = model_management.unet_offload_device()
    
    vae_scaling_factor = 0.13025
    try:
        model_management.soft_empty_cache()
    except Exception:
        print('Cannot clear cache...')
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
    latentWidth, latentHeigth = utility.getLatentSize(latent_image)

    latent_out = pipeline(
        prompt = None,
        latents = None,  #samples_in if latent_image is not None else None,
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