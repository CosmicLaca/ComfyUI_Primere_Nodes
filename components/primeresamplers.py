import random
import nodes
import torch
import datetime
from ..components import latentnoise
from ..components import utility
import comfy_extras.nodes_align_your_steps as nodes_align_your_steps
import comfy.samplers
import comfy_extras.nodes_custom_sampler as nodes_custom_sampler
import comfy_extras.nodes_stable_cascade as nodes_stable_cascade
import comfy_extras.nodes_flux as nodes_flux

def PKSampler(self, device, seed, model,
              steps, cfg, sampler_name, scheduler_name,
              positive, negative,
              latent_image, denoise,
              variation_extender, variation_batch_step_original, batch_counter, variation_extender_original, variation_batch_step, variation_level, variation_limit, align_your_steps, noise_extender):

    '''if variation_batch_step_original > 0:
        if batch_counter > 0:
            variation_batch_step = variation_batch_step_original * batch_counter
        variation_extender = round(variation_extender_original + variation_batch_step, 2)'''

    if variation_level == True:
        samples = latentnoise.noisy_samples(model, device, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise, seed, noise_extender)
    else:
        if variation_extender_original > 0 or device != 'DEFAULT' or variation_batch_step_original > 0:
            '''if (variation_extender > 1):
                random.seed(batch_counter)
                noise_extender_low = round(random.uniform(0.00, variation_limit), 2)
                noise_extender_high = round(random.uniform((1 - variation_limit), 1), 2)
                variation_extender = random.choice([noise_extender_low, noise_extender_high])
            if variation_batch_step == 0:
                variation_seed = batch_counter + seed
            else:
                variation_seed = seed'''
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

    '''if variation_batch_step_original > 0:
        if batch_counter > 0:
            variation_batch_step = variation_batch_step_original * batch_counter
        variation_extender = round(variation_extender_original + variation_batch_step, 2)'''

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
                    '''if (variation_extender > 1):
                        random.seed(batch_counter)
                        variation_extender = round(random.uniform((1 - variation_limit), 1), 2)
                    if variation_batch_step == 0:
                        variation_seed = batch_counter + seed
                    else:
                        variation_seed = seed'''
                    samples_c = latentnoise.noisy_samples(model[1], device, steps, cfg, sampler_name, scheduler_name, positive, negative, c_latent, denoise, seed, noise_extender)[0]
                else:
                    samples_c = nodes.KSampler.sample(self, model[1], seed, steps, cfg, sampler_name, scheduler_name, positive, negative, c_latent, denoise=denoise)[0]
            conditining_c = nodes_stable_cascade.StableCascade_StageB_Conditioning.set_prior(self, positive, samples_c)[0]
            samples = nodes.KSampler.sample(self, model[0], seed, 10, 1.00, sampler_name, scheduler_name, conditining_c, negative, b_latent, denoise=denoise)

    return samples

def PSamplerHyper(self, extra_pnginfo, model, seed, steps, cfg, positive, negative, sampler_name, scheduler_name, latent_image, denoise):
    WORKFLOWDATA = extra_pnginfo['workflow']['nodes']

    HyperSDSelector = utility.getDataFromWorkflow(WORKFLOWDATA, 'PrimereModelConceptSelector', 8)
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

def PSamplerAdvanced(self, model, seed, WORKFLOWDATA, positive, scheduler_name, sampler_name, steps, denoise, latent_image):
    FLUX_GUIDANCE = float(utility.getDataFromWorkflow(WORKFLOWDATA, 'PrimereModelConceptSelector', 21))
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