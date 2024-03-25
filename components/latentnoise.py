import numpy as np
import torch
import comfy.samplers
import random

def prepare_noise(latent_image, seed, noise_inds=None, noise_device="cpu", variation_seed=None, variation_strength=None):
    latent_size = latent_image.size()
    latent_size_1batch = [1, latent_size[1], latent_size[2], latent_size[3]]

    if variation_strength is not None and variation_strength > 0:
        if noise_device == "cpu":
            variation_generator = torch.manual_seed(variation_seed)
        else:
            torch.cuda.manual_seed(variation_seed)
            variation_generator = None

        variation_latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout, generator=variation_generator, device=noise_device)
    else:
        variation_latent = None

    def apply_variation(input_latent):
        if variation_latent is None:
            return input_latent
        else:
            strength = variation_strength

            variation_noise = variation_latent.expand(input_latent.size()[0], -1, -1, -1)
            result = (1 - strength) * input_latent + strength * variation_noise
            return result

    if noise_device == "cpu":
        generator = torch.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
        generator = None

    if noise_inds is None:
        latents = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device=noise_device)
        latents = apply_variation(latents)
        return latents

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device=noise_device)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

def noisy_samples(model, device, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise, seed, variation_extender):
    torch_device = comfy.model_management.get_torch_device()
    noise_device = "cpu" if device == "CPU" else torch_device
    batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None

    random.seed(seed)
    variation_seed = random.randint(0, 0xffffffffffffffff)

    noise = prepare_noise(latent_image["samples"], seed, batch_inds, noise_device, variation_seed=variation_seed, variation_strength=variation_extender)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        preview_bytes = None
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image["samples"], denoise=denoise, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, callback=callback, disable_pbar=False, seed=seed)
    out = latent_image.copy()
    out["samples"] = samples
    return (out,)

