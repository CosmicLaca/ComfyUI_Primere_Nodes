import hashlib
import difflib
import folder_paths
import comfy.samplers
import os
import numpy as np
from pathlib import Path

def get_model_hash(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()[0:10]

def check_model_from_exif(model_hash_exif, model_name_exif, model_name, model_hash_check):
    ckpt_path = folder_paths.get_full_path("checkpoints", model_name_exif)
    if ckpt_path is not None:
        if os.path.exists(ckpt_path):
            return model_name_exif

    checkpointpaths = folder_paths.get_folder_paths("checkpoints")[0]
    allcheckpoints = folder_paths.get_filename_list("checkpoints")

    source_model_name = model_name_exif
    if '\\\\' not in model_name_exif:
        source_model_name = model_name_exif.split('_', 1)[-1]

    cutoff_list = list(np.around(np.arange(0.1, 1.05, 0.05).tolist(), 2))[::-1] # [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    is_found = []
    for trycut in cutoff_list:
        is_found = difflib.get_close_matches(source_model_name, allcheckpoints, cutoff=round(trycut, 2))
        if len(is_found) == 1:
            break

    if len(is_found) != 1:
        for trycut in cutoff_list:
            is_found = difflib.get_close_matches(model_name_exif, allcheckpoints, cutoff=round(trycut, 2))
            if len(is_found) == 1:
                break

    if len(is_found) == 1:
        valid_model = is_found[0]
        model_full_path = checkpointpaths + os.sep + valid_model

        if model_hash_check == True:
            match_model_hash = get_model_hash(model_full_path)
            if match_model_hash == model_hash_exif:
                model_name = valid_model
            else:
                print('Model name:' + model_name_exif + ' not available by hashcheck, using system source: ' + model_name)
        else:
            model_name = valid_model
    else:
        if len(is_found) == 0:
            print('Model name:' + model_name_exif + ' not available by diffcheck, using system source: ' + model_name)
        if len(is_found) > 0:
            print('Model name:' + model_name_exif + ' more than one results by diffcheck, using first found: ' + str(is_found[0]))
            model_name = is_found[0]

    return model_name


def change_exif_samplers(sampler_name_exif, comfy_schedulers):
    lastchars = sampler_name_exif[-2:]
    if lastchars == ' a':
        sampler_name_exif = sampler_name_exif.rsplit(' a', 1)[0] + ' ancestral'

    sampler_name_exif = sampler_name_exif.replace(' a ', ' ancestral ').replace(' ', '_').replace('++', 'pp').replace('dpm2', 'dpm_2').replace('unipc', 'uni_pc')

    for comfy_scheduler in comfy_schedulers:
        sampler_name_exif = sampler_name_exif.removesuffix(comfy_scheduler).removesuffix('_')

    return sampler_name_exif


def check_sampler_from_exif(sampler_name_exif, sampler_name, scheduler_name):
    comfy_samplers = comfy.samplers.KSampler.SAMPLERS
    comfy_schedulers = comfy.samplers.KSampler.SCHEDULERS

    sampler_name_exif_for_cutoff = change_exif_samplers(sampler_name_exif, comfy_schedulers)
    is_found_sampler = []
    is_found_scheduler = []

    cutoff_list_samplers = list(np.around(np.arange(0.6, 1.05, 0.1).tolist(), 2))[::-1]
    for trycut in cutoff_list_samplers:
        is_found_sampler = difflib.get_close_matches(sampler_name_exif_for_cutoff, comfy_samplers, cutoff=trycut)

    if len(is_found_sampler) >= 1:
        sampler_name = is_found_sampler[0]

    if " " in sampler_name_exif:
        if any((match := substring) in sampler_name_exif for substring in comfy_schedulers):
            scheduler_name = match
        else:
            cutoff_list_schedulers = list(np.around(np.arange(0.4, 0.8, 0.1).tolist(), 2))[::-1]
            for trycut in cutoff_list_schedulers:
                is_found_scheduler = difflib.get_close_matches(sampler_name_exif, comfy_schedulers, cutoff=trycut)

    if len(is_found_scheduler) >= 1:
        scheduler_name = is_found_scheduler[0]

    if sampler_name not in comfy_samplers:
        sampler_name = comfy_samplers[0]

    if scheduler_name not in comfy_schedulers:
        scheduler_name = comfy_schedulers[0]

    return {'sampler': sampler_name, 'scheduler': scheduler_name}

def comfy_samplers2a11(comfy_sampler, comfy_scheduler):
  sampler_name_a11 = comfy_sampler.capitalize().replace('_', ' ').replace('pp', '++').replace('Dpm 2', 'DPM2').replace('Uni pc', 'UniPC').replace(' ancestral', ' a') + ' ' + comfy_scheduler.capitalize()
  sampler_name_a11 = sampler_name_a11.replace(' Normal', '')
  sampler_name_a11 = sampler_name_a11.replace('Ddim', 'DDIM').replace('Dpm', 'DPM').replace('2m', '2M').replace('sde', 'SDE').replace('2s', '2S').replace('Lms', 'LMS')

  return  sampler_name_a11

def check_vae_exif(vae_name_exif, vae_name):
    comfy_vaes = folder_paths.get_filename_list("vae")
    cutoff_list = list(np.around(np.arange(0.7, 1.05, 0.1).tolist(), 2))[::-1]
    is_found = []

    for trycut in cutoff_list:
        is_found = difflib.get_close_matches(vae_name_exif, comfy_vaes, cutoff=trycut)
        if len(is_found) == 1:
            break

    if len(is_found) == 1:
        vae_name = is_found[0]

    return vae_name