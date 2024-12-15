import torch
from comfy import model_management
import numpy as np

device = model_management.get_torch_device()
vae_dtype = model_management.vae_dtype(device, [torch.float16, torch.bfloat16, torch.float32])

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
class first_stage_model:
    def __init__(self, vae):
        self.vae = vae

    @torch.inference_mode()
    def encode(self, image):
        self.vae.to(device, vae_dtype)
        image = (image * 2.0 - 1).permute(0, 3, 1, 2)
        latent = self.vae.encode(image.to(device, vae_dtype))
        latent = latent * self.vae.cfg.scaling_factor
        self.vae.to(model_management.vae_offload_device())
        return latent

    @torch.inference_mode()
    def decode(self, latent):
        from diffusers.image_processor import PixArtImageProcessor
        vae_scale_factor = 2 ** (len(self.vae.cfg.encoder.width_list) - 1)
        image_processor = PixArtImageProcessor(vae_scale_factor=vae_scale_factor)
        self.vae.to(device, vae_dtype)
        # with torch.inference_mode():
        result = self.vae.decode(latent.to(device, vae_dtype).detach() / self.vae.cfg.scaling_factor)
        result = image_processor.postprocess(result.cpu().float())
        results = []
        for img in result:
            results.append(pil2tensor(img))
            # results.append((img / 2 + 1).clamp(-1, 1).cpu().float())
        result = torch.cat(results, dim=0)
        self.vae.to(model_management.vae_offload_device())
        return result

class cond_stage_model:
    def __init__(self, tokenizer, text_encoder):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    @torch.no_grad
    def tokenize(self, text):
        tokens = self.tokenizer(
            text,
            max_length=300,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.text_encoder.device)

        return tokens

    def encode_from_tokens_scheduled(self, tokens):
        self.text_encoder.to(device)
        cond = self.text_encoder(tokens.input_ids, tokens.attention_mask)[0]
        emb_masks = tokens.attention_mask.to(self.text_encoder.device)
        cond = cond * emb_masks.unsqueeze(-1)
        self.text_encoder.to(model_management.text_encoder_offload_device())

        return [[cond, {"emb_masks": emb_masks}]]