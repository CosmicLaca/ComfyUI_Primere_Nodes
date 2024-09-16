import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.fft as fft
import random

class PowerLawNoise(nn.Module):
    def __init__(self, device='cpu'):
        super(PowerLawNoise, self).__init__()
        self.device = device
        
    @staticmethod
    def get_noise_types():
        return ["white", "blue", "brownian_fractal", "violet"]

    def get_generator(self, noise_type):
        if noise_type in self.get_noise_types():
            if noise_type == "white":
                return self.white_noise
            elif noise_type == "blue":
                return self.blue_noise
            elif noise_type == "violet":
                return self.violet_noise
            elif noise_type == "brownian_fractal":
                return self.brownian_fractal_noise
        else:
            raise ValueError(f"`noise_type` is invalid. Valid types are {', '.join(self.get_noise_types())}")

    def set_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)

    def white_noise(self, batch_size, width, height, scale, seed, alpha=0.0, **kwargs):
        self.set_seed(seed)
        scale = scale
        noise_real = torch.randn((batch_size, 1, height, width), device=self.device)
        noise_power_law = torch.sign(noise_real) * torch.abs(noise_real) ** alpha
        noise_power_law *= scale
        return noise_power_law.to(self.device)

    def blue_noise(self, batch_size, width, height, scale, seed, alpha=2.0, **kwargs):
        self.set_seed(seed)

        noise = torch.randn(batch_size, 1, height, width, device=self.device)
        
        freq_x = fft.fftfreq(width, 1.0)
        freq_y = fft.fftfreq(height, 1.0)
        Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
        
        power = (Fx**2 + Fy**2)**(alpha / 2.0)
        power[0, 0] = 1.0
        power = power.unsqueeze(0).expand(batch_size, 1, width, height).permute(0, 1, 3, 2).to(device=self.device)
        
        noise_fft = fft.fftn(noise)
        power = power.to(noise_fft)
        noise_fft = noise_fft / torch.sqrt(power)
        
        noise_real = fft.ifftn(noise_fft).real
        noise_real = noise_real - noise_real.min()
        noise_real = noise_real / noise_real.max()
        noise_real = noise_real * scale
        
        return noise_real.to(self.device)

    def violet_noise(self, batch_size, width, height, alpha=1.0, device='cpu', **kwargs):
        white_noise = torch.randn((batch_size, 1, height, width), device=device)
        violet_noise = torch.sign(white_noise) * torch.abs(white_noise) ** (alpha / 2.0)
        violet_noise /= torch.max(torch.abs(violet_noise))
        
        return violet_noise

    def brownian_fractal_noise(self, batch_size, width, height, scale, seed, alpha=1.0, modulator=1.0, **kwargs):
        def add_particles_to_grid(grid, particle_x, particle_y):
            for x, y in zip(particle_x, particle_y):
                grid[y, x] = 1

        def move_particles(particle_x, particle_y):
            dx = torch.randint(-1, 2, (batch_size, n_particles), device=self.device)
            dy = torch.randint(-1, 2, (batch_size, n_particles), device=self.device)
            particle_x = torch.clamp(particle_x + dx, 0, width - 1)
            particle_y = torch.clamp(particle_y + dy, 0, height - 1)
            return particle_x, particle_y

        self.set_seed(seed)
        n_iterations = int(5000 * modulator)
        fy = fft.fftfreq(height).unsqueeze(1) ** 2
        fx = fft.fftfreq(width) ** 2
        f = fy + fx
        power = torch.sqrt(f) ** alpha
        power[0, 0] = 1.0

        grid = torch.zeros(height, width, dtype=torch.uint8, device=self.device)

        n_particles = n_iterations // 10 
        particle_x = torch.randint(0, int(width), (batch_size, n_particles), device=self.device)
        particle_y = torch.randint(0, int(height), (batch_size, n_particles), device=self.device)

        neighborhood = torch.tensor([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=torch.uint8, device=self.device)

        for _ in range(n_iterations):
            add_particles_to_grid(grid, particle_x, particle_y)
            particle_x, particle_y = move_particles(particle_x, particle_y)

        brownian_tree = grid.clone().detach().float().to(self.device)
        brownian_tree = brownian_tree / brownian_tree.max()
        brownian_tree = F.interpolate(brownian_tree.unsqueeze(0).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False)
        brownian_tree = brownian_tree.squeeze(0).squeeze(0)

        fy = fft.fftfreq(height).unsqueeze(1) ** 2
        fx = fft.fftfreq(width) ** 2
        f = fy + fx
        power = torch.sqrt(f) ** alpha
        power[0, 0] = 1.0

        noise_real = brownian_tree * scale

        amplitude = 1.0 / (scale ** (alpha / 2.0))
        noise_real *= amplitude

        noise_fft = fft.fftn(noise_real.to(self.device))
        noise_fft = noise_fft / power.to(self.device)
        noise_real = fft.ifftn(noise_fft).real
        noise_real *= scale

        return noise_real.unsqueeze(0).unsqueeze(0)

    def forward(self, batch_size, width, height, alpha=2.0, scale=1.0, modulator=1.0, noise_type="white", seed=None):
        if noise_type not in self.get_noise_types():
            raise ValueError(f"`noise_type` is invalid. Valid types are {', '.join(self.get_noise_types())}")

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        channels = []
        for i in range(3):
            gen_seed = seed + i
            random.seed(gen_seed)
            noise = normalize(self.get_generator(noise_type)(batch_size, width, height, scale=scale, seed=gen_seed, alpha=alpha, modulator=modulator))
            channels.append(noise)

        noise_image = torch.cat((channels[0], channels[1], channels[2]), dim=1)
        noise_image = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min())
        noise_image = noise_image.permute(0, 2, 3, 1).float()

        return noise_image.to(device="cpu")

def normalize(latent, target_min=None, target_max=None):
    min_val = latent.min()
    max_val = latent.max()

    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val

    normalized = (latent - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled