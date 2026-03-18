import numpy as np
import io
from PIL import Image
from PIL.ImageFilter import UnsharpMask
from scipy.ndimage import uniform_filter


def add_film_grain(arr: np.ndarray, intensity: float = 6.5) -> np.ndarray:
    arr = arr.astype(np.float32)
    luminance = np.mean(arr, axis=-1, keepdims=True) / 255.0
    grain = np.random.normal(0, intensity, arr.shape[:2])[..., np.newaxis]
    grain *= (0.4 + 0.8 * (1 - luminance))
    color_shift = np.random.normal(0, intensity * 0.3, (1, 1, 3))
    return np.clip(arr + grain * color_shift, 0, 255).astype(np.uint8)


def perturb_frequency(arr: np.ndarray, strength: float = 0.019) -> np.ndarray:
    arr = arr.astype(np.float32)
    result = np.zeros_like(arr)
    for c in range(3):
        f = np.fft.fft2(arr[:, :, c])
        fshift = np.fft.fftshift(f)
        rows, cols = fshift.shape
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((y - rows//2)**2 + (x - cols//2)**2)
        dist_norm = np.clip(dist / (max(rows, cols) / 2), 0, 1)
        high_freq_mask = dist_norm ** 2
        noise = np.random.normal(0, strength, fshift.shape) * high_freq_mask
        fshift_pert = fshift + noise + 1j * noise
        ch_pert = np.real(np.fft.ifft2(np.fft.ifftshift(fshift_pert)))
        result[:, :, c] = ch_pert
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_local_variance(arr: np.ndarray, strength: float = 0.32) -> np.ndarray:
    arr = arr.astype(np.float32)
    mean = uniform_filter(arr, size=5, mode='reflect')
    var = uniform_filter(arr**2, size=5, mode='reflect') - mean**2
    std_local = np.sqrt(np.clip(var, 1e-8, None))
    noise = np.random.normal(0, strength, arr.shape)
    noise *= (std_local / (np.mean(std_local) + 1e-5))
    return np.clip(arr + noise, 0, 255).astype(np.uint8)


def add_chromatic_aberration(arr: np.ndarray, strength: float = 1.2) -> np.ndarray:
    arr = arr.copy().astype(np.float32)
    shift = int(strength)
    arr[:,:,0] = np.roll(arr[:,:,0], -shift//2, axis=1)
    arr[:,:,2] = np.roll(arr[:,:,2],  shift//2, axis=1)
    return np.clip(arr, 0, 255).astype(np.uint8)


def add_vignette(arr: np.ndarray, strength: float = 0.18) -> np.ndarray:
    h, w = arr.shape[:2]
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - w/2)**2 + (y - h/2)**2)
    max_dist = np.sqrt((w/2)**2 + (h/2)**2)
    vignette = 1 - strength * (dist / max_dist)**2
    vignette = np.clip(vignette[..., np.newaxis], 0.75, 1.0)
    return np.clip(arr * vignette, 0, 255).astype(np.uint8)


def apply_jpeg_cycles(img: Image.Image, quality: int = 92, cycles: int = 3) -> Image.Image:
    for _ in range(cycles):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True, subsampling=0)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def bypass_ai_detector(
    image:             Image.Image,
    grain_intensity:   float = 6.5,
    freq_strength:     float = 0.019,
    variance_strength: float = 0.32,
    ca_strength:       float = 1.2,
    vignette_strength: float = 0.18,
    unsharp_percent:   int   = 38,
    jpeg_quality:      int   = 92,
    jpeg_cycles:       int   = 3,
) -> Image.Image:

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    arr = add_film_grain(arr, intensity=grain_intensity)
    arr = perturb_frequency(arr, strength=freq_strength)
    arr = adjust_local_variance(arr, strength=variance_strength)
    arr = add_chromatic_aberration(arr, strength=ca_strength)
    arr = add_vignette(arr, strength=vignette_strength)

    edited = Image.fromarray(arr.astype(np.uint8))
    edited = edited.filter(UnsharpMask(radius=0.75, percent=unsharp_percent, threshold=0))

    if jpeg_cycles > 0:
        edited = apply_jpeg_cycles(edited, quality=jpeg_quality, cycles=jpeg_cycles)

    return edited
