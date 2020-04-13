import numpy as np


def apply_gaussian_noise(patches: np.ndarray, max_sigma: float = 1.0) -> np.ndarray:
    mu = 0.0
    noise = np.random.normal(loc=mu, scale=max_sigma,  size=patches.shape)
    noisy = patches + noise
    return np.clip(noisy, a_min=0.0, a_max=255.0)


def normalize_patches(images: np.ndarray) -> np.ndarray:
    normalized_patches = images / 255.0
    normalized_patches = np.expand_dims(normalized_patches, axis=-1)
    return normalized_patches

