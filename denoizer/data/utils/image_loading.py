from typing import List, Tuple
import os
from PIL import Image
import numpy as np
import logging

from denoizer.data.utils.image_augmentations import normalize_patches, apply_gaussian_noise

logger = logging.Logger(name=__file__)


def generate_examples(images_path: str, patch_size: int, max_sigma: float = 1.0, example_repetitions: int = 1)\
        -> Tuple[np.ndarray, np.ndarray]:
    image_names = os.listdir(images_path)
    image_paths = [os.path.join(images_path, img_name) for img_name in image_names
                   if (img_name.endswith("png") or img_name.endswith("jpg"))]
    image_patches = _load_image_patches(image_paths=image_paths,
                                        patch_size=patch_size,
                                        example_repetitions=example_repetitions)
    noisy_patches = apply_gaussian_noise(patches=image_patches, max_sigma=max_sigma)
    normalized_patches = normalize_patches(images=image_patches)
    normalized_noisy_patches = normalize_patches(images=noisy_patches)
    return normalized_noisy_patches, normalized_patches


def _load_image_patches(image_paths: List[str], patch_size: int, example_repetitions: int = 1) -> np.ndarray:
    image_patches = []
    for img_path in image_paths:
        img = np.asarray(Image.open(img_path).convert('L'))
        img_width, img_height = img.shape
        if img_width >= patch_size and img_height >= patch_size:
            crop = _extract_img_crop(img=img, patch_size=patch_size)
            for i in range(example_repetitions):
                image_patches.append(crop)
        else:
            logger.info(f"Image at {img_path} is smaller than the defined patch size! Skipping!")

    if example_repetitions > 1:
        return np.asarray(image_patches).reshape((-1, example_repetitions, patch_size, patch_size))
    else:
        return np.asarray(image_patches)


def _extract_img_crop(img: np.ndarray, patch_size: int) -> np.ndarray:
    width, height = img.shape
    x_offset = np.random.randint(0, width - patch_size)
    y_offset = np.random.randint(0, height - patch_size)
    return img[x_offset: x_offset + patch_size, y_offset: y_offset + patch_size]
