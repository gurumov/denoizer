import tensorflow as tf
from PIL import Image
import numpy as np
import os


def transform_to_image(tensor: tf.Tensor) -> Image:
    img = np.copy(np.asarray(tensor))
    img *= 255
    img = np.clip(img, 0.0, 255.0)
    img = img.astype(np.uint8)
    img = np.squeeze(img, axis=-1)
    img = np.squeeze(img, axis=0)
    return Image.fromarray(img)


def save_image(image: Image, save_location: str, image_name: str):
    os.makedirs(save_location, exist_ok=True)
    img_path = f"{save_location}/{image_name}.png"
    image.save(open(img_path, 'wb'), "PNG")