import numpy as np
from typing import Tuple
import tensorflow as tf


class TensorBoardImage(tf.keras.callbacks.Callback):
    """
    A custom callback to store images to Tensorboard
    """
    def __init__(self, log_dir: str, model: tf.keras.models.Model,
                 image_example: Tuple[np.ndarray, np.ndarray]):
        """
        :param log_dir - The directory where the logs of the run are stored
        :param model - The model to be used for evaluation
        :param image_example - A test example, for which the model prediction will be plotted
        """
        self.log_dir = log_dir
        self.noisy_image, self.gt_image = image_example
        if self.noisy_image.shape != self.gt_image:
            self.single_noisy = self.noisy_image[0]
        else:
            self.single_noisy = self.noisy_image
        self.model = model
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        denoized_image = self.model.predict(self.noisy_image)

        noisy_image = _transform_to_image(tensor=self.single_noisy)
        gt_image = _transform_to_image(tensor=self.gt_image)
        denoized_image = _transform_to_image(tensor=denoized_image)

        writer = tf.summary.create_file_writer(self.log_dir)

        with writer.as_default():
            tf.summary.image(name="noisy", data=noisy_image, step=epoch)
            tf.summary.image(name="original", data=gt_image, step=epoch)
            tf.summary.image(name="denoized", data=denoized_image, step=epoch)
            writer.flush()


def _transform_to_image(tensor: tf.Tensor) -> np.ndarray:
    img = np.copy(np.asarray(tensor))
    img *= 255
    img = np.clip(img, 0.0, 255.0)
    img = img.astype(np.uint8)
    return img