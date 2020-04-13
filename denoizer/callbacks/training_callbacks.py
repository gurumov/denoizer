import os
from datetime import datetime
from typing import Dict, Type, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from denoizer.callbacks.custom_callbacks import TensorBoardImage
from denoizer.helpers.utils import get_current_model_name, get_current_path, get_checkpoint_save_path, get_logs_dir, \
    get_update_frequency


def get_model_callbacks(modelIO_config: Dict, model: Type[tf.keras.Model],
                        test_example: Tuple[np.ndarray, np.ndarray]) -> List[Type[tf.keras.callbacks.Callback]]:
    current_model_name = get_current_model_name(modelIO_config=modelIO_config)
    checkpoint_save_path = get_current_path(basepath=get_checkpoint_save_path(modelIO_config=modelIO_config),
                                            current_model_name=current_model_name)
    log_dir = get_current_path(basepath=get_logs_dir(modelIO_config=modelIO_config),
                               current_model_name=current_model_name)
    update_frequency = get_update_frequency(modelIO_config=modelIO_config)
    loss_name = _get_loss_name(model=model)
    print(f'Estimated loss name is {loss_name}')

    callbacks = [TensorBoardImage(log_dir=log_dir, model=model, image_example=test_example),
                 ModelCheckpoint(filepath=checkpoint_save_path, verbose=1),
                 TensorBoard(log_dir=log_dir, update_freq=update_frequency),
                 ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, mode="min", min_delta=0.01),
                 EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode="min")]
    return callbacks


def _get_loss_name(model: Type[tf.keras.Model]) -> str:
    loss = model.loss_functions[0]
    return getattr(loss, "name", None) or getattr(loss, '__name__')
