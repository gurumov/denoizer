from typing import Dict, Type
import os
import logging
import tensorflow as tf
from datetime import datetime

logger = logging.Logger(name=__file__)


def validate_config(config: Dict):
    assert 'DataPipelineConfig' in config, "The config file is missing a DataPipelineConfig"
    assert 'ModelConfig' in config, "The config file is missing a ModelConfig"
    assert 'CallbacksConfig' in config, "The config file is missing a CallbacksConfig"


def set_random_seed(config: Dict):
    seed = int(config.get('seed', 123))
    tf.random.set_seed(seed=seed)


def load_model(modelIO_config: Dict) -> Type[tf.keras.Model]:
    model_name = modelIO_config.get('model_name', "")
    assert model_name.strip() != "", "No model name provided, can't load a model!"
    model_load_path = get_current_path(basepath=get_checkpoint_save_path(modelIO_config=modelIO_config),
                                       current_model_name=model_name)
    try:
        model = tf.keras.models.load_model(model_load_path)
    except Exception as e:
        logger.error(f"Couldn't load a model from path {model_load_path}")
        raise e
    return model


def save_model(model: tf.keras.Model, modelIO_config: Dict):
    current_model_name = get_current_model_name(modelIO_config=modelIO_config)
    checkpoint_save_path = get_current_path(basepath=get_checkpoint_save_path(modelIO_config=modelIO_config),
                                            current_model_name=current_model_name)
    with open(checkpoint_save_path, 'w') as f:
        model.save(f, include_optimizer=False)


def get_current_model_name(modelIO_config: Dict):
    model_name = modelIO_config.get("model_name", "")
    model_name = datetime.now().strftime('%m-%d-%Y-%H-%M-%S') if model_name == "" else model_name
    return model_name


def get_current_path(basepath: str, current_model_name: str):
    return os.path.join(basepath, current_model_name)


def get_checkpoint_save_path(modelIO_config: Dict) -> str:
    return modelIO_config.get('checkpoint_save_path', './checkpoints')


def get_logs_dir(modelIO_config: Dict) -> str:
    return modelIO_config.get('log_dir', './logs')


def get_update_frequency(modelIO_config: Dict) -> int:
    return int(modelIO_config.get("log_update_frequency", 100))