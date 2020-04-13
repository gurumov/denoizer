import numpy as np
import logging
from typing import Dict, List, Type
from tensorflow.keras import optimizers, metrics, losses
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, BatchNormalization
from denoizer.losses import image_losses


logger = logging.Logger(name=__file__)


def resolve_model_metrics(metrics_list: List[str]) -> List[str]:
    resolved_metrics = []
    for metric in metrics_list:
        if metric in dir(metrics):
            resolved_metrics.append(metric)
        else:
            logger.warning(f"Couldn't get a metric named {metric}!")
    return resolved_metrics


def resolve_loss(loss_name: str) -> Type[losses.Loss]:
    if loss_name in dir(image_losses):
        loss = getattr(image_losses, loss_name)
    elif loss_name in dir(losses):
        loss = getattr(losses, loss_name)
    else:
        logger.warning(f"Couldn't get a loss named {loss_name}!")
        loss = losses.MSE
    logger.info(f"Building model with {loss.__name__} loss.")
    return loss


def resolve_optimizer(optimizer_name: str) -> Type[optimizers.Optimizer]:
    if optimizer_name in dir(optimizers):
        optimizer = getattr(optimizers, optimizer_name)
    else:
        logger.warning(f"Couldn't get an optimizer named {optimizer_name}! Will use Adam.")
        optimizer = optimizers.Adam
    return optimizer


def create_multiple_image_denoizing_model(model_config: Dict) -> Sequential:
    optimizer_name = model_config.get('optimizer', 'Adam')
    learning_rate = float(model_config.get('learning_rate', 0.001))
    loss_name = model_config.get('loss_name', 'MAE')
    metrics_list = model_config.get('metrics_list', ['MAE'])
    dropout_rate = float(model_config.get('dropout_rate', 0.0))

    simple_model = Sequential(layers=[
        Conv2D(filters=32, kernel_size=7, padding="SAME", activation="relu", input_shape=(None, None, 1)),
        BatchNormalization(),
        Dropout(rate=dropout_rate),
        Conv2D(filters=64, kernel_size=3, padding="SAME", activation="relu"),
        BatchNormalization(),
        Dropout(rate=dropout_rate),
        Conv2D(filters=32, kernel_size=3, padding="SAME", activation="relu"),
        BatchNormalization(),
        Dropout(rate=dropout_rate),
        Conv2D(filters=1, kernel_size=1, padding="SAME", activation="relu")
    ])

    optimizer = resolve_optimizer(optimizer_name=optimizer_name)
    loss = resolve_loss(loss_name=loss_name)
    resolved_metrics = resolve_model_metrics(metrics_list=metrics_list)
    simple_model.compile(optimizer=optimizer(lr=learning_rate), loss=loss, metrics=resolved_metrics)
    return simple_model


    # model = Sequential(layers=[
    #     Conv2D(filters=32, kernel_size=7, padding="SAME", activation="relu", input_dim=(None, None)),
    #     BatchNormalization(),
    #     Conv2D(filters=32, kernel_size=3, padding="SAME", activation="relu"),
    #     BatchNormalization(),
    #     Conv2D(filters=64, kernel_size=7, padding="SAME", stride=2, activation="relu"),
    #     BatchNormalization(),
    #     Conv2D(filters=64, kernel_size=3, padding="SAME", activation="relu"),
    #     BatchNormalization(),
    # ])