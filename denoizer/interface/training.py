from typing import Dict
from logging import Logger
from denoizer.data.images_pipeline import get_dataset_pipeline
from denoizer.models.single_image_model import create_single_image_denoizing_model
from denoizer.models.recurrent_image_model import create_multiple_image_denoizing_model
from denoizer.callbacks.training_callbacks import get_model_callbacks
from denoizer.helpers.utils import validate_config, set_random_seed, save_model

logger = Logger(name=__file__)


def run_training(config: Dict):
    set_random_seed(config=config)
    validate_config(config=config)
    data_pipeline_config = config['DataPipelineConfig']
    model_config = config['ModelConfig']
    modelIO_config = config['ModelIOConfig']
    example_repetitions = data_pipeline_config.get('example_repetitions', 1)

    logger.info("Creating training dataset pipeline")
    training_dataset = get_dataset_pipeline(data_pipeline_config=data_pipeline_config,
                                            mode="training",
                                            example_repetitions=example_repetitions)
    logger.info("Creating test dataset pipeline")
    test_dataset = get_dataset_pipeline(data_pipeline_config=data_pipeline_config,
                                        mode="test",
                                        example_repetitions=example_repetitions)
    if example_repetitions > 1:
        logger.info("Creating a single image model")
        model = create_single_image_denoizing_model(model_config=model_config)
    else:
        logger.info("Creating a multi image model")
        model = create_multiple_image_denoizing_model(model_config=model_config)

    test_example = [e for e in test_dataset.take(1)][0]
    callbacks = get_model_callbacks(modelIO_config=modelIO_config,
                                    model=model,
                                    test_example=test_example)
    model.summary()
    logger.info("Running training")
    model.fit(training_dataset,
              shuffle=False,
              epochs=500,
              callbacks=callbacks,
              validation_data=test_dataset)

    logger.info("Training finished! Saving model")
    save_model(model=model, modelIO_config=modelIO_config)



