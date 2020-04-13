from typing import Dict
import logging
from tqdm import tqdm
from denoizer.data.images_pipeline import get_dataset_pipeline
from denoizer.helpers.utils import validate_config, set_random_seed, load_model
from denoizer.data.utils.image_saving import save_image, transform_to_image

logger = logging.Logger(name=__file__)


def run_inference(config: Dict, save_location: str):
    set_random_seed(config=config)
    validate_config(config=config)
    modelIO_config = config['ModelIOConfig']
    data_pipeline_config = config['DataPipelineConfig']
    logger.info("Creating test dataset pipeline")
    test_dataset = get_dataset_pipeline(data_pipeline_config=data_pipeline_config, mode="test")
    logger.info("Loading pretrained model")
    model = load_model(modelIO_config=modelIO_config)
    logger.info("Running inference on test dataset")

    for i, (noisy_patch, gt_patch) in tqdm(enumerate(test_dataset)):
        predicted_denoized = model.predict(x=noisy_patch)
        save_image(image=transform_to_image(predicted_denoized),
                   save_location=save_location,
                   image_name=f"{i}_denoized")
        save_image(image=transform_to_image(gt_patch),
                   save_location=save_location,
                   image_name=f"{i}_gt")
        save_image(image=transform_to_image(noisy_patch),
                   save_location=save_location,
                   image_name=f"{i}_noisy")
