import os
import argparse
import yaml
import logging

from denoizer.interface.training import run_training
from denoizer.interface.inference import run_inference

logger = logging.Logger(name=__file__)

IMAGE_MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs/image_denoizing_config.yaml")


def instantiate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--save_to", dest='save_location', type=str, default="./inference_results")
    return parser


if __name__ == "__main__":
    parser = instantiate_parser()
    arguments = parser.parse_args()

    config = yaml.load(open(IMAGE_MODEL_CONFIG_PATH, 'r'), Loader=yaml.Loader)

    if arguments.train:
        run_training(config=config)
    elif arguments.test:
        save_location = arguments.save_location
        logger.info(f"Inference results will be saved at {save_location}")
        run_inference(config=config, save_location=save_location)
    else:
        logger.error("One of the flags '--train' or '--test' should be used with this script! Exiting!")