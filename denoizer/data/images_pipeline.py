from typing import Dict, Tuple
import tensorflow as tf
from denoizer.data.utils.image_loading import generate_examples


def get_dataset_pipeline(data_pipeline_config: Dict[str, Dict[str, str]], mode: str, example_repetitions: int = 1) \
        -> tf.data.Dataset:
    if mode.lower() == "test":
        assert 'TestPipeline' in data_pipeline_config, "The config file is missing a TestPipeline definition"
        pipeline_config = data_pipeline_config["TestPipeline"]
    elif mode.lower() == "training":
        assert 'TrainingPipeline' in data_pipeline_config, "The config file is missing a TrainingPipeline definition"
        pipeline_config = data_pipeline_config["TrainingPipeline"]
    else:
        raise ValueError(f"Unknown mode selected - {mode}")

    assert 'images_path' in pipeline_config, f"The {mode} pipeline config is missing a images_path property"
    images_path = pipeline_config["images_path"]

    patch_size = int(pipeline_config.get("patch_size", 256))
    batch_size = int(pipeline_config.get("batch_size", 1))
    max_sigma = float(pipeline_config.get("max_sigma", 1.0))

    noisy_images, gt_images = generate_examples(images_path=images_path,
                                                patch_size=patch_size,
                                                max_sigma=max_sigma,
                                                example_repetitions=example_repetitions)

    pipeline = tf.data.Dataset.from_tensor_slices((noisy_images, gt_images))
    if mode.lower() == "training":
        pipeline = pipeline.shuffle(reshuffle_each_iteration=True, buffer_size=len(noisy_images))
    pipeline = pipeline.prefetch(batch_size)
    pipeline = pipeline.batch(batch_size, drop_remainder=True)
    pipeline = pipeline.prefetch(tf.data.experimental.AUTOTUNE)
    return pipeline
