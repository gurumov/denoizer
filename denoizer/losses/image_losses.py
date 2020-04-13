import tensorflow as tf


def custom_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    flatten_true = tf.squeeze(y_true, axis=-1)
    flatten_pred = tf.squeeze(y_pred, axis=-1)
    absolute_difference = tf.abs(flatten_true - flatten_pred)
    partial_error = tf.reduce_sum(absolute_difference, axis=-1)
    error_per_patch = tf.reduce_sum(partial_error, axis=-1)
    return tf.reduce_mean(error_per_patch)

