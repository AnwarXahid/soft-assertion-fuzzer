import torch
import tensorflow as tf


def is_safe_torch(x, y):
    if torch.isinf(x).any() or torch.isnan(x).any() or torch.isinf(y).any() or torch.isnan(y).any():
        return False

    unstable_matmul = torch.matmul(x, y)
    if torch.isnan(unstable_matmul).any() or torch.isinf(unstable_matmul).any() or torch.isneginf(
            unstable_matmul).any():
        return False

    return True


def is_safe_tf(x, y):
    if tf.reduce_any(tf.math.is_inf(x)) or tf.reduce_any(tf.math.is_nan(x)) or tf.reduce_any(
            tf.math.is_inf(y)) or tf.reduce_any(tf.math.is_nan(y)):
        return False

    unstable_matmul = tf.matmul(x, y)

    if tf.reduce_any(tf.math.is_nan(unstable_matmul)) or tf.reduce_any(
            tf.math.is_inf(unstable_matmul)):
        return False

    return True
