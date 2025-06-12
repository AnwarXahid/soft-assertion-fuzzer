import torch
import tensorflow as tf


def is_safe_torch(x):
    """
    Checks if a torch.Tensor is safe for unstable version of log.

    A torch.Tensor is considered safe if after applying unstable version of log,
    it does not contain any NaN or INF or -INF values.

    Args:
        x: A torch.Tensor.

    Returns:
        True if x is safe for unstable_log, False otherwise.
    """

    if torch.isinf(x).any() or torch.isnan(x).any() or torch.isneginf(x).any():
        return False

    unstable_log = torch.log(x)

    if torch.isnan(unstable_log).any() or torch.isinf(unstable_log).any() or (torch.isneginf(unstable_log).any()):
        return False
    return True


def is_safe_tf(x):
    if tf.reduce_any(tf.math.is_inf(x)) or tf.reduce_any(tf.math.is_nan(x)) or tf.reduce_any(x == float('-inf')):
        return False

    log_x = tf.math.log(x)

    if tf.reduce_any(tf.math.is_nan(log_x)) or tf.reduce_any(tf.math.is_inf(log_x)) or tf.reduce_any(
            log_x == float('-inf')):
        return False

    return True
