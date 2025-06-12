import torch
import tensorflow as tf


def unstable_softmax_torch(x):
    """
    Computes the softmax of an input vector.

    Args:
        x: A torch.Tensor of any shape.

    Returns:
        A torch.Tensor of the same shape as x, containing the softmax of each element in x.

    Notes:
        This function is numerically unstable, and can lead to NaN or INF.
    """

    e_x = torch.exp(x)
    return e_x / torch.sum(e_x)


def is_safe_torch(x):
    """
    Checks if a torch.Tensor is safe for unstable version of softmax.

    A torch.Tensor is considered safe if after applying unstable version of softmax,
    it does not contain any NaN or INF values.

    Args:
        x: A torch.Tensor.

    Returns:
        True if x is safe for unstable_softmax, False otherwise.
    """

    if torch.isinf(x).any() or torch.isnan(x).any():
        return False

    if torch.isnan(unstable_softmax_torch(x)).any() or torch.isinf(unstable_softmax_torch(x)).any():
        return False
    return True


def unstable_softmax_tf(x):
    """
    Computes the softmax of an input vector using TensorFlow.

    Args:
        x: A TensorFlow tensor of any shape.

    Returns:
        A TensorFlow tensor of the same shape as x, containing the softmax of each element in x.

    Notes:
        This function is numerically unstable, and can lead to NaN or INF.
    """
    e_x = tf.exp(x)
    return e_x / tf.reduce_sum(e_x)


def is_safe_tf(x):
    """
    Checks if a TensorFlow tensor is safe for an unstable version of softmax.

    A TensorFlow tensor is considered safe if, after applying an unstable version of softmax,
    it does not contain any NaN or INF values.

    Args:
        x: A TensorFlow tensor.

    Returns:
        True if x is safe for unstable_softmax, False otherwise.
    """
    # Check if the input tensor contains any NaN or Inf values
    if tf.reduce_any(tf.math.is_inf(x)) or tf.reduce_any(tf.math.is_nan(x)):
        return False

    # Apply softmax
    softmax_x = tf.nn.softmax(x)

    # Check if the softmax output contains any NaN or Inf values
    if tf.reduce_any(tf.math.is_nan(softmax_x)) or tf.reduce_any(tf.math.is_inf(softmax_x)):
        return False

    return True
