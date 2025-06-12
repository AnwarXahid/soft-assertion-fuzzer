import torch
import tensorflow as tf


def unstable_sigmoid(x):
    """
    Computes the sigmoid of an input vector.

    Args:
        x: A torch.Tensor of any shape.

    Returns:
        A torch.Tensor of the same shape as x, containing the sigmoid of each element in x.

    Notes:
        This function is numerically unstable, and can lead to NaN or INF.
    """

    return 1 / (1 + torch.exp(-x))


def is_safe_torch(x):
    """
    Checks if a torch.Tensor is safe for unstable version of sigmoid.

    A torch.Tensor is considered safe if after applying unstable version of sigmoid,
    it does not contain any NaN or INF values.

    Args:
        x: A torch.Tensor.

    Returns:
        True if x is safe for unstable_sigmoid, False otherwise.
    """

    if torch.isinf(x).any() or torch.isnan(x).any() or torch.isneginf(x).any():
        return False

    sigm = torch.sigmoid(x)

    if torch.isnan(sigm).any() or torch.isinf(sigm).any() or torch.isneginf(sigm).any():
        return False

    # # condition
    # if sigm == 0.5:
    #     return False

    return True


# def is_safe_torch(x):
#     """
#     Checks if a torch.Tensor is safe for unstable version of sigmoid.
#
#     A torch.Tensor is considered safe if after applying unstable version of sigmoid,
#     it does not contain any NaN or INF values.
#
#     Args:
#         x: A torch.Tensor.
#
#     Returns:
#         True if x is safe for unstable_sigmoid, False otherwise.
#     """
#
#     # Check if x has nan or inf
#     if torch.isinf(x).any() or torch.isnan(x).any():
#         return False
#
#     expo = torch.exp(-x)
#
#     # check for internal nan of inf
#     if torch.isinf(expo).any() or torch.isnan(expo).any():
#         return False
#
#     # Check for divide by zero
#     if torch.eq(expo, -1).any():
#         return False
#
#     # check for nan or inf produced by unstable_sigmoid
#     if torch.isnan(unstable_sigmoid(x)).any() or torch.isinf(unstable_sigmoid(x)).any():
#         return False
#     return True


def has_nan_inf(x):
    """
        Checks if a torch.Tensor has nan or inf.

        Args:
            x: A torch.Tensor.

        Returns:
            True if x has nan or inf, False otherwise.
        """

    # Check if x has nan or inf
    if torch.isinf(x).any() or torch.isnan(x).any():
        return True
    return False


def unstable_sigmoid_tf(x):
    """
    Computes the sigmoid of an input tensor using TensorFlow.

    This function is numerically unstable and can lead to NaN or INF values.

    Args:
        x: A tf.Tensor of any shape.

    Returns:
        A tf.Tensor of the same shape as x, containing the sigmoid of each element in x.
    """
    return 1 / (1 + tf.exp(-x))


def is_safe_tf(x):
    """
    Checks if a tf.Tensor is safe for the unstable version of the sigmoid function in TensorFlow.

    A tf.Tensor is considered safe if after applying the unstable version of sigmoid,
    it does not contain any NaN or INF values.

    Args:
        x: A tf.Tensor.

    Returns:
        True if x is safe for unstable_sigmoid_tf, False otherwise.
    """

    # Check if x has nan or inf
    if tf.reduce_any(tf.math.is_inf(x)) or tf.reduce_any(tf.math.is_nan(x)):
        return False

    expo = tf.exp(-x)

    # Check for internal nan or inf
    if tf.reduce_any(tf.math.is_inf(expo)) or tf.reduce_any(tf.math.is_nan(expo)):
        return False

    # Check for divide by zero
    # Note: TensorFlow handles divide by zero internally by returning inf, so no explicit check is needed here

    # Check for nan or inf produced by unstable_sigmoid_tf
    sigmoid_result = unstable_sigmoid_tf(x)
    if tf.reduce_any(tf.math.is_nan(sigmoid_result)) or tf.reduce_any(tf.math.is_inf(sigmoid_result)):
        return False

    return True
