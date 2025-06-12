import torch


def is_safe_torch(x):
    """
    Checks if a torch.Tensor is safe for unstable version of tanh.

    A torch.Tensor is considered safe if after applying unstable version of tanh,
    it does not contain any NaN or INF values.

    Args:
        x: A torch.Tensor.

    Returns:
        True if x is safe for unstable_tanh, False otherwise.
    """

    # Check if x has nan or inf
    if torch.isinf(x).any() or torch.isnan(x).any() or torch.isneginf(x).any():
        return False

    unstable_tanh = torch.tanh(x)
    # check for nan or inf produced by unstable_tanh
    if torch.isnan(unstable_tanh).any() or torch.isinf(unstable_tanh).any() or (torch.isneginf(unstable_tanh).any()):
        return False
    return True
