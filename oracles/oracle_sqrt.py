import torch


def is_safe_torch(x):
    """
    Checks if a torch.Tensor is safe for unstable version of sqrt.

    A torch.Tensor is considered safe if after applying unstable version of sqrt,
    it does not contain any NaN or INF values.

    Args:
        x: A torch.Tensor.

    Returns:
        True if x is safe for unstable_sqrt, False otherwise.
    """

    if torch.isinf(x).any() or torch.isnan(x).any() or torch.isneginf(x).any():
        return False

    unstable_sqrt = torch.sqrt(x)

    if torch.isnan(unstable_sqrt).any() or torch.isinf(unstable_sqrt).any() or torch.isneginf(unstable_sqrt).any():
        return False

    return True
