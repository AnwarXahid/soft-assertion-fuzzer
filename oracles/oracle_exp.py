import torch


def is_safe_torch(x):
    """
    Checks if a torch.Tensor is safe for unstable version of exp.

    A torch.Tensor is considered safe if after applying unstable version of exp,
    it does not contain any NaN or INF values.

    Args:
        x: A torch.Tensor.

    Returns:
        True if x is safe for unstable_exp, False otherwise.
    """

    if torch.isinf(x).any() or torch.isnan(x).any() or torch.isneginf(x).any():
        return False
    unstable_exp = torch.exp(x)
    if torch.isnan(unstable_exp).any() or torch.isinf(unstable_exp).any() or torch.isneginf(unstable_exp).any():
        return False
    return True
