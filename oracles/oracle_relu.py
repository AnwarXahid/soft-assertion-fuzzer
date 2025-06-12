import torch


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
    if torch.isinf(x).any() or torch.isnan(x).any() or torch.isneginf(x).any():
        return False

    unstable_relu = torch.nn.functional.relu(x)
    if torch.isinf(unstable_relu).any() or torch.isnan(unstable_relu).any() or torch.isneginf(unstable_relu).any():
        return False
    if torch.all(unstable_relu) == 0:
        return False

    return True
