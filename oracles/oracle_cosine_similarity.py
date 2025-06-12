import torch
import numpy as np


def unstable_cosine_similarity(x, y):
    def cosine_similarity_torch(x, y):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        x_tensor = x_tensor if len(x_tensor.shape) == 2 else x_tensor.unsqueeze(0)
        y_tensor = y_tensor if len(y_tensor.shape) == 2 else y_tensor.unsqueeze(0)
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(x_tensor, y_tensor, dim=1)
        return cos_sim.numpy()

    def cosine_similarity_numpy(x, y):
        x_array = np.array(x, dtype=np.float32)
        y_array = np.array(y, dtype=np.float32)
        dot_product = np.dot(x_array, y_array)
        norm_x = np.linalg.norm(x_array)
        norm_y = np.linalg.norm(y_array)
        cos_sim = dot_product / (norm_x * norm_y)
        return cos_sim

    cos_sim_torch = cosine_similarity_torch(x, y)
    # print(cos_sim_torch)
    cos_sim_numpy = cosine_similarity_numpy(x, y)
    # print(cos_sim_numpy)

    abs_difference = abs(cos_sim_torch - cos_sim_numpy)
    # print("abs_difference",abs_difference)
    if (abs_difference <= 0.05).any():
        return True
    else:
        return False


def is_safe_torch(x, y):
    if torch.isinf(x).any() or torch.isnan(x).any() or torch.isinf(y).any() or torch.isnan(y).any():
        return False

    if not (unstable_cosine_similarity(x, y)):
        return False

    return True
