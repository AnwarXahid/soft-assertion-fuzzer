import torch

def get_2d_random_input_torch(tensor_size=(3, 3), min_value=-10, max_value=100, data_type=torch.float32):
    return torch.rand(tensor_size, dtype=data_type) * (max_value - min_value) + min_value

def get_2d_random_input_matmul_torch(tensor_size=(3, 3), min_value=-10, max_value=100, data_type=torch.float32):
    a = torch.rand(tensor_size[0], tensor_size[1], dtype=data_type) * (max_value - min_value) + min_value
    b = torch.rand(tensor_size[1], tensor_size[0], dtype=data_type) * (max_value - min_value) + min_value
    return a, b

def get_2d_random_input_cos_sim_torch(tensor_size=(3, 3), min_value=-10, max_value=100, data_type=torch.float32):
    a = torch.rand(tensor_size, dtype=data_type) * (max_value - min_value) + min_value
    b = torch.rand(tensor_size, dtype=data_type) * (max_value - min_value) + min_value
    return a, b

def get_2d_random_input_log_torch(tensor_size=(3, 3), min_value=0.01, max_value=100, data_type=torch.float32):
    return torch.rand(tensor_size, dtype=data_type) * (max_value - min_value) + min_value

def get_2d_random_input_sigmoid_torch(tensor_size=(3, 3), min_value=-10, max_value=10, data_type=torch.float32):
    return torch.rand(tensor_size, dtype=data_type) * (max_value - min_value) + min_value

def get_2d_random_input_softmax_torch(tensor_size=(3, 3), min_value=-10, max_value=10, data_type=torch.float32):
    return torch.rand(tensor_size, dtype=data_type) * (max_value - min_value) + min_value

def get_2d_random_input_sqrt_torch(tensor_size=(3, 3), min_value=0.01, max_value=100, data_type=torch.float32):
    return torch.rand(tensor_size, dtype=data_type) * (max_value - min_value) + min_value

def get_2d_random_input_tanh_torch(tensor_size=(3, 3), min_value=-10, max_value=10, data_type=torch.float32):
    return torch.rand(tensor_size, dtype=data_type) * (max_value - min_value) + min_value

