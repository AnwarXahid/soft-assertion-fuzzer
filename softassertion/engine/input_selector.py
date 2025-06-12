def get_input_generator(func_name):
    from generators.random_input_generator import (
        get_2d_random_input_torch,
        get_2d_random_input_matmul_torch,
        get_2d_random_input_cos_sim_torch,
        get_2d_random_input_log_torch,
        get_2d_random_input_sigmoid_torch,
        get_2d_random_input_softmax_torch,
        get_2d_random_input_sqrt_torch,
        get_2d_random_input_tanh_torch,
    )

    mapping = {
        "relu": get_2d_random_input_torch,
        "exp": get_2d_random_input_torch,
        "matmul": get_2d_random_input_matmul_torch,
        "cosine_similarity": get_2d_random_input_cos_sim_torch,
        "log": get_2d_random_input_log_torch,
        "sigmoid": get_2d_random_input_sigmoid_torch,
        "softmax": get_2d_random_input_softmax_torch,
        "sqrt": get_2d_random_input_sqrt_torch,
        "tanh": get_2d_random_input_tanh_torch,
    }
    return mapping.get(func_name)

