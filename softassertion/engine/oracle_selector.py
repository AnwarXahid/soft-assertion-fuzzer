def get_oracle_for_function(func_name):
    from oracles.oracle_relu import is_safe_torch as relu_oracle
    from oracles.oracle_exp import is_safe_torch as exp_oracle
    from oracles.oracle_matmul import is_safe_torch as matmul_oracle
    from oracles.oracle_cosine_similarity import is_safe_torch as cos_sim_oracle
    from oracles.oracle_log import is_safe_torch as log_oracle
    from oracles.oracle_sigmoid import is_safe_torch as sigmoid_oracle
    from oracles.oracle_softmax import is_safe_torch as softmax_oracle
    from oracles.oracle_sqrt import is_safe_torch as sqrt_oracle
    from oracles.oracle_tanh import is_safe_torch as tanh_oracle

    mapping = {
        "relu": relu_oracle,
        "exp": exp_oracle,
        "matmul": matmul_oracle,
        "cosine_similarity": cos_sim_oracle,
        "log": log_oracle,
        "sigmoid": sigmoid_oracle,
        "softmax": softmax_oracle,
        "sqrt": sqrt_oracle,
        "tanh": tanh_oracle,
    }
    return mapping.get(func_name)
