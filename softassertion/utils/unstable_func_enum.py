from enum import Enum


class UnstableFuncEnum(str, Enum):
    LOG = 'log'
    SOFTMAX = 'softmax'
    SIGMOID = 'sigmoid'
    SQRT = 'sqrt'
    EXP = 'exp'
    TANH = 'tanh'
    RSQRT = 'rsqrt'
    MATMUL = 'matmul'
    CONV2D = 'con2d'
    CROSS_ENTROPY = 'cross_entropy'
    RELU = 'relu'
    DIV = 'div'
    COS_SIM = 'cosine_similarity'
    WEIGHT_INIT = 'weight_init'
