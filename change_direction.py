from run_models import get_direction_for_softmax, get_direction_for_sigmoid, get_direction_for_exp, \
    get_direction_for_tanh, get_direction_for_log, get_direction_for_sqrt, get_direction_for_rsqrt, \
    get_direction_for_relu, get_direction_for_matmul, get_direction_for_conv2d, get_direction_for_div, \
    get_direction_for_cos_sim
from softassertion.utils.unstable_func_enum import UnstableFuncEnum


def get_change_direction(input_x, unstable_func, model_type='3_by_3'):
    if unstable_func == UnstableFuncEnum.LOG.value:
        return get_direction_for_log(input_x, model_type)
    elif unstable_func == UnstableFuncEnum.SOFTMAX.value:
        return get_direction_for_softmax(input_x, model_type)
    elif unstable_func == UnstableFuncEnum.SIGMOID.value:
        return get_direction_for_sigmoid(input_x, model_type)
    elif unstable_func == UnstableFuncEnum.EXP.value:
        return get_direction_for_exp(input_x, model_type)
    elif unstable_func == UnstableFuncEnum.TANH.value:
        return get_direction_for_tanh(input_x, model_type)
    elif unstable_func == UnstableFuncEnum.SQRT.value:
        return get_direction_for_sqrt(input_x, model_type)
    elif unstable_func == UnstableFuncEnum.RSQRT.value:
        return get_direction_for_rsqrt(input_x, model_type)
    elif unstable_func == UnstableFuncEnum.RELU.value:
        return get_direction_for_relu(input_x, model_type)
    elif unstable_func == UnstableFuncEnum.CONV2D.value:
        return get_direction_for_conv2d(input_x, model_type)


def get_change_direction_for_binary_operand(input_x, input_y, unstable_func, model_type='3_by_3'):
    if unstable_func == UnstableFuncEnum.MATMUL.value:
        return get_direction_for_matmul(input_x, input_y, model_type)
    if unstable_func == UnstableFuncEnum.DIV.value:
        return get_direction_for_div(input_x, input_y, model_type)
    if unstable_func == UnstableFuncEnum.COS_SIM.value:
        return get_direction_for_cos_sim(input_x, input_y, model_type)
