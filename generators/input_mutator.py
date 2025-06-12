import torch
import tensorflow as tf
from utils.input_generator.random_input_generator import get_2d_random_input_torch, get_2d_random_input_tf


def get_mutated_input(original_input, rate=0.1, direction='pos',
                      variation='', random_min_val=0, random_max_val=100, flag='torch'):
    if flag == 'torch':
        return get_mutated_input_torch(original_input, rate, direction, variation, random_min_val, random_max_val)
    else:
        return get_mutated_input_tf(original_input, rate, direction, variation, random_min_val, random_max_val)


def get_mutated_input_torch(tensor, rate=0.1, direction='pos', variation='',
                            random_min_val=0, random_max_val=100):
    # Ensure tensor is floating-point for mutation
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)

    # Generate random noise with the same data type as the original tensor
    random_noise = torch.rand(tensor.size(), dtype=tensor.dtype) * (random_max_val - random_min_val) + random_min_val

    # Apply mutation based on the specified direction and variation
    if direction == 'neg':
        random_noise = -random_noise

    # Apply the random noise to the tensor
    if variation == 'rand':
        mutated_tensor = tensor + rate * random_noise
    elif variation == 'sino':
        mutated_tensor = tensor + torch.sin(rate * random_noise)
    elif variation == 'expo':
        mutated_tensor = tensor + torch.exp(rate * random_noise)
    else:
        mutated_tensor = tensor + rate * random_noise  # Default case

    return mutated_tensor


def get_mutated_input_tf(original_input, rate=0.1, direction='pos',
                         variation='', random_min_val=0, random_max_val=100):
    """
    Mutates an input tensor by adding a random noise to it.

    Args:
        original_input: A tf.Variable of any shape.
        rate: The mutation rate. A value between 0 and 1.
        direction: The direction of the mutation. Can be either 'pos' (positive) or 'neg' (negative).
        variation: The type of mutation. Can be either 'sino' (sinusoidal), 'expo' (exponential), or 'rand' (random).
        random_min_val: The minimum value of the random noise.
        random_max_val: The maximum value of the random noise.

    Returns:
        A tf.Variable of the same shape as original_input, containing the mutated input.
    """

    tensor_shape = tf.shape(original_input)
    tensor_dtype = original_input.dtype

    if direction == 'pos':
        min_value_for_random = random_min_val
        max_value_for_random = random_max_val
    else:
        min_value_for_random = -random_max_val
        max_value_for_random = random_min_val

    random_input = tf.random.uniform(tensor_shape, min_value_for_random, max_value_for_random, dtype=tensor_dtype)

    if variation == 'sino':
        mutated_input = original_input + tf.multiply(tf.sin(random_input), rate)
    elif variation == 'expo':
        mutated_input = original_input + tf.multiply(tf.exp(random_input), rate)
    elif variation == 'rand':
        mutated_input = original_input + tf.multiply(random_input, rate)
    else:
        mutated_input = original_input + rate

    return mutated_input


def get_mutated_input_tuple(original_input, rate=0.1, direction='pos',
                            variation='', random_min_val=0, random_max_val=100, flag='torch'):
    if flag == 'torch':
        mutated_inputs = tuple(get_mutated_input_torch(tensor, rate, direction, variation,
                                                       random_min_val, random_max_val) for tensor in original_input)
    else:
        mutated_inputs = tuple(get_mutated_input_tf(tensor, rate, direction, variation,
                                                    random_min_val, random_max_val) for tensor in original_input)
    return mutated_inputs
