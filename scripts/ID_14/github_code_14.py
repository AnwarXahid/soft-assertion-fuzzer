import numpy as np

"""
Source: https://github.com/fakufaku/diffusion-separation/blob/main/utils/linalg.py
"""

def solve_numpy(A, b):
    """Solve Ax = b using NumPy."""
    x = np.linalg.solve(A, b)
    return x
