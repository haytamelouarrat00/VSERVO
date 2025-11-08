import numpy as np


def orthogonalize_matrix(R):
    U, S, V = np.linalg.svd(R)
    R = U @ V
    return R

