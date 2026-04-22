import numpy as np


def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Gradient through residual blocks:
    product over (I + J_l) applied to x
    """
    x = np.array(x)
    grad = x.copy()

    for J in gradients_F:
        J = np.array(J)
        I = np.eye(J.shape[0])
        grad = (I + J).T @ grad

    return grad

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # YOUR CODE HERE
    x = np.array(x)
    grad = x.copy()

    for J in gradients_F:
        J = np.array(J)
        grad = J.T @ grad

    return grad