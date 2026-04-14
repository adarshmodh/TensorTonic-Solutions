import numpy as np


def relu(x):
    return np.maximum(0, x)
    
def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE
    x_arr = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    
    h = relu(x_arr @ W1.T)
    y = h @ W2.T
    out = relu(y + x_arr)
    return out
