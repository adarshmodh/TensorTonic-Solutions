import numpy as np


def relu(x):
    return np.maximum(0, x)

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    # YOUR CODE HERE
    x_arr = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    Ws = np.array(Ws)
    # Main path
    h = relu(x @ W1)
    main = h @ W2

    # Shortcut path
    shortcut = x @ Ws

    # Combine + activation
    out = relu(main + shortcut)

    return out


    
    