import numpy as np

def relu(x):
    return np.maximum(0, x)
    
def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    x_arr = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)  
    Ws = np.array(Ws)
    
    shortcut = x_arr @ Ws
    y = relu(x_arr @ W1)
    y = relu(y @ W2)
    y = y @ W3
    return relu(y + shortcut)   
