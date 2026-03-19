import numpy as np

def relu(x):
    return np.maximum(0, x)

def conv_relu(x, out_channels):
    B, H, W, C = x.shape
    W_weights = np.random.randn(C, out_channels) * 0.1
    x = x @ W_weights
    return relu(x)

def vgg_conv_block(x: np.ndarray, num_convs: int, out_channels: int) -> np.ndarray:
    """
    Implement a VGG-style convolutional block.
    """
    # Your implementation here
    
    # Get dimensions
    for i in range(num_convs):
        if i == 0:
            out = conv_relu(x, out_channels)
        else:
            out = conv_relu(out, out_channels)
            
    return out