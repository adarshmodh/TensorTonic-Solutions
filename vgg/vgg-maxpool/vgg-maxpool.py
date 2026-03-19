import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    # Your implementation here
    B, H, W, C = x.shape
    # convert 224, 224 to 112,2 and 112, 2, because spatial dimensions are halved 
    x = x.reshape(B, H//2, 2, W//2, 2, C)
    return x.max(axis=(2, 4))
