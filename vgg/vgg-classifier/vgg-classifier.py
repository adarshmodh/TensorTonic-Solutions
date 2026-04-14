import numpy as np

def relu(x):
    return np.maximum(0, x)

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, num_classes) with classification logits
    """
    # Your implementation here
    B, C, H, W = features.shape
    flat_feats = features.reshape(B, C*H*W)
    
    fc1 = relu(flat_feats @ W1 + b1)    
    fc2 = relu(fc1 @ W2 + b2)    
    fc3 = fc2 @ W3 + b3
    return fc3

