import numpy as np

def relu(x):
    return np.maximum(0, x)
  
def vgg_classifier(features: np.ndarray, num_classes: int = 1000) -> np.ndarray:
    """
    Implement VGG's fully connected classifier.
    """
    # Your implementation here
    B, C, H, W = features.shape
    flat_feats = features.reshape(B, C*H*W)
    
    weights = np.random.randn(C*H*W, 4096) * 0.01
    bias = np.ones(4096)*0.5
    fc1 = relu(flat_feats @ weights + bias)
    
    weights = np.random.randn(4096, 4096) * 0.01
    fc2 = relu(fc1 @ weights + bias)
    
    weights = np.random.randn(4096, num_classes) * 0.01
    bias = np.ones(num_classes)*0.5
    fc3 = fc2 @ weights + bias
    return fc3
