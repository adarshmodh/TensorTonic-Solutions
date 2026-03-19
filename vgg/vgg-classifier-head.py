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
    
    W1 = np.random.randn(C*H*W, 4096) * 0.01
    B1 = np.ones(4096)*0.5
    fc1 = relu(flat_feats @ W1 + B1)
    
    W2 = np.random.randn(4096, 4096) * 0.01
    B2 = np.ones(4096)*0.5
    fc2 = relu(fc1 @ W2 + B2)
    
    W3 = np.random.randn(4096, num_classes) * 0.01
    B3 = np.ones(num_classes)*0.5
    fc3 = fc2 @ W3 + B3
    return fc3
