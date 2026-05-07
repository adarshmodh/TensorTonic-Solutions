import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    scores = Q @ K.transpose(-2, -1) # raw attention scores
    d_k = K.shape[-1] #dimensionality of keys
    scores = scores/math.sqrt(d_k)  #scaled attention
    attn = F.softmax(scores, dim=-1) # normalize attn scores along key dimension
    attn = attn @ V
    return attn