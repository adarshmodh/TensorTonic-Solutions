import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net: trace shape through 4 encoder blocks, bottleneck, 4 decoder blocks, output.
    Each block: two 3x3 unpadded convs (reduce by 4), encoder pools (halve), decoder upsamples (double).
    Returns zero array with correct output shape.
    """
    # Your implementation here
    enc_out = np.zeros(x.shape)
    for i in range(4):
        B, H, W, C = enc_out.shape
        C_out = 32
        enc_out = np.zeros((B, (H-4)//2, (W-4)//2, C_out*2))

    B, H, W, C = enc_out.shape
    bn_out = np.zeros((B, H-4, W-4, C*2))

    dec_out = np.zeros(bn_out.shape)
    for i in range(4):
        B, H, W, C = dec_out.shape
        if i !=3:
            dec_out = np.zeros((B, 2*H-4, 2*W-4, C//2))
        else:
            dec_out = np.zeros((B, 2*H-4, 2*W-4, num_classes))
            
    return dec_out
