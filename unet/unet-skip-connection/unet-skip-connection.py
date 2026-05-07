import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims, then concatenate along channels.
    """
    # Your implementation here
    B_e, H_e, W_e, C_e = encoder_features.shape
    B_d, H_d, W_d, C_d = decoder_features.shape
    start_H = (H_e - H_d)//2
    start_W = (W_e - W_d)//2
    
    cropped_enc_features = encoder_features[:, start_H: start_H + H_d, start_W: start_W + W_d, :]
    return np.concat((cropped_enc_features, decoder_features), axis=3 )