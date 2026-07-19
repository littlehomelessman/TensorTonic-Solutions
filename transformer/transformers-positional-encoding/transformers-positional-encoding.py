import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    row = np.arange(seq_length).reshape(-1, 1)
    
    col = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pe = np.zeros((seq_length, d_model))

    angle_rads = row * col

    pe[:, 0::2] = np.sin(angle_rads)
    pe[:, 1::2] = np.cos(angle_rads)

    return pe