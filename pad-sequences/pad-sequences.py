import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len == None:
        max_len = max(len(s) for s in seqs)
    res = np.full((len(seqs), max_len), pad_value)
    for i in range(len(seqs)):
        s = seqs[i]
        r = np.full(max_len, pad_value)
        r[:min(max_len, len(s))] = s[:min(max_len, len(s))]
        res[i, :] = r
    return res
    pass