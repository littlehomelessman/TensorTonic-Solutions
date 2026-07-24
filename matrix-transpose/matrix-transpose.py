import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    # Write code here
    N = A.shape[0]
    M = A.shape[1]
    res = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            res[m][n] = A[n][m]
    return res
    pass
