import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    N = len(y)
    # Write code here
    w = np.zeros((X.shape[1]))
    b = 0.0

    for _ in range(steps):
        logits = X @ w + b
        preds = _sigmoid(logits)
        gradient_w = X.T @ (preds - y) / N
        gradient_b = np.mean(preds - y)
        # print(gradient_w, gradient_b)
        w = w - lr * gradient_w
        b = b - lr * gradient_b
    return w, b
    pass