import numpy as np

def relu(x):
    return np.maximum(0, x)


def bn(z, gamma, beta):
    mean = z.mean(axis=0, keepdims=True)
    std  = z.std(axis=0, keepdims=True) + 1e-5
    return gamma * ((z - mean) / std) + beta


def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):

    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)

    if mode == "pre":
        # BN -> ReLU -> W1
        y = bn(x, gamma1, beta1)
        y = relu(y)
        y = y @ W1

        # BN -> ReLU -> W2
        y = bn(y, gamma2, beta2)
        y = relu(y)
        y = y @ W2

        # No ReLU after skip
        out = {}
        out['output'] = y+x
        out['mode'] = mode
        return out


    elif mode == "post":
        # W1 -> BN -> ReLU
        y = x @ W1
        y = bn(y, gamma1, beta1)
        y = relu(y)

        # W2 -> BN
        y = y @ W2
        y = bn(y, gamma2, beta2)

        # ReLU after skip
        out = {}
        out['output'] = relu(y+x)
        out['mode'] = mode
        return out
