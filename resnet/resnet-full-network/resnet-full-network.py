import numpy as np


def relu(x):
    return np.maximum(0, x)


def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    x:      (B,2)
    conv1:  (2,2)

    block1 keeps dim 2->2
    block2 projects 2->3
    fc maps 3->2 logits

    returns: (B,2)
    """

    x = np.array(x)

    conv1 = np.array(conv1)
    W1_b1 = np.array(W1_b1)
    W2_b1 = np.array(W2_b1)

    W1_b2 = np.array(W1_b2)
    W2_b2 = np.array(W2_b2)
    Ws_b2 = np.array(Ws_b2)

    fc = np.array(fc)

    # -------------------
    # Initial conv + ReLU
    # -------------------
    x = relu(x @ conv1)

    # -------------------
    # Block 1 (identity)
    # -------------------
    h = relu(x @ W1_b1)
    main = h @ W2_b1

    x = relu(main + x)

    # -------------------
    # Block 2 (projection)
    # -------------------
    h = relu(x @ W1_b2)
    main = h @ W2_b2

    shortcut = x @ Ws_b2

    x = relu(main + shortcut)

    # -------------------
    # Final classifier
    # -------------------
    logits = x @ fc

    return logits