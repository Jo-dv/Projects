import numpy as np

def evaluate(fit):
    scaling = fit + np.abs(np.min(fit))

    return 1 / (1 + scaling)