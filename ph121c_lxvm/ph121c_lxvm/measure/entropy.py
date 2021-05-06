"""Calculate entropies.
"""

import numpy as np


def entanglement (vals):
    """Calculate the entanglement entropy of the Schmidt values"""
    return - sum((vals ** 2) * np.log(vals ** 2))

def Shannon (vals):
    """Calculate the Shannon entropy of the values"""
    return - sum(vals * np.log(vals))