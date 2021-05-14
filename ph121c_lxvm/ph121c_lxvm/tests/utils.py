from itertools import product

import numpy as np

def tfim_sweep (
    L  = np.arange(4, 11, 2),
    h  = np.array([0.3, 1.0, 1.7]),
    bc = np.array(['o', 'c']),
    **kwargs
):
    """Generate parameter values to sweep over"""
    keys = ['L', 'h', 'bc']
    keys.extend(kwargs.keys())
    for values in product(L, h, bc, *kwargs.values()):
        yield { k : v for k, v in zip(keys, values) }
     