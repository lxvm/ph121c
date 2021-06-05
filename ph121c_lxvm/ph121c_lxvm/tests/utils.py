from itertools import product

import numpy as np

def tfim_sweep (
    L  = np.arange(4, 11, 2),
    h  = np.array([0.3, 1.0, 1.7]),
    bc = np.array(['o', 'c']),
    **kwargs
):
    """Generate named TFIM parameter values to sweep over."""
    yield from sweep(L=L, h=h, bc=bc, **keys)
    
def sweep (**kwargs):
    """Generate named parameter values to sweep over."""
    for values in product(*kwargs.values()):
        yield { k : v for k, v in zip(kwargs.keys(), values) }
     