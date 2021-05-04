"""Calculate spin correlations in states.
"""

import numpy as np

def corr_sz (i, j, v):
    """Compute the correlation C^{zz} (z basis) between sites i, j."""
    indices = np.arange(v.size)
    # test if both bits are the same
    if i <= j:
        mask = (((indices & (2 ** i)) << (j-i) ) == (indices & (2 ** j)))
    else:
        mask = (((indices & (2 ** i)) >> (i-j) ) == (indices & (2 ** j)))
    # assign true to 1 (aligned), false to -1 (anti-aligned) and normalize
    return sum(((mask * v) - (~mask * v)) * v) / sum(v ** 2)

def var_mag (v):
    """Compute the magnetization of a state per site."""
    mag = 0
    for i in range(v.size):
        mag += corr_sz(0, i, v) / v.size
    return mag