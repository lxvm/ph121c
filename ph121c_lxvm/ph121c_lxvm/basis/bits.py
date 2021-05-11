"""Ordinary bit operations.

Bits are number from zero and up, where zero is the fastest changing bit.
"""

import numpy as np


def btest (n, i):
    """Return 0 if the ith bit is 0 or 1 if it is 1."""
    return ((n >> i) & 1)

def swap (n, i, j):
    """Swap bit positions i, j in n."""
    # Do nothing if bits i, j are the same
    # xor flip if bits i, j are not the same
    matched_mask = btest(n, i) == btest(n, j)
    return (n * matched_mask + (n ^ ((1 << i) + (1 << j))) * (matched_mask ^ 1))

def Ising_parity_lookup (n):
    """This is the same as my Fortran function parity_diag."""
    if n > 1:
        table = Ising_parity_lookup(n-1)
        return np.append(table, table ^ 1)
    else:
        return np.arange(2)

# Lookup table of size 256
parity_lookup = Ising_parity_lookup(8)

def poppar (n):
    """Calculate the parity of the population of bits in an integer.
    
    https://graphics.stanford.edu/~seander/bithacks.html
    """
    n ^= n >> 16
    n ^= n >> 8
    return parity_lookup[n & 0xff]

vpoppar = np.vectorize(poppar, otypes=[int])