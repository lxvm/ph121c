"""Define useful unitary matrices.
"""

import numpy as np

from . import bits


def Hadamard (L=1):
    """Return a Hadamard unitary applied to each site of a length L spin chain."""
    coef = np.sqrt(2) ** (-L)
    defn = np.array([[1, 1], [1, -1]])
    gate = np.array([[1, 1], [1, -1]])
    
    for i in range(L-1):
        gate = np.kron(gate, defn)
    return coef * gate

def Ising (L=1, inverse=False):
    """Return a permutation that relates the +/- sectors to the computational basis.
    
    Use convention that even parity belongs to + sector and odd parity to - sector.
    """
    diag = bits.Ising_parity_lookup(L)
    sort = np.concatenate((
        np.argwhere(diag == 0).ravel(),
        np.argwhere(diag == 1).ravel(),
    ))
    if inverse:
        sort = sort.argsort()
    return sort
        