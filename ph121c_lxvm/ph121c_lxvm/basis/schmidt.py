"""Computes Schmidt decompositions of vectors relative to subsystems.

Generically speaking, for row-major array storage, this comes down to reshaping
a vector into an array if the subsystem used to matricize is either on the big
end or little end of the bit string. Here, however, I try to support arbitrary
subsystems by applying the correct permutations where necessary.
"""

import numpy as np

from . import bits, combinatorics

def permutation (L, A, v, reverse=False):
    """Return the permutation to use for sorting in Schmidt decomposition."""
    perm = sorted(A) + sorted(set(range(L)).difference(A))
    if reverse:
        perm = [ i for i, _ in sorted(enumerate(perm), key=lambda x: x[1]) ]
    return perm
    
def permute (L, A, v, reverse=False):
    """Permute the state vector by sorting bits in A to fastest-changing position."""
    # Find the final desired arrangement of the indices
    perm = permutation(L, A, v, reverse)
    indices = np.arange(v.size)
    for cycle in combinatorics.cycles(perm):
        for i, j in zip(cycle[:-1], cycle[1:]):
            indices = bits.swap(indices, i, j)
    return v[indices]

def matricize (L, A, v):
    """Matricize a vector onto a subsystem A."""
    return permute(L, A, v).reshape((2 ** (L - len(A)), 2 ** len(A)))
    
def vectorize (L, A, M):
    """Vectorize a matrix."""
    return permute(L, A, M.reshape((2 ** L,)), reverse=True)

def values (L, A, v):
    """Calculate the Schmidt values of v in subsystem A."""
    _, vals, _ = np.linalg.svd(matricize(L, A, v), full_matrices=False)
    return vals