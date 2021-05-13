"""Computes Schmidt decompositions of vectors relative to subsystems.

Generically speaking, for row-major array storage, this comes down to reshaping
a vector into an array if the subsystem used to matricize is either on the big
end or little end of the bit string. Here, however, I try to support arbitrary
subsystems by applying the correct permutations where necessary.

For reference: https://arxiv.org/pdf/1008.3477.pdf.
"""

import numpy as np

from . import bits, combinatorics


def permutation (v, A, L, inverse=False):
    """Return the permutation to use for sorting in Schmidt decomposition."""
    
    subsystem = np.sort(A)
    perm = np.concatenate((
        subsystem,
        np.sort([ i for i in range(L) if i not in subsystem ])
    ))
    if inverse:
        perm = np.argsort(perm)
    return perm
    
def permute (v, A, L, inverse=False):
    """Permute v by sorting sites in A to fastest-changing position."""

    perm = permutation(v, A, L, inverse)
    indices = np.arange(v.size)
    # permute the bits of the computational basis via cycles to achieve the perm
    for cycle in combinatorics.cycles(perm):
        for i, j in zip(cycle[:-1], cycle[1:]):
            indices = bits.swap(indices, i, j)
    return v[indices]

def matricize (v, A, L):
    """Matricize a vector onto a spin chain subsystem A.
    
    This takes a vector |a>|b> and returns a matrix |a><b|.
    The subsystem A specifies the bit positions in the spin chain.
    This does not implement qudit systems or truncated systems (d == 2).
    """
    return permute(v, A, L, inverse=True).reshape((2 ** len(A), 2 ** (L - len(A))), order='F')
    
def vectorize (M, A, L):
    """Vectorize a matrix: the inverse of matricization.
    
    This takes a matrix |a><b| and returns a vector |a>|b>.
    The subsystem A specifies the bit positions in the spin chain.
    This does not implement qudit systems for d > 2.
    """
    return permute(M.ravel(order='F'), A, L, inverse=False)

def values (v, A, L):
    """Calculate the Schmidt values of v in subsystem A."""
    _, vals, _ = np.linalg.svd(matricize(v, A, L), full_matrices=False)
    return vals

def svd_tr (u, s, vh, n):
    """Truncate n largest principal values of svd"""
    if n == s.size:
        return (u, s, vh)
    else:
        return (up[:, :n], s[:n], vh[:n, :])

def svd_rc (u, s, vh, n):
    """Reconstruct matrix using n largest principal values of svd"""
    # equivalently, due to broadcasting rules
    # u[:, :n] @ (s[:n, None] * vh[:n, :])
    return (u[:, :n] * s[:n]) @ vh[:n, :]