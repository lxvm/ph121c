"""Provides Hamiltonians for the toy Rydberg atom model.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..fortran import scars
from ..tensor  import mpo

def H_vec (v, L, O):
    """Compute H|v> efficiently."""
    assert L >= 3
    N_el = 2 ** L
    return scars.h_vec(v, L, O, N_el)

def H_oper (L, O):
    """Return a Linear Operator wrapper to compute H|v>"""
    def H_vec_L_h (v):
        return H_vec(v, L, O)
    return LinearOperator((2**L, 2**L), matvec=H_vec_L_h)

def H_dense (L, O):
    """Return the dense Hamiltonian"""
    H = np.zeros((2 ** L, 2 ** L), dtype='complex')
    v = np.zeros(2 ** L, dtype='complex')
    for i in range(2 ** L):
        v[i]    = 1
        H[:, i] = H_vec(v, L, O)
        v[i]    = 0
    return H

def H_kron (L, O):
    """Build the dense Hamiltonian as a Kronecker product (mostly for testing.)"""
    terms = []
    sx = np.array([[0, 1],  [1,  0]])
    sy = np.array([[0,-1j], [1j, 0]])
    sz = np.array([[1, 0],  [0, -1]])
    # Transverse sx
    for i in range(L):
        terms.append(mpo(L, d=2))
        terms[-1][i] = (O / 2) * sx
    # 3-site befuddler
    for i in range(L):
        # 1-local sz term
        terms.append(mpo(L, d=2))
        terms[-1][(i + 2) % L] = sz / 4
        # sx sx sz term
        terms.append(mpo(L, d=2))
        terms[-1][i] = - sx / 4
        terms[-1][(i + 1) % L] = sx
        terms[-1][(i + 2) % L] = sz
        # sx sy sz term
        terms.append(mpo(L, d=2))
        terms[-1][i] = - sx / 4
        terms[-1][(i + 1) % L] = sy
        terms[-1][(i + 2) % L] = sz
        # sx sx sz term
        terms.append(mpo(L, d=2))
        terms[-1][i] = - sx / 4
        terms[-1][(i + 1) % L] = sx
        terms[-1][(i + 2) % L] = sz
        # sx sz sz term
        terms.append(mpo(L, d=2))
        terms[-1][i] = - sx / 4
        terms[-1][(i + 1) % L] = sz
        terms[-1][(i + 2) % L] = sz
        # sy sx sz term
        terms.append(mpo(L, d=2))
        terms[-1][i] = - sy / 4
        terms[-1][(i + 1) % L] = sx
        terms[-1][(i + 2) % L] = sz
        # sy sy sz term
        terms.append(mpo(L, d=2))
        terms[-1][i] = - sy / 4
        terms[-1][(i + 1) % L] = sy
        terms[-1][(i + 2) % L] = sz
        # sy sz sz term
        terms.append(mpo(L, d=2))
        terms[-1][i] = - sy / 4
        terms[-1][(i + 1) % L] = sz
        terms[-1][(i + 2) % L] = sz
        # sz sx sz term
        terms.append(mpo(L, d=2))
        terms[-1][i] = - sz / 4
        terms[-1][(i + 1) % L] = sx
        terms[-1][(i + 2) % L] = sz
        # sz sy sz term
        terms.append(mpo(L, d=2))
        terms[-1][i] = - sz / 4
        terms[-1][(i + 1) % L] = sy
        terms[-1][(i + 2) % L] = sz
        # sz sz sz term
        terms.append(mpo(L, d=2))
        terms[-1][i] = - sz / 4
        terms[-1][(i + 1) % L] = sz
        terms[-1][(i + 2) % L] = sz
    return sum(e.toarray() for e in terms)