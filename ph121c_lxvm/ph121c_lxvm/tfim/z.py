"""Provides Transverse-Field Ising Model Hamiltonians in z basis.

Supports closed and open (default) boundary conditions.
Supports a single parameter h for tuning transver field.
Supports the following formats for storing the Hamiltonian:
- numpy.ndarray
- scipy.sparse.linalg.LinearOperator
- scipy.sparse.linalg.csr_matrix
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator


def H_row (L, h, row, bc='o'):
    """Compute row of matrix elements of H as 3 lists for COO format"""
    # Matrix elements in entry, row, column format
    m_el = ([], [], [])
    def append_entry (*entry):
        for i, e in enumerate(m_el):
            e.append(entry[i])
    # spin flips, sigma x
    for i in range(L):
        append_entry(-h , row, row ^ (2 ** i))
    # sign flips, sigma z
    diag = 0
    for i in range(L-1):
        diag -= (2 * (((row & (2 ** i)) << 1) == (row & (2 ** (i+1)))) - 1)
    if bc == 'c':
        diag -= (2 * (((row & (2 ** 0)) << (L-1)) == (row & (2 ** (L-1)))) - 1)
    append_entry(diag, row, row)
    return m_el

def H_sparse (L, h, bc='o'):
    """Return the Hamiltonian in sparse CSR format"""
    # Matrix elements in entry, row, column format
    m_el = ([], [], [])
    def extend_entry (*entries):
        for i, e in enumerate(m_el):
            e.extend(entries[i])
    for i in range(2 ** L):
        extend_entry(*H_row(L, h, i, bc))
    H = coo_matrix((m_el[0], m_el[1:]), shape=(2**L, 2**L))    
    return H.tocsr()

def H_vec (L, h, v, bc='o'):
    """Compute H|v> efficiently"""
    w       = np.zeros(v.shape)
    indices = np.arange(2 ** L)
    ### spin flips, sigma x
    for i in range(L):
        # flip i-th bit using xor
        w[indices ^ (2 ** i)] -= (h * v)
    ### sign flips, sigma z
    for i in range(L-1):
        # test if adjacent bits are the same
        mask = (((indices & (2 ** i)) << 1) == (indices & (2 ** (i+1))))
        # assign true to 1 (aligned), false to -1 (anti-aligned)
        w -= ((mask * v) - (~mask * v))
    # periodic boundary conditions
    if bc == 'c':
        # test if extremal bits are the same
        mask = (((indices & (2 ** 0)) << (L-1)) == (indices & (2 ** (L-1))))
        # assign true to 1 (aligned), false to -1 (anti-aligned)
        w -= ((mask * v) - (~mask * v))
    return w

def L_vec (L, h, bc='o'):
    """Return a Linear Operator wrapper to compute H|v>"""
    def H_vec_L_h (v):
        return H_vec(L, h, v, bc)
    return LinearOperator((2**L, 2**L), matvec=H_vec_L_h)

def H_dense (L, h, bc='o'):
    """Return the dense Hamiltonian (mostly for testing)"""
    H = np.zeros((2 ** L, 2 ** L))
    v = np.zeros(2 ** L)
    for i in range(2 ** L):
        v[i]    = 1
        H[:, i] = H_vec(L, h, v , bc)
        v[i]    = 0
    return H