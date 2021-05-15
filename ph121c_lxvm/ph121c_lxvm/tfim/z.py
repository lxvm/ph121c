"""Provides Transverse-Field Ising Model Hamiltonians in z basis.

Supports closed and open (default) boundary conditions.
Supports a single parameter h for tuning transverse field.
Supports the following formats for storing the Hamiltonian:
- numpy.ndarray
- scipy.sparse.linalg.LinearOperator
- scipy.sparse.linalg.csr_matrix
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator

from ..basis   import bits
from ..fortran import tfim_z


sign = lambda x: ((2 * x) - 1)

def H_sparse (L, h, bc):
    """Return the Hamiltonian in sparse CSR format."""
    assert bc in ['c', 'o']
    N_el = (L + 1) * (2 ** L)

    data, rows, cols = tfim_z.h_coo(N_el, L, h, bc)     
    return coo_matrix((data, (rows, cols)), shape=(2**L, 2**L)).tocsr()

def H_vec (v, L, h, bc):
    """Compute H|v> efficiently."""
    assert bc in ['c', 'o']
    N_el = 2 ** L
    
    return tfim_z.h_vec(v, L, h, bc, N_el)

def old_H_sparse (L, h, bc):
    """Return the Hamiltonian in sparse CSR format.
    
    Use the fact that the Hamiltonian has (L+1) * (2 ** L) elements because
    there are exactly L+1 entries per row (unique columns):
    Diagonal is unique and bit flip at distinct entries is unique by Cantor's
    diagonalization argument.
    """
    assert bc in ['c', 'o']
    # Build matrix in COO format
    N_el = (L+1) * (2 ** L) 
    data = np.zeros(N_el) 
    rows = np.zeros(N_el)
    cols = np.zeros(N_el)
    n    = 0
    # The sign flip functions could be optimized by removing the loop
    # which can be done by counting the number of nonzero bits.
    # Refer to the Fortran implementation for how to do this using only
    # xor, bit shifts, cyclic bit shifts, and popcounts.
    # These functions could be written in Python using bin() and bit intrinsics
    for i in range(2 ** L):
        # spin flips, sigma x
        for j in range(L):
            data[n] = -h
            rows[n] = i
            cols[n] = i ^ (2 ** j)
            n += 1
        # siign flips, sigma z
        rows[n] = i
        cols[n] = i
        for j in range(L - 1 + (bc == 'c')):
            data[n] -= sign(bits.btest(i, j) == bits.btest(i, (j+1) % L))
        n += 1
    return coo_matrix((data, (rows, cols)), shape=(2**L, 2**L)).tocsr()

def old_H_vec (v, L, h, bc):
    """Compute H|v> efficiently"""
    assert bc in ['c', 'o']
    w       = np.zeros(v.shape)
    indices = np.arange(2 ** L)
    
    # spin flips, sigma x
    for i in range(L):
        # flip i-th bit using xor
        w[indices ^ (2 ** i)] -= (h * v)
    # siign flips, sigma z
    for i in range(L - 1 + (bc == 'c')):
        # test if adjacent bits are the same
        w -= v * sign(bits.btest(indices, i) == bits.btest(indices, (i+1) % L))
    return w

def H_oper (L, h, bc):
    """Return a Linear Operator wrapper to compute H|v>"""
    assert bc in ['c', 'o']
    def H_vec_L_h (v):
        return H_vec(v, L, h, bc)

    return LinearOperator((2**L, 2**L), matvec=H_vec_L_h)

def H_dense (L, h, bc):
    """Return the dense Hamiltonian (mostly for testing)"""
    assert bc in ['c', 'o']
    H = np.zeros((2 ** L, 2 ** L))
    v = np.zeros(2 ** L)
    
    for i in range(2 ** L):
        v[i]    = 1
        H[:, i] = H_vec(v, L, h, bc)
        v[i]    = 0
    return H