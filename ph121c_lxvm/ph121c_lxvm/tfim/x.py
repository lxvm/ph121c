"""Provides Transverse-Field Ising Model Hamiltonians in x basis.

Exploits the Ising symmetry sectors to reduce the computational cost.
The Hamiltonians implement either the + or - symmetry sector.

Supports closed and open (default) boundary conditions.
Supports a single parameter h for tuning transver field.
Supports the following formats for storing the Hamiltonian:
- numpy.ndarray
- scipy.sparse.linalg.LinearOperator
- scipy.sparse.linalg.csr_matrix

Conceptually, Hamiltonians in the x-basis are obtain from those in the z-basis 
by applying a Hadamard gate to each site:
[[1,  1],
 [1, -1]] * (np.sqrt(2) ** -1)
This means sigma z in the x basis has the matrix elements of sigma x in the z
basis, and vice versa.
In implementation, we just apply the appropriate bit flip operators.

Open boundary conditions don't give a Hamiltonian that exhibits the duality
sigma_i^x -> tau_i^x tau_{i+1}^x and sigma_i^z sigma_{i+1}^z -> tau_i^x,
but it is easy enough to implement this case as a basis due to the bitwise
construction of the Hamiltonian, simply ommiting the effect of the periodic
boundary term sigma_{L-1}^z sigma_0^z.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator

from ..basis   import bits
from ..fortran import tfim


sign = lambda x: ((2 * x) - 1)

def H_sparse (L, h, bc, sector):
    """Return the Hamiltonian in sparse CSR format."""
    assert bc in ['c', 'o']
    assert sector in ['+', '-', 'f']
    if sector == 'f':
        N = L
        N_el = (L + (bc == 'c')) * (2 ** N)
    else:
        N = L - 1
        N_el = (L + (bc == 'c')) * (2 ** N)

    data, rows, cols = tfim.h_x_coo(N_el, L, h, bc, sector)
    return coo_matrix((data, (rows, cols)), shape=(2**N, 2**N)).tocsr()

def H_vec (v, L, h, bc, sector):
    """Compute H|v> efficiently."""
    assert bc in ['c', 'o']
    assert sector in ['+', '-', 'f']
    if sector == 'f':
        N_el = 2 ** L
    else:
        N_el = 2 ** (L - 1)
    
    return tfim.h_x_vec(v, L, h, bc, sector, N_el)

def old_H_sparse (L, h, bc, sector):
    """Return the Hamiltonian in sparse CSR format.
    
    Use the fact that the Hamiltonian has (L + 1) * (2 ** (L - 1)) or
    L * (2 ** (L - 1)) elements for closed and open systems, respectively.
    Diagonal is unique and bit flip at distinct entries is unique by Cantor's
    diagonalization argument.
    """
    assert bc in ['c', 'o']
    assert sector in ['+', '-']
    # Build matrix in COO format
    N_el = (L + (bc == 'c'))* (2 ** (L - 1)) 
    data = np.zeros(N_el) 
    rows = np.zeros(N_el)
    cols = np.zeros(N_el)
    n    = 0
    # The sign flip functions could be optimized by removing the loop
    # which can be done by counting the number of nonzero bits.
    # Refer to the Fortran implementation for how to do this using only
    # xor, bit shifts, cyclic bit shifts, and popcounts.
    # These functions could be written in Python using bin() and bit intrinsics
    for i in range(2 ** (L-1)):
        k = ((2 * i) + (bits.poppar(i) ^ (sector == '-')))
        # sign flips, sigma x
        for j in range(L):
            data[n] -= h * sign(bits.btest(k, j))
        rows[n] = i
        cols[n] = i
        n += 1
        # spiin flips, sigma z
        for j in range(L - 1 + (bc == 'c')):
            m = k ^ ((2 ** j) + (2 ** ((j + 1) % L)))
            data[n] -= 1
            rows[n] = i
            cols[n] = (m - (m % 2)) // 2
            n += 1
    return coo_matrix((data, (rows, cols)), shape=(2**(L-1), 2**(L-1))).tocsr()

def old_H_vec (v, L, h, bc, sector):
    """Compute H|v> efficiently"""
    assert bc in ['c', 'o']
    assert sector in ['+', '-']
    w       = np.zeros(v.shape)
    indices = np.arange(2 ** (L-1))
    
    # Relate sector index to full index
    k = ((2 * indices) + (bits.vpoppar(indices) ^ (sector == '-')))
    # sign flips, sigma x
    for i in range(L):
        w -= h * v * sign(bits.btest(k, i))
    # spiin flips, sigma z
    for i in range(L - 1 + (bc == 'c')):
        m = k ^ ((2 ** i) + (2 ** ((i + 1) % L)))
        m = (m - (m % 2)) // 2
        w[m] -= v
    return w

def H_oper (L, h, bc, sector):
    """Return a Linear Operator wrapper to compute H|v>"""
    def H_vec_L_h (v):
        return H_vec(v, L, h, bc, sector)
    assert bc in ['c', 'o']
    assert sector in ['+', '-', 'f']
    if sector == 'f':
        N = L
    else:
        N = L - 1
        
    return LinearOperator((2**N, 2**N), matvec=H_vec_L_h)

def H_dense (L, h, bc, sector):
    """Return the dense Hamiltonian (mostly for testing)"""
    assert bc in ['c', 'o']
    assert sector in ['+', '-', 'f']
    if sector == 'f':
        N = L
    else:
        N = L - 1
    H = np.zeros((2 ** N, 2 ** N))
    v = np.zeros(2 ** N)
    
    for i in range(2 ** N):
        v[i]    = 1
        H[:, i] = H_vec(v, L, h, bc, sector)
        v[i]    = 0
    return H