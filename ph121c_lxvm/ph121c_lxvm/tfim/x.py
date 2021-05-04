"""Provides Transverse-Field Ising Model Hamiltonians in x basis.

Exploits the Ising symmetry sectors to reduce the computational cost.
The Hamiltonians implement either the + or - symmetry sector.

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


def mylookup (n):
    """This is the same as my Fortran function parity_diag"""
    if n > 1:
        table = mylookup(n-1)
        return np.append(table, table ^ 1)
    else:
        return np.arange(2)

# Lookup table of size 256
parity_lookup = mylookup(8)

def poppar (n):
    """Calculate the parity of the population of bits in an integer
    https://graphics.stanford.edu/~seander/bithacks.html
    """
    n ^= n >> 16
    n ^= n >> 8
    return parity_lookup[n & 0xff]

vpoppar = np.vectorize(poppar, otypes=[int])

def convert (n, at, to):
    """Transform indicies to and from the symmetry sectors"""
    if at == 'full':
        return (n - n % 2) // 2
    elif at == '+':
        return 2 * n + vpoppar(n)
    elif at == '-':
        return 2 * n + vpoppar(n) ^ 1

def H_row (L, h, row, bc='o', sector='+'):
    """Compute row of matrix elements of H as 3 lists for COO format"""
    # Matrix elements in entry, row, column format
    m_el = ([], [], [])
    def append_entry (*entry):
        for i, e in enumerate(m_el):
            e.append(entry[i])
    # Relate sector index to full index
    full = convert(row, sector, 'full')
    # sign flips, sigma x
    diag = 0
    for i in range(L):
        diag -= h*(2 * (((full & (2 ** i)) << 1) == (full & (2 ** (i+1)))) - 1)
    append_entry(diag, row, row)
    # spin flips, sigma z
    for i in range(L-1):
        append_entry(-1 , row, convert(full ^ (3 * (2 ** i)), 'full', sector))
    if bc == 'c':
        append_entry(-1 , row, convert(full ^ ((2 ** (L-1))+1), 'full', sector))
    return m_el

def H_sparse (L, h, bc='o', sector='+'):
    """Return the Hamiltonian in sparse CSR format"""
    # Matrix elements in entry, row, column format
    m_el = ([], [], [])
    def extend_entry (*entries):
        for i, e in enumerate(m_el):
            e.extend(entries[i])
    for i in range(2 ** (L-1)):
        extend_entry(*H_row(L, h, i, bc, sector))
    H = coo_matrix((m_el[0], m_el[1:]), shape=(2**(L-1), 2**(L-1)))    
    return H.tocsr()

def H_vec (L, h, v, bc='o', sector='+'):
    """Compute H|v> efficiently"""
    w       = np.zeros(v.shape)
    indices = np.arange(2 ** (L-1))
    # Relate sector index to full index
    full = convert(indices, sector, 'full')
    ### spin flips, sigma z
    for i in range(L):
        w -= h*v * (2 * (((full & (2 ** i)) << 1) == (full & (2 ** (i+1)))) - 1)
    ### sign flips, sigma x
    for i in range(L-1):
        w[convert(full ^ (3 * (2 ** i)), 'full', sector)] -= v
    if bc == 'c':
        w[convert(full ^ ((2 ** (L-1))+1), 'full', sector)] -= v
    return w

def L_vec (L, h, bc='o', sector='+'):
    """Return a Linear Operator wrapper to compute H|v>"""
    def H_vec_L_h (v):
        return H_vec(L, h, v, bc, sector)
    return LinearOperator((2**(L-1), 2**(L-1)), matvec=H_vec_L_h)

def H_dense (L, h, bc='o', sector='+'):
    """Return the dense Hamiltonian (mostly for testing)"""
    H = np.zeros((2 ** (L-1), 2 ** (L-1)))
    v = np.zeros(2 ** (L-1))
    for i in range(2 ** (L-1)):
        v[i]    = 1
        H[:, i] = H_vec(L, h, v , bc, sector)
        v[i]    = 0
    return H