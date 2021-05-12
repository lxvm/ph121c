"""Implement matrix-product state factorization and operations.

Conventional to use left-canonical MPS.
For reference: https://arxiv.org/pdf/1008.3477.pdf.
"""

import numpy as np
import scipy.sparse.linalg as sla

from .. import tests, tfim


def dim_mps (i, L, d):
    """Calculate the maximal rank for an svd at bond index i in mps"""
    if i <= (L + 1) / 2:
        return d ** i
    else:
        return d ** (L + 1 - i)
    
def test_valid_scheme (r, L, d):
    """Tests whether a rank function is a valid mps approximation scheme."""
    prev_val = 1
    for i in range(L + 1):
        if not (1 <= r(i, L, d) <= d * prev_val):
            return False
        prev_val = d * r(i, L, d)
    return True

# def index_mps (r, bonds, pos=None, d):
#     """Calculate the index in the storage vector of selected bond indices.
    
#     Bonds needs to be a 1d array enumerating the desired bonds.
#     Pos needs to be a 3d array with each row containing the desired indices,
#     both physical and virtual, of the corresponding bond. If any physical or 
#     virtual index is none, then a slice of that dimension is returned.
#     """
#     assert bonds.size == pos.shape[0]
#     indices = []
    
#     for i, bond in enumerate(bonds):
#         if pos[i] == None:
#             indices.append(np.arange())
#     return indices
            
# def tr_merge (s, w, r, d):
#     """Truncates virtual index and merges the next available physical index."""
#     index = np.repeat(np.arange(r), d) \
#         + np.repeat(np.arange(0, d*r, r).reshape(d*r, 1), r, axis=0).ravel()
#     return (s[:r, None] * w[:r, :]).reshape((r*d, w.shape[1] // d), order='F')[index, :]

def regroup (w, d):
    """Merges the next available physical index from columns to rows."""
    assert (w.shape[1] % d == 0) and (w.shape[1] > 0)

    index = np.repeat(np.arange(w.shape[0]), d) \
        + w.shape[0] * np.repeat(np.arange(0, d)[:, None].T, w.shape[0], axis=0).ravel()
    return w.reshape((w.shape[0] * d, w.shape[1] // d), order='F')[index, :]
                      
def convert_vec_to_mps (v, r, L, d):
    """Return a MPS representation of a wavefunction."""
    assert v.size == d ** L
    assert test_valid_scheme(r, L, d)
    A    = np.empty(L, dtype='object')
    w    = v[:, None].T

    for i in range(L):
        # shift columns to rows
        w    = regroup(w, d)
        a,s,w= np.linalg.svd(w, full_matrices=False)
        # truncate results
        A[i] = a[:, :r(i+1, L, d)]
        w    = s[:r(i+1, L, d), None] * w[:r(i+1, L, d), :]
    print(w)
    return A

class my_mps:
    """Class for representing and operating on MPS wavefunctions.
    
    Ok with qudits already in the computational basis.
    """
    def __init__ (self, v, r, L, d=2):
        """Create an mps representation of vector with compression scheme."""
        # desired feature: argument: order=[perm(range(L))]
        # to construct the mps from a different order of the virtual indices
        self.v = v
        self.d = d
        self.r = r
        self.L = L
        self.A = convert_vec_to_mps(v, r, L, d)

    def contract_bonds (self):
        """Contract the bond indices and return the approximate physical vector."""
        v = np.zeros(2 ** self.L)
        
        for i in range(2 ** self.L):
            v[i] = self.get_component(i)
        return v
        
#     def lower_rank (self, r):
#         """Return a new my_mps with lower rank."""
#         assert 1 <= r < self.r
#         return my_mps(
#             self.v,
#             self.L,
#             r,
#             self.d,
#             self.A.reshape((self.A.size // self.d, self.d))[:, :r].ravel(),
#         )
    
    def get_component (self, i):
        """Calculate a single component of the physical index."""
        assert 0 <= i <= ((2 ** self.L) - 1)
        # the indices of self.A are big-endian
        # the computational basis of integers is little-endian
        index = bin(i)[:1:-1].ljust(self.L, '0')
        coef  = np.ones(1).reshape(1, 1)
        
        for j, e in enumerate(index):
            coef = coef @ self.A[j][
                (1 + int(e)) * np.arange(min(
                    self.r(j,     self.L, self.d),
                    self.r(j + 1, self.L, self.d)
                ))
            ]
        return coef
        