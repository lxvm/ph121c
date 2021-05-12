"""Implement matrix-product state factorization and operations.

Conventional to use left-canonical MPS.
For reference: https://arxiv.org/pdf/1008.3477.pdf.

Other good explanations available here:
https://tensornetwork.org/mps/
"""
from itertools import product

import numpy as np
import scipy.sparse.linalg as sla

from .. import tests, tfim


def dim_mps (i, L, d):
    """Calculate the maximal rank for an svd at bond index i in mps"""
    if i <= L / 2:
        return d ** i
    else:
        return d ** (L - i)
    
def test_valid_scheme (r, L, d):
    """Tests whether a rank function is a valid mps approximation scheme."""
    prev_val = 1
    for i in range(L + 1):
        if not (1 <= r(i) <= min(prev_val, dim_mps(i, L, d))):
            return False
        prev_val = d * r(i)
    return True
   
def convert_vec_to_mps (v, r, L, d):
    """Return a MPS representation of a wavefunction."""
    assert v.size == d ** L
    assert test_valid_scheme(r, L, d)
    A    = np.empty(L, dtype='object')
    w    = v[:, None].T

    for i in range(L):
        # shift columns to rows, keeping the physical index contiguous
        w    = w.reshape(w.shape[0] * d, w.shape[1] // d, order='F')
        a,s,w= np.linalg.svd(w, full_matrices=False)
        # truncate results
        A[i] = a[:, :r(i+1, L, d)]
        w    = s[:r(i+1, L, d), None] * w[:r(i+1, L, d), :]
    return A

class my_mps:
    """Class for representing and operating on MPS wavefunctions.
    
    Ok with qudits already in the computational basis.
    """
    def __init__ (self, v, r, L, d, A=None):
        """Create an mps representation of vector with compression scheme.
        
        v needs to be an ndarray of shape (d ** L, ) whose entries are already
        sorted in the computational basis in the order (0, ..., 0), (0, ..., 1)
        ... (d, ..., d).
        r needs to be an integer function whose minimum is 1 and whose maximum
        on the for inputs of 1 to L is the same as mps.dim_mps.
        """
        # desired feature: argument: order=[perm(range(L))]
        # to construct the mps from a different order of the virtual indices
        assert test_valid_scheme(r, L, d)
        assert v.size == d ** L
        self.v = v
        self.d = d
        self.L = L
        self.r = r
        self.dim = lambda i: dim_mps(i, self.L, self.d)
        if not A:
            self.A = np.empty(L, dtype='object')
            w = self.v[:, None].T
            # Create the mps representation
            for i in range(L):
                # shift columns to rows, keeping the physical index contiguous
                w    = w.reshape(w.shape[0] * self.d, w.shape[1] // d, order='F')
                a,s,w= np.linalg.svd(w, full_matrices=False)
                # truncate results
                self.A[i] = a[:, :r(i+1)]
                w    = s[:self.r(i + 1), None] * w[:self.r(i + 1), :]
        else:
            self.A = A
            
    def contract_bonds (self):
        """Contract the bond indices and return the approximate physical vector."""
        v = np.zeros(self.d ** self.L)
        
        for i, index in enumerate(product(np.arange(self.d), repeat=self.L)):
            v[i] = self.get_component(np.array(index)[::-1])
        return v
        
    def lower_rank (self, r):
        """Lower the rank of the MPS **IN PLACE**"""
        assert test_valid_scheme(r, self.L, self.d)
        assert all(r(i + 1) <= self.r(i + 1) for i in range(self.L))
        for i in range(self.L):
            self.A[i] = self.A[i].reshape(
                self.r(i), self.d * self.r(i + 1), order='F'
            )[:r(i), :].reshape(
                self.d * r(i), self.r(i + 1), order='F'
            )[:, :r(i + 1)]
        self.r = r

    def get_component (self, index):
        """Calculate a single component of the physical index.
        
        Index is an ndarray of length L with the physical indices in big-endian
        order and with integer entries ranging from 0 to d - 1
        """
        assert index.size == self.L
        assert np.all(0 <= index) and np.all(index < self.d)
        coef  = np.ones(1).reshape(1, 1)
        
        for j, e in enumerate(index):
            coef = coef @ self.A[j][e * self.r(j) + np.arange(self.r(j))]
        return coef
        
    def size (self):
        """Return total number of coefficients."""
        return sum(e.size for e in self.A)
    
    def shapes (self):
        """Return a list of the shapes of the mps representations."""
        return [ e.shape for e in self.A ]
    
    def oper (self, sites, oper):
        """Apply an operator to any number of sites **IN PLACE**."""
        for i in sites:
            self.A[i] = np.kron(np.eye(oper, self.r(i))) @ self.A[i]
            
    def inner (self, B):
        """Take the inner product with another mps wavefunction."""
        assert isinstance(B, my_mps)
        assert self.L == B.L
        assert self.d == B.d
        val = self.A[0]
        
        for i in range(self.L - 1):
            # collapse the physical index
            val = np.conj(np.transpose(B.A[i])) @ val
            # collapse the A_i index
            val = np.kron(np.eye(self.d), val) @ self.A[i + 1]
        return np.conj(np.transpose(B.A[self.L - 1])) @ val
        
        