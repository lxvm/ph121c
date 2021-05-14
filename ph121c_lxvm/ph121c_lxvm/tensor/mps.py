"""Define the `mps` class to create and operate on MPS wavefunctions.

The instances can represent qudit systems of length L when d is same at all sites.
In addition, the user must supply a rank function when creating an `mps`
instance, which calculates the bond dimensions, that must return integer
An example rank function with constant maximal rank is `bond_rank`.
The mathematical prescription of this function is described in the assignment 2
part 5 jupyter notebook.

The conventions in use (left-canonical MPS) and a thorough introduction here:
https://arxiv.org/pdf/1008.3477.pdf.
"""
# Desired feature/knowledge: make these subclasses of numpy.ndarray
# https://numpy.org/doc/stable/user/basics.subclassing.html

from copy import deepcopy
from itertools import product

import numpy as np
import scipy.sparse.linalg as sla


def dim_mps (i, L, d):
    """Calculate the maximal rank for an svd at bond index i in mps"""
    if i <= L / 2:
        return d ** i
    else:
        return d ** (L - i)

def bond_rank (chi, L, d):
    """Return a function to calculate the bond ranks with a constant truncation."""
    return lambda i: max(1, min(chi, dim_mps(i, L, d)))

def test_valid_scheme (r, L, d):
    """Tests whether a rank function is a valid mps approximation scheme."""
    prev_val = 1
    for i in range(L + 1):
        if not (1 <= r(i) <= min(prev_val, dim_mps(i, L, d))):
            return False
        prev_val = d * r(i)
    return True

class mps:
    """Class for representing and operating on MPS wavefunctions."""
    def __init__ (self, v, r, L, d, A=None):
        """Create an MPS representation of vector with compression scheme.
        
        v needs to be an ndarray of shape (d ** L, ) whose entries are already
        sorted in the computational basis in the order (0, ..., 0), (0, ..., 1)
        ... (d, ..., d).
        r needs to be an integer function whose minimum is 1 and whose maximum
        on the for inputs of 1 to L is the same as tensor.dim_mps.
        """
        # desired feature: argument: order=[perm(range(L))]
        # to construct the mps from a different order of the virtual indices
        # could do this with w = schmidt.permute(v, order, L, inverse=True).T
        assert test_valid_scheme(r, L, d)
        assert v.size == d ** L
        assert np.allclose(1, np.inner(v, v)), 'initial vector not normalized'
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
                # shift columns to rows, keeping the physical index the outer index
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
    
    def oper (self, local_oper):
        """Apply an operator to any physical indices and return a new mps object.
        
        The way to create an operator is to instantiate one with
        np.empty(L, dtype='object')
        and then to populate each index with a d x d matrix.
        Unset values remain None and are equivalent to the identity operator.
        """
        if isinstance(local_oper, mpo):
            assert self.L == local_oper.L
            assert self.d == local_oper.d
        elif isinstance(local_oper, np.ndarray):
            assert local_oper.dtype == 'O'
            assert local_oper.size == self.A.size
        else:
            raise UserWarning('matrix product operator is of unsupported type')
        B = deepcopy(self)
        
        for i, oper in enumerate(local_oper):
            if np.any(oper) and isinstance(oper, np.ndarray):
                B.A[i] = np.kron(oper, np.eye(self.r(i))) @ self.A[i]
        return B
            
    def inner (self, B):
        """Take the inner product with another mps wavefunction."""
        assert isinstance(B, mps)
        assert self.L == B.L
        assert self.d == B.d
        val = self.A[0].copy()
        
        for i in range(self.L - 1):
            # collapse the physical index
            val = np.conj(np.transpose(B.A[i])) @ val
            # collapse the A_i index
            val = np.kron(np.eye(self.d), val) @ self.A[i + 1]
        val = np.conj(np.transpose(B.A[self.L - 1])) @ val
        assert val.size == 1
        return val[0, 0]
        
    def mel (self, O, B):
        """Calculate the matrix element <b|O|a>."""
        return self.oper(O).inner(B)
        
    def expval (self, O):
        """Calculate the expectation value <a|O|a>."""
        return self.mel(O, self)