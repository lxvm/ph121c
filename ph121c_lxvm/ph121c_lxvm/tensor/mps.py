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

from .utils   import *
from .indices import *
from .site    import *
from .train   import *


class mps (train):
    """Class for representing and operating on MPS wavefunctions."""
    def __init__ (self, L, d, center=None, Nsites=0, data=None, **kwargs):
        """Instantiate an mps object for a qudit chain of length L."""
        self.d = d
        self.L = L
        self.center = center
        super().__init__(Nsites, data)
        
    def from_vec (self, v, r, center=-1):
        """Create Left-canonical MPS representation of vector with compression.
        
        v needs to be an ndarray of shape (d ** L, ) whose entries are already
        sorted in the computational basis in the order (0, ..., 0), (0, ..., 1)
        ... (d, ..., d).
        r needs to be an integer function whose minimum is 1 and whose maximum
        on the for inputs of 1 to L is the same as tensor.dim_mps.
        """
        assert test_valid_scheme(r, self.L, self.d)
        assert v.size == self.d ** self.L
        assert np.allclose(1, np.inner(v, v)), 'initial vector not normalized'
        # Create the initial wavefunction
        super().__init__(1)
        self.data[0] = site(
            mat=v[:, None].T,
            ind=multi_index((multi_index(), multi_index(
                [ quanta(range(self.d), i) for i in range(self.L, 0, -1) ]
            )))
        )
        self.center = center
#         # Create the left-canonical mps representation
#         for i in range(self.L):
#             self.split_site(self.data[i])
#             # shift columns to rows, keeping the physical index the outer index
#             w = w.reshape(w.shape[0] * self.d, w.shape[1] // self.d, order='F')
#             a, s, w = np.linalg.svd(w, full_matrices=False)
#             # truncate results
#             self.A[i] = a[:, :r(i+1)]
#             w = s[:self.r(i + 1), None] * w[:self.r(i + 1), :]
    
    def from_tensor (self, A, center):
        """Create a mps from a tensor."""
        pass
    
    def canonize (self, center):
        """Send an MPS into canonical form with orthogonality center j **IN PLACE**.

        center=0 is right-canonical, and center=-1 is left-canonical.
        """
        # sweep left to right, left-canonical
        for i in range(len(self.A[:center])):
            # contract virtual index: (qa, b), (rb, c) -> (rqa, c)
            tmp = np.kron(np.eye(self.d), self.A[i]) @ self.A[i+1]
            # separate physical indices by row, column: (rqa, c) -> (qa, rc)
            tmp = tmp.reshape(
                (tmp.shape[0]//self.d, self.d*tmp.shape[1]), order='F'
            )[:, exchange(self.d, tmp.shape[1])]
            # insert bond: (qa, rc) -> (qa, b), (b, rc)
            u, s, vh = np.linalg.svd(tmp, full_matrices=False)
            # left-canonical: place Schmidt values to the right
            self.A[i] = u
            # restore column physical index to row: (b, rc) -> (rb, c) TODO!!!
            self.A[i+1] = (s[:, None] * vh).reshape(
                (vh.shape[0]*self.d, vh.shape[1]//self.d),
            )[exchange(vh.shape[0], self.d), :]
        # sweep right to left, right canonical
        for i in range(len(self.A[center:-1])):
            # contract virtual index: (qa, b), (rb, c) -> (rqa, c)
            tmp = np.kron(np.eye(self.d), self.A[i]) @ self.A[i+1]
            # separate physical indices by row, column: (rqa, c) -> (qa, rc)
            tmp = tmp.reshape(
                (tmp.shape[0]//self.d, self.d*tmp.shape[1]), order='F'
            )[:, exchange(self.d, tmp.shape[1])]
            # insert bond: (qa, rc) -> (qa, b), (b, rc)
            u, s, vh = np.linalg.svd(tmp, full_matrices=False)
            # right-canonical: place Schmidt values to the left
            self.A[i] = u * s
            # restore column physical index to row: (b, rc) -> (rb, c) TODO!!!
            self.A[i+1] = vh.reshape(
                (vh.shape[0]*self.d, vh.shape[1]//self.d)
            )[exchange(vh.shape[0], self.d), :]
        self.center = center
        
    def contract_bonds (self, bonds=None):
        """Contract the given bond indices and return vector/mps."""
        if isinstance(bonds, type(None)):
            # Contract all bonds and return physical vector
            v = np.zeros(self.d ** self.L)
            for i, index in enumerate(product(np.arange(self.d), repeat=self.L)):
                v[i] = self.get_component(np.array(index)[::-1])
            return v
        else:
            # let a, b, c be bond indices and q, r be physical indices
            # Contract specified bonds:(qa, b), (rb, c) -> (rqa, c)
            tmp = np.kron(np.eye(d), mps_in[i]) @ mps_in[i+1]
            # regroup physical indices by row/column: (rqa,c) -> (qa, rc)
            tmp = tmp.reshape(
                (tmp.shape[0]//d, d*tmp.shape[1]), order='F'
            )[:, exchange(d, tmp.shape[1])]
            
        
    def trim_rank (self, r):
        """Trim the rank of the MPS **IN PLACE**."""
        assert test_valid_scheme(r, self.L, self.d)
        for i in range(self.L):
            self.A[i] = self.A[i].reshape(
                self.r(i), self.d * self.r(i + 1), order='F'
            )[:r(i), :].reshape(
                self.d * min(self.r(i), r(i)), self.r(i + 1), order='F'
            )[:, :r(i + 1)]
        self.r_prev = self.r
        self.r = lambda i: min(self.r_prev(i), r(i))
        
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
            
    def inner (self, B):
        """Take the inner product with another mps wavefunction."""
        assert isinstance(B, mps)
        assert self.L == B.L
        assert self.d == B.d
        val = self.A[0].copy()
        
        for i in range(self.L - 1):
            # collapse the physical index
            val = B.A[i].conj().T @ val
            # collapse the A_i index
            val = np.kron(np.eye(self.d), val) @ self.A[i + 1]
        val = B.A[self.L - 1].conj().T @ val
        if isinstance(val, np.ndarray):
            assert val.size == 1
            return val[0, 0]
        elif isinstance(val, (float, complex)):
            return val

    def mel (self, O, B):
        """Calculate the matrix element <b|O|a>."""
        return O.oper(self).inner(B)
    
    def copy (self):
        """Return a deep copy of self."""
        return deepcopy(self)