"""Define the `mpo` class to represent matrix product operators (MPO).
"""

import numpy as np

from .train import train
from .mps import mps


class mpo (train):
    """A class to build mps operators compatible with mps instances.
    
    Also does some sanity checking to make things consistent.
    
    This uses the convention that the operators are ordered from the fastest-
    changing physical index, 0, to the slowest changing index, L - 1.
    """
    def __init__ (self, L, d):
        """Instantiate a mpo object acting on L qudits."""
        assert isinstance(L, int) and isinstance(d, int)
        assert (0 < L) and (0 < d)
        self.L = L
        self.d = d
        self.oper = []
        self.info = []
    
    def vacant (self, rng):
        """Check that requested sites are available to fill."""
        return NotImplemented
    
    def set_local_oper (self, oper, left_site, separate=False):
        """Add a k-local operator at a site in the operator.
        
        Optionally, if k>1, the operator can be 
        """
        assert all(e % d == 0 for e in oper.shape)
        assert oper.shape[0] == oper.shape[1]
        assert vacant(range(left_site, left_site + get_phys_dim(oper.shape[0], self.d)))
        pass
    
    def from_arr (self):
        """."""
        pass
    
    def to_arr (self):
        """Turn the operator into its dense matrix representation.
        
        This reconstructs the matrix so that the index in operator 0 is the
        fastest-changing and that the index in operator L - 1 is slowest-changing.
        That is, O = kron(O_{L-1}, kron( ... kron(O_1, O_0))).
        """
        arr = 1
        for e in self:
            if isinstance(e, np.ndarray):
                arr = np.kron(e, arr)
            else:
                arr = np.kron(np.eye(self.d), arr)
        return arr

    def oper (self, mps_in):
        """Apply operator to a tensor.mps instance **IN PLACE**."""
        assert isinstance(mps_in, mps)
        assert self.L == mps_in.L
        assert self.d == mps_in.d
        
        pass

    def combine (self, oper):
        """Combine two compatible mpo's into one operator."""
        assert isinstance(oper, mpo)
        pass
    
    def mel (self, mps_a, mps_b):
        """Calculate the matrix element <b|O|a> of this operator."""
        return self.oper(mps_a).inner(mps_b)
    
    def expval (self, mps_in):
        """Calculate the expectation value <a|O|a> of an operator."""
        return self.mel(mps_in, mps_in)