"""Define the `mpo` class to represent matrix product operators (MPO).
"""

import numpy as np


class mpo:
    """A small class to build mps operators compatible with mps instances.
    
    Also does some sanity checking to make things consistent.
    """
    def __init__ (self, L, d):
        """Instantiate a mpo object acting on L qudits."""
        assert isinstance(L, int) and isinstance(d, int)
        assert (0 < L) and (0 < d)
        self.L = L
        self.d = d
        self.oper = np.empty(self.L, dtype='object')
    
    def __getitem__ (self, i):
        """Retrieve the operator at the ith site."""
        return self.oper[i]
    
    def __setitem__ (self, i, oper):
        """Set the operator at the ith site."""
        assert oper.shape == (self.d, self.d)
        self.oper[i] = oper

    def __iter__ (self):
        """Iterate over the local operators in the larger operator."""
        return iter(self.oper)
    
    def toarray (self):
        """Turn the operator into its dense matrix representation."""
        arr = 1
        for e in self:
            if isinstance(e, np.ndarray):
                arr = np.kron(arr, e)
            else:
                arr = np.kron(arr, np.eye(self.d))
        return arr
