"""Define the `mpo` class to represent matrix product operators (MPO).
"""

import numpy as np


class mpo:
    """A small class to build mps operators compatible with mps instances.
    
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
        self.oper = np.empty(self.L, dtype='object')
    
    def __getitem__ (self, i):
        """Retrieve the operator at the ith site."""
        return self.oper[i]
    
    def __setitem__ (self, i, oper):
        """Set the operator at the ith site."""
        assert isinstance(i, (int, list, np.ndarray))
        if isinstance(i, int):
            assert oper.shape == (self.d, self.d)
            self.oper[i] = oper
        else:
            assert , 'Require contiguity of indices'
            assert oper.shape == tuple( self.d ** len(i) for _ in range(2) )

    def __iter__ (self):
        """Iterate over the local operators in the larger operator."""
        return iter(self.oper)
    
    def toarray (self):
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
