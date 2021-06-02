"""Define the `mpo` class to represent matrix product operators (MPO).

These should be input as single or multi-site operators acting on the physical
indices.

A thorough review in section 5 of: https://arxiv.org/abs/1008.3477.
"""

import numpy as np

from .utils   import *
from .indices import *
from .site    import *
from .train   import *
from .mps     import *


class mpo (train):
    """A class to build mps operators compatible with mps instances.
    
    Here, the quanta tags are negative for a raised index, positive if lowered.
    """
    def __init__ (self, iterable=(), L=None, d=None):
        """Instantiate a mpo object acting on L qudits."""
        assert isinstance(L, int) and isinstance(d, int)
        assert (0 < L) and (0 < d)
        super().__init__(iterable)
        self.L = L
        self.d = d

    def set_local_oper (self, oper, left_index):
        """Add a k-local operator at a site in the operator.
        
        Arguments:
        oper :: np.ndarray :: Square matrix of shape (d**k, d**k) for some k<=L
        left_index :: int>0 ::  oper acts on indices left_index + (0, ..., k-1)
        """
        assert (0 < left_index <= self.L)
        assert (oper.shape[0] == oper.shape[1])
        # Get the site and index indices to determine where to insert operator
        quanta_tags = []
        sites_pos = []
        for i, sight in enumerate(self):
            tmp = []
            for e in sight.get_type(quantum):
                if (e.tag > 0):
                    tmp.append(e.tag)
            quanta_tags.extend(sorted(tmp))
            sites_pos.extend(i for _ in range(len(tmp)))
        ndim = get_phys_dim(oper.shape[0], self.d)
        assert ((ndim + left_index - 1) <= self.L)
        assert all(
            (e not in quanta_tags) for e in range(left_index, left_index+ndim)
        ), 'requested indices are already occupied.'
        assert all((e == sorted(e)) for e in [quanta_tags, sites_pos]), \
            'Indices or sites found out of order.'
        # Find the site position which gives the desired index position
        pos = 0
        for i, e in enumerate(quanta_tags):
            pos = sites_pos[i]
            if (left_index < e):
                if (i == 0):
                    break
                assert (sites_pos[i] > sites_pos[i-1]), \
                    'No gap found to insert operator: split the site.'
                break
            pos += 1
        self.insert(pos, site(
            mat=oper,
            ind=multi_index((
                # row index is lowered (+)
                multi_index([
                    quantum(range(self.d), left_index + i)
                    for i in range(ndim-1, -1, -1)
                ]),
                # column index is raised (-)
                multi_index([
                    quantum(range(self.d), -(left_index + i))
                    for i in range(ndim-1, -1, -1)
                ]),
            )),
        ))
    
    def to_arr (self):
        """Return operator as its dense representation in computational basis."""
        self.merge_bonds()
        taken_chunks = [
            sorted( e.tag for e in  sight.get_type(quantum) if (e.tag > 0) )
            for sight in self
        ]
        taken_indices = []
        for chunk in taken_chunks:
            for e in chunk:
                taken_indices.append(e)
        assert (taken_indices == sorted(taken_indices)), \
            'indices are out of order across chunks: debug'
        missing_indices = [ 
            e for e in range(1, self.L + 1) if e not in taken_indices
        ]
        # The taken chunks should correspond to the sites in the train
        missing_chunks = chunk_seq(missing_indices)
        all_chunks = sorted(taken_chunks + missing_chunks)
        # Build the resulting matrix from slowest to fastest index (L to 1)
        if all_chunks[-1] in missing_chunks:
            result = np.eye(self.d ** len(missing_chunks[-1]))
        else:
            result = self[-1].mat
        for i, chunk in enumerate(all_chunks[:-1]):
            if chunk:
                if chunk in missing_chunks:
                    result = np.kron(result, np.eye(self.d ** len(chunk)))
                else:
                    pos = taken_chunks.index(chunk)
                    result = np.kron(result, self[pos].mat)
        return result

    ## These methods must not modify the mps or mpo instance!
    
    def oper (self, mps_in, inplace=False):
        """Apply operator to a tensor.mps instance."""
        assert isinstance(mps_in, mps)
        assert self.L == mps_in.L
        assert self.d == mps_in.d
        return NotImplemented

    def combine (self, mpo_in):
        """Combine two compatible mpo's into one operator."""
        assert isinstance(mpo_in, self.__class__)
        assert self.L == mpo_in.L
        assert self.d == mpo_in.d
        return NotImplemented
    
    def mel (self, mps_a, mps_b):
        """Calculate the matrix element <b|O|a> of this operator."""
        self.oper(mps_a, inplace=False)
        return mps_a.inner(mps_b)
    
    def expval (self, mps_in):
        """Calculate the expectation value <a|O|a> of an operator."""
        return self.mel(mps_in, mps_in)