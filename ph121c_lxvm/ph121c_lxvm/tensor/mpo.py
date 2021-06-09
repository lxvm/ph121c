"""Define the `mpo` class to represent matrix product operators (MPO).

These should be input as single or multi-site operators acting on the physical
indices.

A thorough review in section 5 of: https://arxiv.org/abs/1008.3477.
"""

from copy import deepcopy

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
        assert (0 < left_index <= self.L), f'0, {left_index}, {self.L}'
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
    
    def __setitem__ (self, i, item):
        """Set a local operator at a site. Shorthand for `set_local_oper`."""
        self.set_local_oper(item, i)
        
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
        for chunk in all_chunks[-2::-1]:
            if chunk:
                if chunk in missing_chunks:
                    result = np.kron(result, np.eye(self.d ** len(chunk)))
                else:
                    pos = taken_chunks.index(chunk)
                    result = np.kron(result, self[pos].mat)
        return result

    def groupby_quanta_tag (self, tag_group):
        """Regroup quanta **IN PLACE** so that those in group fuse into a site.
        
        Also makes this new, fused site the orthogonality center.
        
        Arguments:
        tag_group :: list/tuple of consecutive integers :: these are merged
        """
        # Check for tags not in the operator and insert them as identities
        quanta_tags = []
        for sight in self:
            quanta_tags.append([
                e.tag for ax in (0, 1)
                for e in sight.ind[ax].get_type(quantum)
                if (e.tag > 0)
            ])
        to_fill = []
        for i in tag_group:
            if not any( i in e for e in quanta_tags ):
                to_fill.append(i)
        for chunk in chunk_seq(to_fill):
            if chunk:
                self.set_local_oper(np.eye(self.d ** len(chunk)), min(chunk))
        return super().groupby_quanta_tag(tag_group)
    
    ## These methods must not modify the mps or mpo instance (except reshape)!
    
    def oper (self, tren, inplace=False):
        """Apply operator to a tensor.train instance.
        
        The process of applying the operator without breaking the connectivity
        looks an awful lot like DNA replication :) There is a zipper of sites
        which are copied/unzipped and then wound back together again.
        """
        assert isinstance(tren, train)
        assert self.L == tren.L
        assert self.d == tren.d
        
        prev_train_site = None
        prev_mpo_site = None
        copy_train_site = None
        copy_mpo_site = None
        if inplace:
            output = tren
        else:
            output = deepcopy(tren)
        
        for i in range(len(self)):
            tags = [ e.tag for e in self[i].get_type(quantum) if (e.tag > 0) ]
            output.groupby_quanta_tag(tags)
            center_pos = output.index(output.center)
            # Here is where we apply the operator to the new center
            # we hijack the matching physical indices with bonds
            # we also do this on a copy of each to not cause side-effects
            oper = self[i].copy(prev_mpo_site, copy_mpo_site)
            oper.link_quanta(output.center)
            oper.contract(output.center)
            try: output.center.relink_bonds(output[center_pos - 1], 0)
            except IndexError: pass
            try: output.center.relink_bonds(output[center_pos + 1], 1)
            except IndexError: pass
            output[center_pos] = oper
            output.center = oper
            prev_mpo_site = self[i]
            copy_mpo_site = oper
        if not inplace:
            return output

    def mel (self, tren_a, tren_b):
        """Calculate the matrix element <b|O|a> of this operator."""
        output = self.oper(tren_a).inner(tren_b)
        return np.trace(output.mat)
    
    def expval (self, tren):
        """Calculate the expectation value <a|O|a> of an operator."""
        tags = [ e.tag for s in self for e in s.get_type(quantum) if (e.tag > 0) ]
        if (len(tags) <= 2) and is_consecutive(tags):
            if (len(self) == 1):
                oper = self[0].copy()
            elif (len(self) == 2):
                oper = self[0].copy()
                oper.contract(self[1].copy(self[0], oper))
            else:
                raise IndexError('hard coded sizes do not implement this')
            tren.groupby_quanta_tag(tags)
            output = train([tren.center.copy(), oper])
            output[0].transpose()
            output[0].conj()
            output[0].link_quanta(output[1])
            output.contract_bond(next(
                e for e in output[0].ind[1].get_type(bond) if output[1] in e.tag
            ))
            pos = tren.index(tren.center)
            output.append(tren.center.copy(tren[pos - 1 + 2 * (pos == 0)], output[0]))
            output[0].link_quanta(output[1])
            output[1].relink_bonds(output[0], 0)
            output.contract_bond(next(
                e for e in output[0].ind[1].get_type(bond) if output[1] in e.tag
            ))
            return np.trace(output[0].mat)
        else:
            return self.mel(tren, deepcopy(tren))