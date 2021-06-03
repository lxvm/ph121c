"""This module defines a tensor train, a scaffold for the mps and mpo classes.

Most operations on the train require declaring an orthogonality center, which
can be done by passing a site in the train or by passing the tag of a quantum
index: both will be in the new orthogonality center.

My convention is that unless the train is right-canonical, the physical index
is always lowered at the orthogonality center.
"""
from copy import deepcopy
from collections import UserList

from .utils import *
from .site  import *


class train (UserList):
    """This class represents sites in a mps/mpo train."""
    def __init__ (self, iterable=()):
        """Create a train from an iterable of sites."""
        super().__init__(e for e in iterable if isinstance(e, site))
        self.center = None

    def __repr__ (self):
        return 'train' + super().__repr__()

    def __setitem__ (self, i, item):
        assert isinstance(item, site)
        self[i] = item

    def split_site (self, site_in, center, trim=None, result=False):
        """Split a site by introducing a bond index (aka SVD)."""
        pos = self.index(site_in)
        if isinstance(center, site):
            if (self.index(center) < pos):
                canon = 'right'
            else:
                canon = 'left'
        elif isinstance(center, int):
            quanta_tags = list( e.tag for e in site_in.get_type(quantum) )
            if any( (center >= e) for e in quanta_tags ) or (center == -1):
                canon = 'left'
            else:
                canon = 'right'
        else:
            raise TypeError(f'center={center} needs to be int or site')
        new_sites = site_in.split(trim=trim, canon=canon)
        # If splitting the center, move it
        if (self.center == site_in):
            if (canon == 'left'):
                self.center = new_sites[-1]
            elif (canon == 'right'):
                self.center = new_sites[0]
            else:
                self.center = new_sites[1]
        del self[pos]
        for e in reversed(new_sites):
            self.insert(pos, e)
        if result:
            return self
    
    def contract_bond (self, bond_in, result=False):
        """Join two sites by contracting a bond index **IN PLACE**."""
        # Error checking for compatibility with Kronecker product implementation
        assert isinstance(bond_in, bond), \
            f'received type {type(bond_in)} but need index.bond'
        left_pos, right_pos = sorted(self.index(e) for e in bond_in.tag)
        assert (right_pos - left_pos == 1), \
            'bond should connect adjacent sites.'
        self[left_pos].contract(self[right_pos])
        # If contracting the center, move it
        if (self.center == self[right_pos]):
            self.center = self[left_pos]
        del self[right_pos]
        if result:
            return self
            
    def max_quantum_tag (self, sight):
        """Return the tag of the largest quantum at site. If None, look left."""
        try:
            return max( e.tag for e in sight.get_type(quantum) )
        except ValueError as ex:
            try:
                return self.max_quantum_tag(self[self.index(sight) - 1])
            except Exception:
                raise ex
        
    def center_tag (self, center):
        """Return the (largest) tag in the train of the desired center."""
        if isinstance(center, site):
            return self.max_quantum_tag(center)
        elif isinstance(center, int):
            return center
        else:
            raise TypeError(f'center={center} needs to be int or site')
        
    def split_quanta (self, center, sites=(), N=1, trim=None, result=False):
        """Split the sites in the train by quanta (default: all sites).
        
        Arguments:
        center :: site or int :: the quantum whose tag makes it the
        orthogonality center, or a site
        N :: int 
        """ 
        if not sites:
            sites = [e for e in self]
        if not N:
            N = 1
        if isinstance(N, int):
            P = [ N for _ in range(len(sites)) ]
        elif hasattr(N, '__len__'):
            assert (len(N) == len(sites))
            P = list(N)
        else:
            raise TypeError('N must be an integer, list or tuple (nested depth 1)')
            
        for i, sight in enumerate(sites):
            pos = self.index(sight)
            new_sites = sight.split_quanta(self.center_tag(center), N=P[i], trim=trim)
            del self[pos]
            for e in reversed(new_sites):
                self.insert(pos, e)
                # Move the center if necessary
                for quant in e.get_type(quantum):
                    if (center == abs(quant.tag)):
                        self.center = self[pos]
                        break
        if (center in [0, -1]):
            self.center = self[center]
        if result:
            return self
    
    def canonize (self, center, result=False):
        """Cast train into canonical form with orthogonality center **IN PLACE**.

        Arguments:
        center :: site or int:: if isinstance site, then if that site is 
        contained in the train, it becomes new orthogonality center.
        If isinstance int, the site with that physical index becomes the center.
        Note: center=0 is right-canonical, and center=-1 is left-canonical.
        """
        center_tag = self.center_tag(center)
        center_ind = 0
        side = 'left'
        i = 0
        while (i < (len(self) - 1)):
            if (side == 'left'):
                center_ind = i
                self[i].reset_pos(center_tag)
                if ((self[i].all_quanta_tags('>', center_tag) ^ (center_tag == -1))
                or (self[i].any_quantum_tag('==', center_tag))):
                    side = 'right'
                    continue
                else:
                    # canonize from the left
                    try:
                        bnd = next(self[i].ind[1].get_type(bond))
                        self.split_site(self[i], center_tag)
                        self.contract_bond(bnd)
                    except StopIteration:
                        pass
            elif (side == 'right'):
                j = center_ind - i - 1
                self[j].reset_pos(center_tag)
                # canonize from the right
                try:
                    bnd = next(self[j].ind[0].get_type(bond))
                    self.split_site(self[j], center_tag)
                    self.contract_bond(bnd)
                except StopIteration:
                    pass
            i += 1
        # Because max of i is len(self) - 2, may need to add the last element
        if (len(self) > 1):
            center_ind += int((center_tag == -1) or (center_tag > len(self) - 1))
        self[center_ind].reset_pos(center_tag)
        self.center = self[center_ind]
        if result:
            return self

    def trim_bonds (self, chi, result=False):
        """Trim the bond rank of the MPS by constant chi **IN PLACE**."""
        for i, sight in enumerate(self[:-1]):
            sight.trim_bonds(self[i+1], chi=chi, result=result)
        if result:
            return self
        
    def merge_bonds (self, sites=(), result=False):
        """Contract bonds between given sites **IN PLACE** (default: all)."""
        bonds = multi_index()
        sites = tuple(sites)
        if not sites:
            sites = self
        for sight in sites:
            for bnd in sight.get_type(typeof=bond):
                if any( e in bnd.tag for e in sites if (e != sight) ):
                    if not (bnd in bonds):
                        bonds.append(bnd)
        for i, bnd in enumerate(bonds):
            self.contract_bond(bnd)
        if result:
            return self
    
    def reset_pos (self, result=False):
        """Reset the indices in each site to match the canonical form **IN PLACE**."""
        center_tag = self.center_tag(self.center)
        for sight in self:
            sight.reset_pos(center_tag)
        if result:
            return self

    def view (self):
        """Return the matrices for viewing purposes."""
        return np.array(
            [ e.mat for e in self ],
            dtype='O'
        )
        
    def size (self):
        """Return total number of coefficients."""
        return sum(e.mat.size for e in self)
    
    def shapes (self):
        """Return a list of the shapes of the mps representations."""
        return [ e.mat.shape for e in self ]
            
    def get_sites (self, iterable):
        """Return the sites corresponding to the quanta tags in iterable."""
        quanta_map = [
            [ e.tag for e in sight.get_type(quantum) ] for sight in self
        ]
        yielded = []
        for e in iterable:
            for i, qset in enumerate(quanta_map):
                if (e in qset) and (i not in yielded):
                    yielded.append(i)
                    yield self[i]
            
    def groupby_quanta_tag (self, tag_group, result=False):
        """Regroup quanta **IN PLACE** so that those in group fuse into a site.
        
        Also makes this new, fused site the orthogonality center.
        
        Arguments:
        tag_group :: list/tuple of consecutive integers :: these are merged
        """
        assert is_consecutive(tag_group)
        # Only split those sites whose quanta are not subsets of tag_group
        # This could be done more selectively by only splitting the quanta in
        # tag group
        sites, N_list = [], []
        for sight in self.get_sites(tag_group):
            quanta_tags = [ e.tag for e in sight.get_type(quantum) ]
            if any( (e not in tag_group) for e in quanta_tags if (e > 0) ):
                sites.append(sight)
                incommon = [ e for e in tag_group if e in quanta_tags ]
                uncommon = [ e for e in quanta_tags if (e > 0) and (e not in tag_group) ]
                chunks = chunk_seq(sorted(uncommon))
                if (len(chunks) <= 2):
                    N_list.append([ len(e) for e in sorted([incommon, *chunks]) ])
                else:
                    raise ValueError('tag_group does not bisect quanta_tags')
        self.split_quanta(self.center, sites, N_list)
        self.merge_bonds(self.get_sites(tag_group))
        new_center = list(self.get_sites(tag_group))
        assert (len(new_center) == 1), 'unable to distinguish site.'
        self.center = new_center[0]
        self.canonize(self.center)
        if result:
            return self