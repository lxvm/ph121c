"""This module defines a tensor train, a scaffold for the mps and mpo classes.
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

    def split_site (self, site_in, trim=None, canon=None, result=False):
        """Split a site by introducing a bond index (aka SVD)."""
        pos = self.index(site_in)
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
            
    def split_quanta (self, center, sites=(), trim=None, result=False):
        """Split the sites in the train by quanta (default: all sites).
        
        Arguments:
        center :: int :: the quantum whose tag makes it the orthogonality center
        """ 
        if not sites:
            sites = self
        for site in sites:
            pos = self.index(site)
            new_sites = site.split_quanta(center, trim=trim)
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
    
    def canonize (self, center):
        """Cast train into canonical form with orthogonality center **IN PLACE**.

        Arguments:
        center :: site or int:: if isinstance site, then if that site is 
        contained in the train, it becomes new orthogonality center.
        If isinstance int, the site with that physical index becomes the center.
        Note: center=0 is right-canonical, and center=-1 is left-canonical.
        """
        for sight in self:
            sight.reset_pos(center)
        
            if sight.is_right_of(center):
                break
            else:
                pass
#         # Canonize from the left
#         for i in range(len(self.data) - 1):
#             quantas = [ abs(e.tag) for e in self[i].get_type(typeof=quanta) ]
#             print(i, quantas)
#             if (center == -1) or (center > max(quantas)) :
#                 self[i].contract(self[i+1])
#                 self.split_site(self[i], canon='left')
#                 continue
#             elif (center in quantas):
#                 center_site = i
#                 break
#         # Canonize from the right
#         for i in reversed(range(center_site, len(self.mat) - 1)):
#             self[i].contract(self[i+1])
#             self.split_site(self[i], canon='right')
#         self.center = center_site

    def trim_bonds (self, chi, result=False):
        """Trim the bond rank of the MPS by constant chi **IN PLACE**."""
        for i, sight in enumerate(self[:-1]):
            sight.trim_bonds(self[i+1], chi=chi, result=result)
        if result:
            return self
        
    def merge_bonds (self, sites=(), result=False):
        """Contract bonds between given sites **IN PLACE** (default: all)."""
        bonds = multi_index()
        if not sites:
            sites = self
        for site in sites:
            for bond_el in site.get_type(typeof=bond):
                if not (bond_el in bonds):
                    bonds.append(bond_el)
        for i, bond_el in enumerate(bonds):
            self.contract_bond(bond_el)
        if result:
            return self
    
    def group_quanta (self, groupby=('e.tag < 0', 'e.tag > 0'), result=False):
        """Lower all physical indices with + tag, raise those with - tag."""
        for i in range(len(self.data)):
            self[i].group_quanta(groupby)
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
            