"""This module defines a tensor train, a scaffold for the mps and mpo classes.
"""

from .utils import *
from .site  import *


class train:
    """This class represents sites in a mps/mpo train."""
    def __init__ (self, Nsites=0, data=None):
        """Create a train with data or given number of sites (empty default)."""
        if data:
            self.data = data
        else:
            self.data = [ site() for _ in range(Nsites) ]
    
    def __str__ (self):
        return str(self.data)
    
    def __repr__ (self):
        return repr(self.data)
    
    def __iter__ (self):
        return iter(self.data)
    
    def split_site (self, site_in, trim=None, canon=None, inplace=False):
        """Split a site by introducing a bond index (aka SVD)."""
        pos = self.data.index(site_in)
        new_sites = site_in.split(trim=trim, canon=canon)
        if inplace:
            del self.data[pos]
            for e in reversed(new_sites):
                self.data.insert(pos, e)
        else:
            return self.__class__(
                data=(self.data[:pos] + list(new_sites) + self.data[pos+1:]),
                **{ k: v for k, v in self.__dict__.items() if k != 'data' }
            )
    
    def contract_bond (self, bond_in, inplace=False):
        """Join two sites by contracting a bond index.
        
        Arguments:
        bond_in :: index.bond
        """
        # Error checking for compatibility with Kronecker product implementation
        assert isinstance(bond_in, bond), \
            f'received type {type(bond_in)} but need index.bond'
        ind_a, ind_b = ((e, self.data.index(e)) for e in bond_in.tag)
        assert abs(ind_a[1] - ind_b[1]) == 1, \
            'bond must connect adjacent sites.'        
        # Based on ordering, set the relative sites and their positions
        if (ind_a[1]  < ind_b[1]):
            left_pos = ind_a[1]
            right_pos = ind_b[1]
            new_site = ind_a[0].contract(ind_b[0])
        else:
            left_pos = ind_b[1]
            right_pos = ind_a[1]
            new_site = ind_b[0].contract(ind_a[0])
        # final steps
        if inplace:
            for e in reversed(sorted([left_pos, right_pos])):
                del self.data[e]
            self.data.insert(min([left_pos, right_pos]), new_site)
        else:
            return self.__class__(
                data=(self.data[:left_pos] + [new_site] + self.data[right_pos+1:]),
                **{ k: v for k, v in self.__dict__.items() if k != 'data' }
            )