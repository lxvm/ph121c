"""Module for creating indices related to mps/mpo tensor networks.

In theory, this is helpful because every tensor operation can be represented as
operations on its indices.
"""

from collections import UserList

import numpy as np

from .utils import *


class index (UserList):
    # On subclassing lists:
    # https://docs.python.org/3/library/collections.html
    """Represent an index.
    
    The class is a list with a dim (list length) and tag attribute.
    """
    def __init__ (self, iterable=(), tag=None):
        """Create an index object. Defaults to a dummy index of dimension 0.
        (additive identity).
        
        Arguments:
        iterable :: iterable :: (default: 1)
        tag :: any :: (default: None)
        """
        super().__init__(iterable)
        self.dim = len(self)
        self.tag = tag
        
    def __str__ (self):
        return ''.join([
            type(self).__name__,
            '(dim=', str(self.dim),
            ', tag=', rstr(self.tag),
            ') # at ', hex(id(self)),
        ])
    
    def __repr__ (self, level=2):
        return ''.join([
            type(self).__name__ + '( # at ' + hex(id(self)),
            indent('\ndata=' + repr(self.data), level),
            indent(',\ntag=' + str(self.tag), level),
            '\n)' 
        ])
    
    def update (self):
        """Update the dim in case someone changes the elements."""
        self.dim = len(self)
        
    def __delitem__ (self, i):
        super().__delitem__(i)
        self.update()
        
    def __setitem__ (self, i, item):
        super().__setitem__(i, item)
        self.update()
    
    def __eq__ (self, other):
        return id(self) == id(other)
        
    def __contains__ (self, item):
        return any( item == e for e in self )
    
    def del_type (self, typeof):
        for i in reversed(list(range(len(self)))):
            if isinstance(self[i], typeof):
                del self[i]
            
    def get_type (self, typeof):
        """Returns pointers to all of the type of index in multi_index."""
        for e in self:
            if isinstance(e, typeof):
                yield e
                
    def take (self, indices):
        """Emulate the numpy.take() function."""
        for i in indices:
            yield self[i]
            
    def copy (self):
        """Shallow copy of an index."""
        return self.__class__(self, self.tag)
        
class multi_index (index):
    """Represent an ordered collection of indices."""
    def __init__ (self, iterable=(), tage=None):
        """Create a multi_index from a list of index objects.
        
        With no arguments, creates an empty multi_index of dimension one.
        (multiplicative identity).
        
        Attributes:
        The indices are stored as a list.
        The product of all the dimensions of the indices is stored as `dim`.
        The number of indices is stored as `tag`.
        """
        assert(isinstance(e, index) for e in iterable)
        super().__init__(iterable, tag=len(iterable))
        if not iterable:
            self.dim = 1
        else:
            self.update()
    
    def update (self):
        """Update the dim in case someone changes the elements."""
        self.dim = np.prod([ e.dim for e in self ], dtype='int64')
        self.tag = len(self)
        
    def __delitem__ (self, i):
        super().__delitem__(i)
        self.update()
        
    def __setitem__ (self, i, item):
        assert isinstance(item, index)
        super().__setitem__(i, item)
        self.update()
        
    def __add__ (self, other):
        assert isinstance(other, self.__class__)
        return super().__add__(other)
        self.update()
            
    def __str__ (self):
        return ''.join([
            type(self).__name__,
            '(dim=', str(self.dim),
            ', tag=', rstr(self.tag),
            ', shape=', str(self.shape()), 
            ') # at ', hex(id(self)),
        ])
        
    def shape (self):
        return tuple(e.dim for e in self)
        
    def pop (self, pos=-1):
        item = super().pop(pos)
        self.update()
        return item
        
    def insert (self, pos, item):
        assert isinstance(item, index)
        super().insert(pos, item)
        self.update()
        
    def append (self, item):
        assert isinstance(item, index)
        super().append(item)
        self.update()
        
    def extend (self, iterable):
        super().extend(iterable)
        self.update()
        
    def clear_ones (self):
        """Remove unnecessary indices of dimension one."""
        for i, item in enumerate(self):
            if (len(self) > 1) and (item.dim == 1):
                del self[i]
    
    def tr_ind (self, new_dims, result=True, inplace=False):
        """Return an array of indices with the indices of a truncated multi-index."""
        indices = np.arange(self.dim)[multi_index_tr( 
            tuple(e.dim for e in reversed(self)), new_dims
        )]
        if inplace:
            for i, ind in enumerate(reversed(self)):
                while (ind.dim > new_dims[i]):
                    del ind[-1]
        if result:
            return indices