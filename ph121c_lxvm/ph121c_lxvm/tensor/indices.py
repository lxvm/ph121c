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
            '(at ', hex(id(self)),
            ', dim=', str(self.dim),
            ', tag=', rstr(self.tag),
            ')'
        ])
    
    def __repr__ (self, level=2):
        return ''.join([
            indent('\n' + type(self).__name__ + '(at ' + hex(id(self)), level),
            indent('\ndata=' + super().__repr__(), 2*level), ',',
            indent('\ndim=' + repr(self.dim), 2*level), ',',
            indent('\ntag=' + rstr(self.tag), 2*level), ',',
            indent('\n)', level), 
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
            
class multi_index (index):
    """Represent an ordered collection of indices."""
    def __init__ (self, iterable=()):
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
    
    def __str__ (self):
        return ''.join([
            super().__str__()[:-1], ', shape=', str(self.shape()), ')'
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
    
    def slice_ind (self):
        """Slice this index with subsets of the subindices."""
        return NotImplemented