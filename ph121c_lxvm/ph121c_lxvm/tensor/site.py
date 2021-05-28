"""This module defines a site, the basic data unit of a tensor network.

It also defines the bond and quanta class, subclassed from index.
"""

import numpy as np

from .utils   import *
from .indices import *


class quanta (index):
    """This class represents a physical index of a finite Hilbert space."""
    def __init__ (self, iterable, tag):
        """Create a finite index for degrees of freedom in quantum phase space.
        
        Arguments:
        iterable :: iterable :: length gives the dimension of the Hilbert space.
        tag :: int :: this is a label of the state. Start numbering a 1
        """
        assert isinstance(tag, int), \
            'Physical index must be tagged by int.'
        super().__init__(iterable, tag)
    
class bond (index):
    """This class represents a bond/virtual index."""
    def __init__ (self, iterable, tag):
        """Create finite index for a bond index.
        
        Arguments:
        iterable :: iterable :: length gives the dimension of the bond
        tag :: unique seq. of two sites :: the two sites connected by the bond
        """
        assert (len(tag) == 2) and all(isinstance(e, site) for e in tag), \
            'Bond must connect two sites.'
        super().__init__(iterable, tag)

class site:
    # On subclassing np.ndarray
    # https://numpy.org/doc/stable/user/basics.subclassing.html
    # I decided against adding an attribute to an exisiting array because
    # I can't implement so many methods. This should really only be a matrix
    # type, but numpy is deprecating those
    """This class represents a collection of indices stored in a matrix.
    
    The constructor requires a matrix, a row multi_index, and a column
    multi_index. We say that row indices are lowered, and that column indices
    are raised.
    """
    def __init__ (
        self,
        mat=np.ones(
            tuple( 1 for _ in range(2) )
        ),
        ind=multi_index(
            tuple( multi_index() for _ in range(2) )
        ),
    ):
        """Create a new (by default empty) site.
        
        Note that users should adhere to the following index convention:
        little-endian: the first index corresponds to the slowest-changing index
        in the matrix, and the last one corresponds to the fastest-changing
        index. This probably means that the order of indices in the multi_index
        is the reverse order of the physical sites, but this also reflects the
        binary representation of the computational basis.
        
        Arguments:
        mat :: np.ndarray :: (default: np.ones((1, 1)))
        ind :: tensor.multi_index of tensor.multi_index :: 
            (default: multi_index(multi_index(), multi_index()))
        """
        assert isinstance(ind, multi_index)
        assert all(isinstance(e, multi_index) for e in ind)
        assert isinstance(mat, np.ndarray) and (mat.ndim == ind.tag)
        assert all(mat.shape[i] == e.dim for i, e in enumerate(ind)), \
                'matrix and multi_index dimensionality inconsistent.'
        self.ind = ind
        self.mat = mat

    def __str__ (self):
        return ''.join([
            type(self).__name__,
            '(at ', hex(id(self)),
            ', mat.shape=', str(self.mat.shape),
            ', ind=', str(self.ind),
            ')'
        ])
    
    def __repr__ (self, level=2):
        return ''.join([
            type(self).__name__, '(at ', hex(id(self)),
            indent(',\nmat=(' + indent('\n' + repr(self.mat), level) + '\n)', level),
            indent(',\nind=(' + str(self.ind) + '\n)', level),
            ')'
        ])
    
    def __eq__ (self, other):
        return id(self) == id(other)
    
    def test_shape (self):
        """Return True if the shape of matrix matches that of """
        return all(self.mat.shape[i] == e.dim
                   for i, e in enumerate(self.ind))
    
    def get_type (self, *axes, typeof=index):
        """Return pointers to instances of type of index object in the site.
        
        Arguments:
        typeof :: type :: A type to count (default: index)
        axes :: list of ints :: which axes to look along (default: all)
        """
        if not axes:
            axes =  tuple( i for i in range(self.ind.tag) )
        for i in range(self.ind.tag):
            if i in axes:
                yield from self.ind[i].get_type(typeof)
            
    def count_type (self, *axes, typeof=index):
        """Return the count of a total type of index object in the site.
        
        Arguments:
        typeof :: type :: A type to count (default: index)
        axes :: list of ints :: which axes to look along (default: all)
        """
        return len(list(self.get_type(*axes, typeof=typeof)))
        
    def transpose (self, inplace=False):
        """Return a new site which is the transpose of the original."""
        new_mat = self.mat.T
        new_ind = multi_index(tuple(reversed(self.ind)))
        if inplace:
            self.mat = new_mat
            self.ind = new_ind
        else:
            return self.__class__(mat=new_mat, ind=new_ind)

    def raise_index (self, first=False, last=False, row=0, col=1):
        """Pop index from row into column."""
        if first:
            # (ab, c) -> (b, ac)
            which = 0
            assert self.ind[row].dim > 1
            self.mat = self.mat.reshape((
                    self.mat.shape[row] // self.ind[row][which].dim, 
                    self.mat.shape[col] *  self.ind[row][which].dim,
                ), order='F'
            )[:, exchange(self.mat.shape[col], self.ind[row][which].dim)]
            self.ind[col].insert(which, self.ind[row].pop(which))
        if last:
            # (ab, c) -> (a, cb)
            which = -1
            assert self.ind[row].dim > 1
            self.mat = self.mat.reshape((
                    self.mat.shape[row] // self.ind[row][which].dim,
                    self.mat.shape[col] *  self.ind[row][which].dim,
                ), order='F'
            )
            self.ind[col].append(self.ind[row].pop(which))
    
    def lower_index (self, first=False, last=False, row=0, col=1):
        """Pop index from column into row."""
        if first:
            # (a, bc) -> (ba, c)
            which = 0
            assert self.ind[col].dim > 1
            self.mat = self.mat.reshape((
                    self.mat.shape[row] *  self.ind[col][which].dim,
                    self.mat.shape[col] // self.ind[col][which].dim,
                ),
            )[exchange(self.mat.shape[row], self.ind[col][which].dim), :]
            self.ind[row].insert(which, self.ind[col].pop(which))
        if last:
            # (a, bc) -> (ac, b)
            which = -1
            assert self.ind[col].dim > 1
            self.mat = self.mat.reshape((
                    self.mat.shape[row] *  self.ind[1][which].dim,
                    self.mat.shape[col] // self.ind[1][which].dim,
                ), order='F'
            )
            self.ind[row].append(self.ind[col].pop(which))

    def split (self, trim=None, canon=None, full_matrices=False, row=0, col=1):
        """Create a bond via SVD, leading to three new sites: u, s, vh."""
        if canon:
            assert canon in ['left', 'right']
        u, s, vh = np.linalg.svd(self.mat, full_matrices=full_matrices)
        if canon:
            if canon == 'left':
                vh = s[:, None] * vh
            if canon == 'right':
                u = u * s
        left_bond = bond(
            range(u[:, :trim].shape[col]),
            tuple([self.__class__(), self.__class__()]),
        ) 
        right_bond = bond(
            range(vh[:trim, :].shape[row]),
            tuple([self.__class__(), self.__class__()]),
        )
        left_site = self.__class__(
            mat=u[:, :trim],
            ind=self.ind[:row+1] + multi_index((multi_index((left_bond, )), )),
        )
        right_site = self.__class__(
            mat=vh[:trim, :], 
            ind=multi_index((multi_index((right_bond, )), )) + self.ind[col:],
        )
        if not canon:
            center_site = self.__class__(
                mat=np.diag(s)[:trim, :trim],
                ind=multi_index((
                    multi_index((left_bond, )),
                    multi_index((right_bond, )),
                )),
            )
            left_bond.tag = tuple([left_site, center_site])
            right_bond.tag = tuple([center_site, right_site])
            return (left_site, center_site, right_site)
        else:
            left_bond.tag = tuple([left_site, right_site])
            right_bond.tag = left_bond.tag
            return (left_site, right_site)
    
    def contract (self, other, select=None, inplace=False):
        """Contract consecutive bonds between two sites."""
        # Determine bonds that connect this sites' raised index to other's low
        raised, lowered = 1, 0
        self_bonds_pos, other_bonds_pos, bonds_in = list(zip(*(
            (i, other.ind[lowered].index(e), e)
            for i, e in enumerate(self.ind[raised])
        )))
        assert bonds_in, \
            'BondNotFoundError: check other order of sites.'
        assert all(
            bonds_pos == tuple(range(min(bonds_pos), max(bonds_pos) + 1))
            for bonds_pos in [self_bonds_pos, other_bonds_pos]
        ), f'OrderNotImplemented: {self_bonds_pos} and {other_bonds_pos} found.'
        assert all(
            (bonds_pos[0] == 0) or (bonds_pos[-1] == (len(who)))
            for bonds_pos, who in 
            [(self_bonds_pos, self), (other_bonds_pos, other)]
        ), f'OrderNotImplemented: {self_bonds_pos} and {other_bonds_pos} found.'
        # At this point, bond positions are consecutive, sorted, and extremal
        raised_left   = self.ind[raised][:min(self_bonds_pos)]
        raised_right  = self.ind[raised][max(self_bonds_pos)+1:]
        lowered_left  = other.ind[lowered][:min(other_bonds_pos)]
        lowered_right = other.ind[lowered][max(other_bonds_pos)+1:]
        # Casework
        if (raised_left.dim == raised_right.dim 
        == lowered_left.dim == lowered_right.dim):
            # They all equal one, so there is only a bond index (easy case)
            left_mat = self.mat
            right_mat = other.mat
            new_ind = multi_index((
                self.ind[lowered],
                other.ind[raised],
            ))
        elif ((raised_left.dim == lowered_right.dim == 1)
        and ((raised_right.dim > 1) or (lowered_left.dim > 1))):
            left_mat = np.kron(
                np.eye(lowered_left.dim),
                self.mat
            )
            right_mat = np.kron(
                other.mat,
                np.eye(raised_right.dim)
            )
            new_ind = multi_index((
                lowered_left + self.ind[lowered], 
                other.ind[raised] + raised_right,
            ))
        elif ((raised_right.dim == lowered_left.dim == 1)
        and ((raised_left.dim > 1) or (lowered_right.dim > 1))):
            left_mat = np.kron(
                self.mat,
                np.eye(lowered_right.dim)
            )
            right_mat = np.kron(
                np.eye(raised_left.dim),
                other.mat
            )
            new_ind = multi_index((
                self.ind[lowered] + lowered_right, 
                raised_left + other.ind[raised],
            ))
        elif ((raised_left.dim == lowered_left.dim == 1)
        and ((raised_right.dim > 1) or (lowered_right.dim > 1))):
            # This case and the next require an exchange to look like one of the
            # previous cases
            return NotImplemented
        elif ((raised_right.dim == lowered_right.dim == 1)
        and ((raised_left.dim > 1) or (lowered_left.dim > 1))):
            return NotImplemented
        else:
            raise Exception('NoClueError: couldnt decide how to contract bond')
        # The non-bond lowered indices are merged
        if inplace:
            self.mat = left_mat @ right_mat
            self.ind = new_ind
        else:
            return self.__class__(mat=left_mat @ right_mat, ind=new_ind)
    
    def product (self, other, inplace=False):
        """Return a Kronecker product between two sites."""
        assert isinstance(other, self.__class__)
        assert len(self.ind) == len(other.ind), \
            'Requires same ndim: could be generalized?'
        new_mat = np.kron(self.mat, other.mat)
        new_ind = multi_index(tuple(
            self.ind[i] + other.ind[i]
            for i in range(len(self.ind))
        ))
        if inplace:
            self.mat = new_mat
            self.ind = new_ind
        else:
            return self.__class__(mat=new_mat, ind=new_ind)
    
    def exchange (self, axis, part_a, part_b, inplace=False):
        """Exchange the indices in size 2 partition along one axis."""
        assert all(
            isinstance(e, index) and is_consecutive(e, self.ind[axis])
            for e in [part_a, part_b]
        ), 'Check that the two-part partition is consecutive (slice existing)'
        pos_a, pos_b = [ 
            [ self.ind[axis].index(e) for e in f ] for f in [part_a, part_b]
        ]
        if (pos_a > pos_b):
            right = part_a
            left  = part_b
        elif (pos_a < pos_b):
            right = part_b
            left  = part_a
        else:
            raise Exception('Partition sets must be disjoint')
        assert list(range(len(self.ind[axis]))) == (pos_a + pos_b), \
            'Check that partition is correct and disjoint.'
        # Now swap right and left
        new_mat = self.mat.take(exchange(left.dim, right.dim), axis=axis)
        new_ind = self.ind[:axis] + multi_index((right + left, )) + self.ind[axis+1:]
        if inplace:
            self.mat = new_mat
            self.ind = new_ind
        else:
            return self.__class__(mat=new_mat, ind=new_ind)