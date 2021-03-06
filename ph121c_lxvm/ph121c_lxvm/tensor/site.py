"""This module defines a site, the basic data unit of a tensor network.

It also defines the bond and quantum class, subclassed from index.
The operations/methods on sites are inherently local to the site,
except that they update bond references where needed.

For a picture of a site, see figure 25 of Schollwoeck
"""

from copy import copy as shallow_copy

import numpy as np

from .utils   import *
from .indices import *


site_reshapes = dict(
    # Each tuple gives the values:
    # give, get, row_op, col_op, order, slc, insert, pop, left, right
    tall=dict(
        ff=(1, 0, '*', '//', 'C', ':N', 'i', '0', 'N:', ':N'),
        fl=(1, 0, '*', '//', 'C', ':N', 'self.ind[get].tag', '0', '', ''),
        lf=(1, 0, '*', '//', 'F', '-N:', '0', '-1', '', ''),
        ll=(1, 0, '*', '//', 'F', '-N:', 'self.ind[get].tag - i', '-1', '-N:', ':-N'),
    ),
    flat=dict(
        ff=(0, 1, '//', '*', 'F', ':N', 'i', '0', 'N:', ':N'),
        fl=(0, 1, '//', '*', 'F', ':N', 'self.ind[get].tag', '0', '', ''),
        lf=(0, 1, '//', '*', 'C', '-N:', '0', '-1', '', ''),
        ll=(0, 1, '//', '*', 'C', '-N:', 'self.ind[get].tag - i', '-1', '-N:', ':-N'),
    ),
)

site_trims = (
    # variables correspond to reshapes and slices: aspect_a, aspect_b, trimmings
    ('flat', 'tall', 'sight.mat[:chi, :]'),
    ('tall', 'flat', 'sight.mat[:, :chi]'),
)

class quantum (index):
    """This class represents a physical index of a finite Hilbert space."""
    def __init__ (self, iterable, tag):
        """Create a finite index for degrees of freedom in quantum phase space.
        
        Arguments:
        iterable :: iterable :: length gives the dimension of the Hilbert space.
        tag :: int :: this is a label of the state. Start numbering a 1
        """
        assert isinstance(tag, (int, np.int64)) and (tag != 0), \
            f'Physical index must be tagged by nonzero int.'
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
        mat=np.ones((1, 1)),
        ind=multi_index((multi_index(), multi_index())),
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
        assert isinstance(mat, np.ndarray) and (mat.ndim == ind.tag == 2)
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
            type(self).__name__, '( # at ',  hex(id(self)),
            indent('\n' + repr(self.mat) + ',', level),
            indent('\n' + str(self.ind), level),
            '\n)'
        ])
    
    def __eq__ (self, other):
        return id(self) == id(other)
       
    def test_shape (self):
        """Return True if the shape of matrix matches that of multi-index."""
        return all( self.mat.shape[i] == e.dim for i, e in enumerate(self.ind) )
       
    def transpose (self):
        """Transpose the site **IN PLACE**."""
        self.mat = self.mat.T
        self.ind = multi_index(tuple(reversed(self.ind)))

    def conj (self):
        """Conjugate the entries of the matrix **IN PLACE**."""
        self.mat = self.mat.conj()

    def get_type (self, typeof=index, axes=(0, 1)):
        """Return pointers to instances of type of index object in the site.
        
        Arguments:
        typeof :: type :: A type to count (default: index)
        axes :: tuple of ints :: which axes to look along (default: (0, 1))
        """
        for axis in axes:
            yield from self.ind[axis].get_type(typeof)
        
    def all_quanta_tags (self, comp, center, tags=None):
        """Returns True if all quanta tags in site satisfy inequality with center.
        
        Arguments:
        comp :: str eg: ['>', '<', '>=', '<='] :: binary comparison for integers
        center :: int :: compare the quantum tags against this
        """
        if not tags:
            tags = [ e.tag for e in self.get_type(quantum) ]
        return all( eval(f'abs(e) {comp} {center}') for e in tags )
            
    def any_quantum_tag (self, comp, center, tags=None):
        """True if any quantum tag in site satisfies comparison with center.
        
        Arguments:
        comp :: str eg: ['>', '<', '>=', '<='] :: binary comparison for integers
        center :: int :: compare the quantum tags against this
        """
        if not tags:
            tags = [ e.tag for e in self.get_type(quantum) ]
        return any( eval(f'abs(e) {comp} {center}') for e in tags )
     
    def link_quanta (self, lowered):
        """Replace matching quanta with a bond **IN PLACE**."""
        bond_pos = (
            k for k, e in enumerate(lowered.ind[0].get_type(quantum))
            if (abs(q.tag) == abs(e.tag)) and (q.dim == e.dim)
        )
        for i, q in enumerate(self.ind[1].get_type(quantum)):
            try:
                bnd = bond(range(q.dim), tag=[self, lowered])
                self.ind[1][i] = bnd
                lowered.ind[0][next(bond_pos)] = bnd
            except StopIteration:
                print('link did not succeed')
                pass
    
    def relink_bonds (self, other, axis):
        """Update bonds in self pointing to other in other."""
        other_bonds_pos = [
            i for i, e in enumerate(other.ind[axis ^ 1]) if isinstance(e, bond)
        ]
        for i, bnd in enumerate(self.ind[axis].get_type(bond)):
            if other in bnd.tag:
                other.ind[axis ^ 1][other_bonds_pos[i]] = bnd
                        
    def copy (self, old=None, new=None):
        """Return a deep enough copy with updated bond tags.
        
        Arguments:
        old :: site :: external bond target which should be changed to new
        new :: site :: new bond target
        look :: tuple subset of (0, 1) :: if an axis is in this tuple, bonds on
        that axis will be updated
        """
        output = shallow_copy(self)
        output.ind = self.ind.copy()
        for axis in (0, 1):
            output.ind[axis] = self.ind[axis].copy()
            for i, ind in enumerate(output.ind[axis]):
                output.ind[axis][i] = ind.copy()
                if isinstance(ind, bond):
                    output.ind[axis][i].tag = output.ind[axis][i].tag.copy()
                    # update bond with reference to copied site
                    for j, e in enumerate(output.ind[axis][i].tag):
                        if (e == self):
                            output.ind[axis][i].tag[j] = output
                        if (e == old):
                            output.ind[axis][i].tag[j] = new
        return output

    def reshape (self, how, which, N=1):
        """Pop N indices from column into row or vice versa **IN PLACE**.

        Arguments:
        how :: str :: must be 'tall' (row to col) or 'flat' (col to row)
        which :: str :: must be 'ff', 'fl', 'lf', or 'll' (first/last * beg/end)
        N :: int :: how many consecutive indices to move (default: 1)
        """
        if (N < 1):
            return
        try:
            give, get, row_op, col_op, order, slc, insert, pop, left, right = \
                site_reshapes[how][which]
        except KeyError:
            raise KeyError('check docs for allowed arguments.')
        assert (self.ind[give].tag >= N), \
            'not enough indices for requested reshape'
        self.mat = self.mat.reshape(eval(f"""(
        self.mat.shape[0] {row_op} self.ind[{give}][{slc}].dim,
        self.mat.shape[1] {col_op} self.ind[{give}][{slc}].dim,
        )"""), order=order)
        
        for i in range(N):
            self.ind[get].insert(eval(insert), self.ind[give].pop(eval(pop)))
        if (left and right):
            self.mat = self.mat.take(
                exchange(
                    eval(f'self.ind[{get}][{left}].dim'),
                    eval(f'self.ind[{get}][{right}].dim'),
                ), 
                axis=get
            ) 
        
    def permute (self, perm, axis):
        """Permute one multi-index of the site **IN PLACE**.
        
        Arguments:
        perm :: multi_index :: this will become the new multi-index
        """
        assert isinstance(perm, multi_index)
        if not (perm.data == self.ind[axis].data):
            # The fastest index is stored last in these multi-indices
            inds, dims = [], []
            toy = multi_index(list(reversed(perm)))
            for e in reversed(self.ind[axis]):
                inds.append(toy.index(e))
                dims.append(toy[inds[-1]].dim)
            # multi_index_perm takes the fastest index first
            new_mat = self.mat.take(multi_index_perm(
                np.asarray(dims), np.asarray(inds)), axis=axis)
            self.mat = new_mat
            self.ind[axis] = perm
            
    def trim_bonds (self, other, chi):
        """Limit the max bond dimension between two sites to chi **IN PLACE**."""
        bonds = [ e for e in self.get_type(bond) if (other in e.tag) ]
        assert (len(bonds) == 1), \
            f'Trying to trim {len(bonds)} bonds: if > 1, SVD instead.'
        bnd = bonds[0]
#         self.relink_bonds(other, 0)
        self.relink_bonds(other, 1)
#         other.relink_bonds(self, 0)
        for i, sight in enumerate([self, other]):
            # Find the axis where the bond is
            for axis in (0, 1):
                if any( bnd == e for e in sight.ind[axis].get_type(bond) ):
                    aspect_a, aspect_b, trimmings = site_trims[axis]
#                     print('ax', axis)
                    break
            # Bond should be the last entry in the axis
            N = len(sight.ind[axis][:-1])
#             print(repr(sight.ind))
            sight.reshape(aspect_a, 'fl', N=N)
            # Bond is distinguished, now trim it
            sight.mat = eval(trimmings)
            sight.reshape(aspect_b, 'lf', N=N)
        # Trim bond as well
        if chi:
            while (len(bnd) > int(chi)):
                del bnd[-1]

    def groupby_quanta_tag (self, groupby=('e.tag < 0', 'e.tag > 0')):
        """Group physical indices with a filter **IN PLACE**.
        
        By default will lower quantum with + sign and raise with - sign.
        The argument `groupby` is a tuple of strings indexed by `axis`
        that will be evaluated in a boolean context (if statement).
        It may also use `e` which will be only instances of quantum.
        Statements which evaluate to true will move that index to other axis.
        """
        for axis, aspect in enumerate(['flat', 'tall']):
            left = multi_index()
            right = multi_index()
            bonds = multi_index()
            for e in self.ind[axis]:
                if isinstance(e, quantum): 
                    if (eval(groupby[axis])):
                        right.append(e)
                    else:
                        left.append(e)
                elif isinstance(e, bond):
                    bonds.append(e)
                else:
                    raise TypeError(f'Unrecognized index type: {type(e)}.')
            if (left or right):
                self.permute(left + bonds + right, axis=axis)
                self.reshape(aspect, 'lf', N=right.tag)
    
    def sort_ind (self, axes=(0, 1)):
        """Sorts the quanta in the site in the computational basis **IN PLACE**.
        
        Also place bonds into fastest position:
        E.g. multi_index((a), (1), (2), (-2), (-1), (b))
        -->  multi_index((2), (-2), (1), (-1), (a), (b))
        """
        for axis in axes:
            if self.ind[axis]:
                quanta = multi_index()
                bonds = multi_index()
                for ind in self.ind[axis]:
                    if isinstance(ind, bond):
                        bonds.append(ind)
                    elif isinstance(ind, quantum):
                        quanta.append(ind)
                # sort the physical indices by magnitude then sign
                # Recall that quanta are 1-indexed but sites are 0-indexed
                try:
                    quanta = multi_index(list(quanta.take(reversed(list(zip(*sorted(
                        (abs(f.tag) - 1, i) for i, f in enumerate(quanta)
                    )))[1]))))
                    for i, quant in enumerate(quanta[:-1]):
                        if ((abs(quant.tag) == abs(quanta[i+1].tag))
                        and (quant.tag < quanta[i+1].tag)):
                            quanta[i]   = quanta[i+1]
                            quanta[i+1] = quant
                except IndexError as ex:
                    if (sum( 1 for _ in quanta ) == 0):
                        pass
                    else:
                        raise ex
                if bonds:
                    quanta.extend(bonds)
                self.permute(quanta, axis)

    def reset_pos (self, center):
        """Reset the positions of the indices in the site **IN PLACE**.
        
        center :: int :: the quantum whose tag makes it the orthogonality center
        To the left and at the center, the quanta with tag > 0 will be placed
        in the rows, while those with tag < 0 will be placed in the columns.
        To the right of the center, this is reversed. The bonds will remain in
        their position at the fastest-changing index.
        This moves the bonds to the fastest-index in each axis.
        """
        if (self.all_quanta_tags('>', center) ^ (center == -1)):
            # right of center
            self.groupby_quanta_tag(groupby=('e.tag > 0', 'e.tag < 0'))
        else:
            # left of center
            self.groupby_quanta_tag(groupby=('e.tag < 0', 'e.tag > 0'))
        self.sort_ind()

    def contract (self, other):
        """Contract all bonds between two sites **IN PLACE**.
        
        The site on whom the method is called get its indices in the fastest
        changing positions, while the one in the argument gets its indices in
        the slowest-changing position. This is to preserve the order of the
        computational basis, since in general the quanta are enumerated in sites
        from left to right which is also the order of fastest changing to
        slowest changing sites. Also, `sort_ind` is called to sort bonds.
        
        If necessary, call sort_ind or reset_pos to reset the indices.
        
        If the sites share no bonds, the Kronecker product is the result.
        """
        assert isinstance(other, self.__class__)
        bonds = multi_index(
            [ e for e in self.ind[1].get_type(bond) if other in e.tag ]
        )
        assert all( e in other.ind[0] for e in bonds ), 'bond tag inconsistent'
        raised = multi_index([ e for e in self.ind[1] if e not in bonds ])
        lowered = multi_index([ e for e in other.ind[0] if e not in bonds ])
        self.permute(bonds + raised, 1)
        other.permute(lowered + bonds, 0)
        self.mat = np.kron(np.eye(lowered.dim), self.mat) \
            @ np.kron(other.mat, np.eye(raised.dim))
        self.ind[0] = lowered + self.ind[0]
        self.ind[1] = other.ind[1] + raised
        self.sort_ind()
        # Update bond references to this contracted site
        for axis in (0, 1):
            for bnd in self.ind[axis].get_type(bond):
                try:
                    bnd.tag[bnd.tag.index(other)] = self
                except ValueError:
                    pass 
                
    def split (self, trim=None, canon=None, full_matrices=False, row=0, col=1):
        """Create a bond via SVD, leading to three new sites: u, s, vh.
        If canon is 'left' or 'right' then s is multiplied into vh or u, resp.
        
        Updates bond tags **IN PLACE**
        """
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
            [self.__class__(), self.__class__()],
        ) 
        right_bond = bond(
            range(vh[:trim, :].shape[row]),
            [self.__class__(), self.__class__()],
        )
        left_site = self.__class__(
            mat=u[:, :trim],
            ind=self.ind[:row+1] + multi_index((multi_index((left_bond, )), )),
        )
        right_site = self.__class__(
            mat=vh[:trim, :], 
            ind=multi_index((multi_index((right_bond, )), )) + self.ind[col:],
        )
        # Update bond references to this split site
        for axis, i, new_site in [(0, 0, left_site), (1, -1, right_site)]:
            for bnd in self.ind[axis].get_type(bond):
                bnd.tag[bnd.tag.index(self)] = new_site
        # In case caller does not specify canonization, return the weight matrix
        if not canon:
            center_site = self.__class__(
                mat=np.diag(s)[:trim, :trim],
                ind=multi_index((
                    multi_index((left_bond, )),
                    multi_index((right_bond, )),
                )),
            )
            left_bond.tag = [left_site, center_site]
            right_bond.tag = [center_site, right_site]
            return (left_site, center_site, right_site)
        else:
            left_bond.tag = [left_site, right_site]
            right_site.ind[0][0] = left_bond
            return (left_site, right_site)

    def split_quanta (self, center, N=1, trim=None):
        """Split the site into sites where each quantum is on its own.
        
        Arguments:
        center :: int :: the quantum whose tag makes it the orthogonality center
        N :: int :: how many quanta per site (default: 1)
        Alternatively, if N is a sequence whose sum matches the total number of
        quanta, the entries of the sequence choose N the group sizes as so:
        the first entry of N keeps the first N[0] entries together and so on.
        """
#         print('splitting', center)
        NQ = sum( 1 for e in self.get_type(quantum) if (e.tag > 0) )
        if isinstance(N, int):
            P_list = [ N for _ in range(NQ) ]
        elif hasattr(N, '__len__'):
            assert (sum( int(e) for e in N ) == NQ), \
                'sequence N does not add up to the total number of quanta'
            P_list = list(N)
        else:
            raise TypeError('N must be an integer, list or tuple')
        left_counter = iter(range(NQ))
        right_counter = iter(range(-1, -NQ-1, -1))
        output, pos = [self, ], 0
#         print('Plist', P_list, 'N', N, NQ)
#         print('self', repr(self.ind))
#         print(repr(self))
#         print(repr(self.ind))
#         print(list(self.get_type(quantum)))
        def inner_split ():
            """Instead of recursion, use a closure to solve the problem."""
            nonlocal P_list, output, pos, left_counter, right_counter, center, trim
            quanta_tags = [ e.tag for e in output[pos].get_type(quantum) ]
            Nquanta = sum( 1 for e in quanta_tags if (e > 0) )
            is_right_of_center = (center == -1) ^ output[pos].all_quanta_tags(
                '>', center, quanta_tags
            )
            if is_right_of_center:
#                 print('right')
                P = P_list[next(right_counter)]
            else:
#                 print('left')
                P = P_list[next(left_counter)]
#             print('  P', P, center)
            output[pos].reset_pos(center)
#             print('POST RESEt', is_right_of_center, repr(output[pos]))
            if (Nquanta <= P):
#                 print('nquanta <= P')
                return False
            elif is_right_of_center:
                max_tag = max( abs(e) for e in quanta_tags )
                try:
                    output[pos].reshape('flat', 'ff',
                        N=sum(
                            P for e in output[pos].ind[0].get_type(quantum)
                            if (abs(e.tag) == max_tag)
                        )
                    )
                except AssertionError:
                    print('yipee')
                    pass
                output[pos].reshape('tall', 'ff',
                    N=sum( 1 for _ in output[pos].ind[1].get_type(quantum) )
                )
                output[pos].reshape('flat', 'ff', 
                    N=sum(
                        P for e in output[pos].ind[0].get_type(quantum)
                        if (abs(e.tag) == max_tag)
                    )
                )
                left_site, right_site = output[pos].split(trim=trim, canon='right')
                right_site.reset_pos(center)
                output[pos] = left_site
                output.insert(pos + 1, right_site)
                return True
            else: # at or left of center
                min_tag = min( abs(e) for e in quanta_tags )
                try:
                    output[pos].reshape('flat', 'ff', 
                        N=sum(
                            1 for e in self.ind[0].get_type(quantum)
                            if (abs(e.tag) > (min_tag + P - 1))
                        )
                    )
                except AssertionError:
                    print('yikes')
                    pass
                output[pos].reshape('tall', 'ff',
                    N=sum( 1 for _ in output[pos].ind[1].get_type(quantum) )
                )
#                 print('step 1', repr(output[pos]))
                output[pos].reshape('flat', 'ff', 
                    N=sum(
                        1 for e in output[pos].ind[0].get_type(quantum)
                        if (abs(e.tag) > (min_tag + P - 1))
                    )
                )
#                 print('step 2', repr(output[pos]))
                if (center in range(min_tag, min_tag + P)):
                    canon = 'right'
                else:
                    canon = 'left'
                left_site, right_site = output[pos].split(trim=trim, canon=canon)
                left_site.reset_pos(center)
#                 print('step left', repr(left_site))
#                 print('step right', repr(right_site))
                output[pos] = left_site
                pos += 1
                output.insert(pos, right_site)
#                 print(output)
                return True
        
        try:
            while inner_split(): pass
        except StopIteration: pass
#         for e in output:
#             print(repr(e))
#             print(repr(e.ind))
        return output