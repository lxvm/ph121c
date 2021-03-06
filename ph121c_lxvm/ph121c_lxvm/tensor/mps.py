"""Define the `mps` class to create and operate on MPS wavefunctions.

These MPS methods are intended for systems with open boundary conditions.
In principle, extending these methods to closed/periodic boundary conditions
should amount to adding an extra 'bond' between the ends that 

A thorough review in section 4 of: https://arxiv.org/abs/1008.3477.
"""

from copy import deepcopy

from .utils   import *
from .indices import *
from .site    import *
from .train   import *


class mps (train):
    """Class for representing and operating on MPS wavefunctions."""
    def __init__ (self, iterable=(), L=None, d=None, center=None):
        """Instantiate an mps object for a qudit chain of length L.
        
        Arguments:
        iterable :: an iterable of sites :: 
        L :: int :: the number of physical indices
        d :: int :: the qudit dimension
        """
        super().__init__(iterable)
        self.d = d
        self.L = L
        self.center = center

    def from_vec (self, vec):
        """Populate an MPS from a vector.
        
        Arguments:
        vec :: np.ndarray(d**L) :: in the computational basis & row-major format
        L :: int :: the number of physical indices
        d :: int :: the qudit dimension
        """
        assert vec.size == self.d ** self.L
        assert np.allclose(1, np.inner(vec, vec)), 'input vector not normalized'
        super().__init__([
            site(
                mat=vec.reshape((1, vec.size)),
                ind=multi_index((
                    multi_index(),
                    multi_index([
                        quantum(range(self.d), i) for i in range(self.L, 0, -1)
                    ]),
                ))
            ),
        ])
        self.center = self[0]

    def from_arr (self, iterable, center):
        """Populate a mps instance from an iterable of matrices.
        
        This function should be an inverse to the `view` method if the MPS
        contains no weight matrices.
        
        Arguments:
        iterable :: iterable of np.ndarrays :: containing the coefficients
        center :: np.ndarray also in iterable or int :: orthogonality center,
        where if a np.ndarray, assigns that array as the center (if in iterable)
        or if an int, gives the site with that physical index the center.
        
        Description:
        The matrices are assumed to have the following structure:
        matrices left-of and including center have their physical indices in the
        rows, whereas those right of the center must have them in the columns.
        Then the column indices at the first site define the dimension of the
        first bond, carried over to the calculation for the next site and so on.
        If the given center does note correspond to the structure of the arrays
        in the iterable, you will probably get an error.
        """
        items = list(
            e for e in iterable
            if (isinstance(e, np.ndarray) and (e.ndim == 2))
        )
        assert (len(items) > 1), 'Use method `from_vec` instead.'
        assert isinstance(center, (int, np.ndarray))
        sites = [
            site(
            # since each object should be unique, cannot rely on site() default
                mat=np.ones((1, 1)),
                ind=multi_index((multi_index(), multi_index()))
            ) for _ in items
        ]
        quanta_tags = iter(range(1, self.L+1))
        prev_bond_dim = 1
        next_bond_dim = None
        left_of_center = True
        center_pos = -1
        try:
            # Populate the indices using the inference rules
            for i, arr in enumerate(items):
                sites[i].mat = arr
                if left_of_center:
                    # insert physical indices
                    sites[i].ind[0].extend(reversed([
                        quantum(range(self.d), tag=next(quanta_tags))
                        for _ in range(get_phys_dim(
                            arr.shape[0] // prev_bond_dim, self.d
                        ))
                    ]))
                    if (i > 0):
                        # fetch the preceding bond
                        # i == 0 is special: no bond to left
                        sites[i].ind[0].extend(
                            sites[i-1].ind[1].get_type(bond)
                        )
                    if ((i + 1) < len(items)):
                        # insert the present bond
                        # last case is special: no bond to right
                        sites[i].ind[1].append(
                            bond(range(arr.shape[1]), tag=sites[i:i+2])
                        )
                    # Identify the orthogonality center as the turning point
                    if (
                        isinstance(center, np.ndarray) 
                        and (id(arr)==id(center))
                    ) or (
                        isinstance(center, int) 
                        and any(
                            center == e.tag
                            for e in sites[i-1].ind[0].get_type(quantum)
                        )
                    ):
                        left_of_center = False
                        center_pos = i
                    prev_bond_dim = arr.shape[1]
                else:
                    sites[i].ind[0].extend(
                        sites[i-1].ind[1].get_type(bond)
                    )
                    if ((i + 1) < len(items)):
                        next_bond_dim = items[i+1].shape[0]
                    else:
                        next_bond_dim = 1
                    sites[i].ind[1].extend(reversed([
                        quantum(range(self.d), tag=next(quanta_tags))
                        for _ in range(get_phys_dim(
                            arr.shape[1] // next_bond_dim, self.d
                        ))
                    ]))
                    if ((i + 1) < len(items)):
                        # last case is special: no bond to right
                        sites[i].ind[1].append(
                            bond(range(next_bond_dim), tag=sites[i:i+2])
                        )
        except StopIteration:
            raise ValueError(f'L={self.L} too small for observed physical indices.')
        except IndexError as ex:
            raise IndexError(f'{str(ex)}: Check position of ortho-center.')
        assert all( e.test_shape() for e in sites ), \
            'matrix and multi_index shape differ: Check position of ortho-center.'
        super().__init__(sites)
        self.center = self[center_pos]
        
    ## These methods must not change the state of the MPS !
        
    def inner (self, other):
        """Take the inner product with another mps wavefunction <self|other>."""
        assert isinstance(other, mps)
        assert self.L == other.L
        assert self.d == other.d
        output = deepcopy(self)
        prev_self_site = None
        prev_output_site = None
        copy_self_site = None
        copy_output_site = None
        
        for i in range(len(self)):
            tags = [ e.tag for e in self[i].get_type(quantum) if (e.tag > 0) ]
            other.groupby_quanta_tag(tags)
        output.canonize(-1)
        assert (output.center == output[-1])
        assert (len(self) == len(output)), 'grouping did not succeed'
        # Tensor contraction
        wave = other[0].copy()
        for i in range(len(self) - 1):
            output[0].conj()
            output[0].transpose()
            # Contract the physical index
            output[0].link_quanta(wave)
            output[0].contract(wave)
            # Contract the virtual index
            wave = other[i+1].copy(other[i], output[0])
            wave.relink_bonds(output[0], 0)
            output[0].contract(wave)
            wave = output.pop(0)
        output[0].conj()
        output[0].transpose()
        output[0].link_quanta(wave)
        output[0].contract(wave)
        return output[0]

    def norm (self):
        """Return the norm of the MPS wavefunction."""
        try:
            center_tag = next(iter(self.center.get_type(quantum))).tag
            self.center.reset_pos(center_tag)
            return np.sqrt(np.trace(
                self.center.mat @ self.center.mat.T.conj()
            ))
        except Exception:
            self.canonize(-1)
            return self.norm()
    
    def normalize (self):
        """Normalize the MPS **IN PLACE**."""
        self.center.mat = self.center.mat / self.norm()