import re

import numpy as np


def dim_mps (i, L, d):
    """Calculate the maximal rank for an svd at bond index i in mps"""
    if i <= L / 2:
        return d ** i
    else:
        return d ** (L - i)

def bond_rank (chi, L, d):
    """Return a function to calculate the bond ranks with a constant truncation."""
    return lambda i: max(1, min(chi, dim_mps(i, L, d)))

def test_valid_scheme (r, L, d):
    """Tests whether a rank function is a valid mps approximation scheme."""
    prev_val = 1
    for i in range(L + 1):
        if not (1 <= r(i) <= min(prev_val, dim_mps(i, L, d))):
            return False
        prev_val = d * r(i)
    return True

def my_arange (dims):
    """Obfuscation at its best. This is a null op, constucting arange from a multiindex."""
    index = np.zeros(np.prod(dims), dtype='int64')
    for i, e in enumerate(dims):
        prev_dim = np.prod(dims[:i], dtype='int64')
        next_dim = np.prod(dims[i+1:], dtype='int64')
        index += np.tile(
            np.repeat(
                next_dim *np.arange(e),
                repeats=next_dim
            ),
            reps= prev_dim
        )
    return index

def exchange (dima, dimb):
    """Create a permutation that exchanges the order of multi-index a, b.
    
    Here index b is assumed to be the faster changing one, and the permutation
    changes that so index a will become the fastest changing one.
    """
    # null op
    # np.repeat(dimb*np.arange(dima), repeats=dimb) + np.tile(np.arange(dimb), reps=dima)
    return np.tile(dimb*np.arange(dima), reps=dimb) + np.repeat(np.arange(dimb), repeats=dima)

def rexchange(dims):
    """Returns perm to reverse a np.ndarray of indices reversed(a1, ..., an).
    
    UNTESTED!!!
    """
    dim_a = dims[0]
    if dims.size > 2:
        dim_b = np.prod(dims[1:])
        return (
            np.tile(dim_b*np.arange(dim_a), reps=dim_b) 
            + np.repeat(rexchange(dims[1:]), repeats=dim_a)
        )
    else:
        dim_b = dims[1]
        return (
            np.tile(dim_b*np.arange(dim_a), reps=dim_b) 
            + np.repeat(np.arange(dim_b), repeats=dim_a)
        )

def multi_index_perm (dims, perm):
    """Return a slice to permute a multi-index in the computational basis.
    
    `dims` and `perm` enumerate 0 to N-1 as the fastest-changing dit to slowest.
    E.g. perm = np.arange(N) is the identity permutation.

    Arguments:
    dims :: np.ndarray(N) :: dimensions of each index
    perm :: np.ndarray(N) :: perm[i] gives the new position of i
    """
    assert dims.size == perm.size
    iperm = np.argsort(perm)
    new_dims = dims[iperm]
    index = np.zeros(np.prod(dims), dtype='int64')
    for i in range(len(dims)):
        index += np.tile(np.repeat(
            np.prod(dims[:iperm[i]], dtype='int64') * np.arange(dims[iperm[i]]),
            repeats=np.prod(new_dims[:i], dtype='int64')
        ), reps=np.prod(new_dims[i+1:], dtype='int64'))
    return index
    
def get_phys_dim (dim, d):
    """Get the exponent k of d ** k."""
    assert d != 1, 'not a physical meaningful qudit, d = 1'
    count = 0
    eat = dim
    while (eat >= d):
        assert ((eat % d) == 0), 'dimension was not a power of d'
        eat = eat // d
        count += 1
    return count

def indent (string, level=0):
    """Indents all newlines in a string by a level of spaces."""
    return re.sub('\n', ''.join(['\n', level * ' ']), string)

def rstr (obj):
    """Recursively get str(obj)."""
    if hasattr(obj, '__iter__'):
        return '(' + '\n'.join( rstr(e) for e in obj ) + ')'
    else:
        return str(obj)
    
def is_consecutive (sequence, container=None):
    """Return True if sequence is nonrepeating and consecutive (in container)."""
    if container:
        positions = [ container.index(e) for e in sequence ]
    else:
        positions = sequence
    return sorted(positions) == list(range(min(positions), max(positions) + 1))

def touches_edge (sequence, container):
    """Returns True if any sequence element is at container boundary."""
    return any(
        (container.index(e) == 0) or (container.index(e) == (len(container)-1))
        for e in sequence
    )

def direct_sum (A, B):
    """Calculate the direct sum of two matrices."""
    assert (A.ndim == B.ndim == 2)
    result = np.zeros(np.add(A.shape, B.shape))
    result[:A.shape[0], :A.shape[1]] = A
    result[A.shape[0]:, A.shape[1]:] = B
    return result

def chunk_seq (iterable):
    """Break a sorted iterable of integers into contiguous chunks."""
    result = []
    buffer = []
    last = None
    for e in iterable:
        if isinstance(e, int):
            if (last != None):
                if ((last + 1) == e):
                    buffer.append(e)
                else:
                    assert (e > last), 'Oops, list not sorted.'
                    result.append(buffer)
                    buffer = [e]
            else:
                buffer.append(e)
            last = e
    result.append(buffer)
    return result