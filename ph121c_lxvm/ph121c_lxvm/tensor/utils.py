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

def exchange (dima, dimb):
    """Create a permutation that exchanges the order of multi-index a, b.
    
    Here index b is assumed to be the faster changing one, and the permutation
    changes that so index a will become the fastest changing one.
    """
    # null op
    # np.repeat(dimb*np.arange(dima), repeats=dimb) + np.tile(np.arange(dimb), reps=dima)
    return np.tile(dimb*np.arange(dima), reps=dimb) + np.repeat(np.arange(dimb), repeats=dima)

def get_phys_dim (dim, d):
    """Get the exponent k of d ** k."""
    assert d != 1, 'not a physical meaningful qudit, d = 1'
    count = 0
    eat = dim
    while eat >= d:
        eat = eat // d
        count += 1
    assert eat == 1, 'dimension was not a power of d'
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