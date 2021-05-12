"""Perform combinatorial tasks.
"""

import numpy as np


def cycles (perm):
    """Return a list of cycles representing the permutation."""
    index = list(range(perm.size))
    assert np.all(index == np.sort(perm)), 'permutation must be zero-indexed'
    cycles = []
    
    for i in index:
        cycle = [i]
        while perm[i] not in cycle:
            i = perm[i]
            cycle.append(index.pop(index.index(i)))
        cycles.append(np.array(cycle, dtype='int64'))
    return np.array(cycles, dtype='object')
