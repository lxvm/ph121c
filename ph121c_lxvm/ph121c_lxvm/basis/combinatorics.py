"""Perform combinatorial tasks.
"""

import numpy as np

from . import bits

def cycles (perm):
    """Return a list of cycles representing the permutation."""
    index = list(range(len(perm)))
    assert set(index) == set(perm), 'permutation must be zero-indexed'
    cycles = []
    for i in index:
        cycle = [i]
        while perm[i] not in cycle:
            i = perm[i]
            cycle.append(index.pop(index.index(i)))
        cycles.append(cycle)
    return cycles
