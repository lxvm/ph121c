"""Test procedures made available by ph121c_lxvm.basis subpackage.

Modules and procedures/classes/methods tested:
schmidt: permute, matricize, vectorize
bits:    swap

Untested (but otherwise verified) procedures:
bits: poppar, vpoppar, Ising_parity_lookup, btest
combinatorics: cycles
unitary: Ising, Hadamard
"""

import unittest
from itertools import combinations

import numpy as np

from .. import basis


class bits_test_case (unittest.TestCase):
    def test_bit_swap (self):
        """Test bit swap."""
        n = 10
        indices = np.arange(2 ** n)
        # swap test
        for i, j in combinations(np.arange(n), r=2):
            swapped = basis.bits.swap(indices, i, j)
            self.assertTrue(np.all(
                (swapped == indices)
                ^ (((indices ^ swapped) - (2 ** i + 2 ** j)) == 0)
            ))
        # null test
        for i in range(n):
            self.assertTrue(
                np.all(indices == basis.bits.swap(indices, i, i))
            )
            
class schmidt_test_case (unittest.TestCase):
    """Test the basis.schmidt module."""
    
    def test_permute (self):
        """Check that the permutation and its inverse is the identity."""
        for L in [3, 4, 5, 6, 7]:
            v = np.random.random(2**L)
            for r in range(1, L):
                for c in combinations(range(L), r=r):
                    with self.subTest(L=L, r=r, c=c):
                        w = basis.schmidt.permute(v, c, L)
                        w = basis.schmidt.permute(w, c, L, inverse=True)
                        self.assertTrue(all(v == w))
    
    def test_matricization (self):
        """Check that matricization and vectorization is the identity."""
        for L in [3, 4, 5, 6, 7]:
            v = np.random.random(2**L)
            for r in range(1, L):
                for c in combinations(range(L), r=r):
                    with self.subTest(L=L, r=r, c=c):
                        M = basis.schmidt.matricize(v, c, L)
                        self.assertTrue(
                            M.shape == (2 ** len(c), 2 ** (L - len(c)))
                        )
                        w = basis.schmidt.vectorize(M, c, L)
                        self.assertTrue(
                            w.shape in [(2**L, ), (2**L, 1)]
                        )
                        self.assertTrue(
                            np.allclose(v, w)
                        )
                        
    
if __name__ == '__main__':
    unittest.main()