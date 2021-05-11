"""Test matricization, matrix product states, and tensor networks.
"""

import unittest
from itertools import combinations

import numpy as np

from .. import basis, measure


class mps_schmidt_decomposition (unittest.TestCase):
    
    def test_permute (self):
        """Check that the permutation and its inverse is the identity."""
        for L in [3, 4, 5, 6, 7]:
            v = np.random.random(2**L)
            for r in range(1, L):
                for c in combinations(range(L), r=r):
                    with self.subTest(L=L, r=r, c=c):
                        w = basis.schmidt.permute(L, c, v)
                        w = basis.schmidt.permute(L, c, w, reverse=True)
                        self.assertTrue(all(v == w))
    
    def test_matricization (self):
        """Check that matricization and vectorization is the identity.
        
        Construction should follow left-canonical MPS.
        See: https://arxiv.org/pdf/1008.3477.pdf.
        """
        for L in [3, 4, 5, 6, 7]:
            v = np.random.random(2**L)
            for r in range(1, L):
                for c in combinations(range(L), r=r):
                    with self.subTest(L=L, r=r, c=c):
                        M = basis.schmidt.matricize(L, c, v)
                        self.assertTrue(
                            M.shape == (2 ** len(c), 2 ** (L - len(c)))
                        )
                        w = basis.schmidt.vectorize(L, c, M)
                        self.assertTrue(
                            w.shape in [(2**L, ), (2**L, 1)]
                        )
                        self.assertTrue(
                            np.allclose(v, w)
                        )
    
    def test_svd_reconstruction (self):
        """Check that svd reconstruction is correct."""
        return NotImplemented
    
    def test_schmidt_truncation (self):
        """Test truncation of svd is correct"""
        return NotImplemented
                        
if __name__ == '__main__':
    unittest.main()