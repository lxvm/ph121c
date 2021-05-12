"""Test matricization, matrix product states, and tensor networks.
"""

import unittest
from itertools import product, combinations

import numpy as np

from .. import basis, measure


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
        """Check that matricization and vectorization is the identity.
        
        Construction should follow left-canonical MPS.
        See: https://arxiv.org/pdf/1008.3477.pdf.
        """
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
    
    def test_svd_reconstruction (self):
        """Check that svd reconstruction is correct."""
        return NotImplemented
    
    def test_schmidt_truncation (self):
        """Test truncation of svd is correct"""
        return NotImplemented

class mps_test_case (unittest.TestCase):
    """Test the basis.mps module."""
    def test_accuracy (self):
        """Check that the compression works for several systems."""
        for d, L in product([2, 3], [5, 6, 7, 8]):
            v = np.random.random(d ** L)
            v = v / np.linalg.norm(v)
            def lossless (i):
                return basis.mps.dim_mps(i, L, d)
            A = basis.mps.my_mps(v, lossless, L, d)
            with self.subTest(name='Check lossless accuracy', d=d, L=L):
                self.assertTrue(np.allclose(v, A.contract_bonds()))
            with self.subTest(name='Test monotonic quality of approximation', d=d, L=L):
                ranks = np.arange(1 + d ** (L // 2), step=d)[::-1]
                norms = np.zeros(ranks.size)
                for i, chi in enumerate(ranks):
                    def rank (i, x=chi):
                        return max(1, min(x, lossless(i)))
                    A.lower_rank(rank)
                    norms[i] = np.inner(v, A.contract_bonds())
                self.assertTrue(
                    np.all(norms.argsort() == np.arange(norms.size)[::-1])
                )
            
    
if __name__ == '__main__':
    unittest.main()