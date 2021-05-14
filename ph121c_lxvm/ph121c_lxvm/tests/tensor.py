"""Test procedures made available by ph121c_lxvm.tensor subpackage.

Classes and methods tested:
mps: __init__, lower_rank, inner

Untested (but otherwise verified) procedures:
mps: oper
mpo
"""

import unittest
from itertools import product, combinations

import numpy as np

from .. import tensor

class mps_test_case (unittest.TestCase):
    """Test the tensor.mps module."""
    def test_accuracy (self):
        """Check that the compression works for several systems."""
        for d, L in product([2, 3], [5, 6, 7, 8]):
            v = np.random.random(d ** L)
            v = v / np.linalg.norm(v)
            def lossless (i):
                return tensor.dim_mps(i, L, d)
            A = tensor.mps(v, lossless, L, d)
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
                
    def test_inner_consistent (self):
        """Compares the inner product calculated two ways.
        
        The first way is by contracting all of the bond indices.
        The second way is by performing an mps inner product.
        """
        for d, L in product([2, 3], [5, 6, 7, 8]):
            v = np.random.random(d ** L)
            v = v / np.linalg.norm(v)
            def lossless (i):
                return tensor.dim_mps(i, L, d)
            A = tensor.mps(v, lossless, L, d)
            self.assertTrue(
                np.allclose(np.inner(v, v), A.inner(A))
            )
    
    def test_local_oper_consistent (self):
        """Test expectation value of a local operator two ways.
        
        The first way is by calculating the operator as a matrix-vector product.
        The second way is by performing an mps inner product.
        """
        pass
    
class mpo_test_case (unittest.TestCase):
    """Test the tensor.mpo module."""
    pass


if __name__ == '__main__':
    unittest.main()