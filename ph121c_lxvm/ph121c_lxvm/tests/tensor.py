"""Test procedures made available by ph121c_lxvm.tensor subpackage.
"""

import unittest
from itertools import product, combinations

import numpy as np

from .. import tensor

class utils_test_case (unittest.TestCase):
    """."""
    pass


class index_test_case (unittest.TestCase):
    """."""
    pass


class multi_index_test_case (unittest.TestCase):
    """."""
    pass
    
    
class bond_test_case (unittest.TestCase):
    """."""
    pass
    
    
class quantum_test_case (unittest.TestCase):
    """."""
    pass

    
class site_test_case (unittest.TestCase):
    """."""
    
    def setUp (self):
        """."""
        pass
    
    def test_split (self):
        """."""
        pass
    
    def test_contract (self):
        """."""
        pass
    
    def test_groupby_quanta_tag (self):
        """."""
        pass
        
    def test_split_quanta (self):
        """."""
        pass
    
        
    def test_trim_bonds (self):
        """."""
        pass
    
    def test_link_quanta (self):
        """."""
        pass
        
    def test_relink_bonds (self):
        """."""
        pass
    
    
class train_test_case (unittest.TestCase):
    """."""
        
    def setUp (self):
        """."""
        pass
    
    def test_split_site (self):
        """."""
        pass
    
    def test_contract_bond (self):
        """."""
        pass
    
    def test_max_quantum_tag (self):
        """."""
        pass
    
    def test_center_tag (self):
        """."""
        pass
    
    def test_split_quanta (self):
        """."""
        pass
    
    def test_canonize (self):
        """."""
        pass
    
    def test_trim_bonds (self):
        """Check that the compression works for several systems."""
        pass
#         for d, L in product([2, 3], [5, 6, 7, 8]):
#             v = np.random.random(d ** L)
#             v = v / np.linalg.norm(v)
#             def lossless (i):
#                 return tensor.dim_mps(i, L, d)
#             A = tensor.mps(v, lossless, L, d)
#             with self.subTest(name='Check lossless accuracy', d=d, L=L):
#                 self.assertTrue(np.allclose(v, A.contract_bonds()))
#             with self.subTest(name='Test monotonic quality of approximation', d=d, L=L):
#                 ranks = np.arange(1 + d ** (L // 2), step=d)[::-1]
#                 norms = np.zeros(ranks.size)
#                 for i, chi in enumerate(ranks):
#                     def rank (i, x=chi):
#                         return max(1, min(x, lossless(i)))
#                     A.lower_rank(rank)
#                     norms[i] = np.inner(v, A.contract_bonds())
#                 self.assertTrue(
#                     np.all(norms.argsort() == np.arange(norms.size)[::-1])
#                 )

    def test_merge_bonds (self):
        """."""
        pass
    
    def test_groupby_quanta (self):
        """."""
        pass

    
class mps_test_case (unittest.TestCase):
    """Test the tensor.mps module."""
    
    def setUp (self):
        """."""
        pass
    
    def test_from_vec (self):
        """."""
        pass
    
    def test_from_arr (self):
        """."""
        pass

    def test_norm (self):
        """."""
        pass
    
    def test_normalize (self):
        """."""
        pass
    
    def test_inner (self):
        """Compares the inner product calculated two ways.
        
        The first way is by contracting all of the bond indices.
        The second way is by performing an mps inner product.
        """
        pass
#         for d, L in product([2, 3], [5, 6, 7, 8]):
#             v = np.random.random(d ** L)
#             v = v / np.linalg.norm(v)
#             def lossless (i):
#                 return tensor.dim_mps(i, L, d)
#             A = tensor.mps(v, lossless, L, d)
#             self.assertTrue(
#                 np.allclose(np.inner(v, v), A.inner(A))
#             )
    def test_get_component (self):
        """."""
        pass
    
    
class mpo_test_case (unittest.TestCase):
    """Test the tensor.mpo module."""
    
    def setUp (self):
        """."""
        pass
        
    def test_set_local_oper (self):
        """."""
        pass
    
    def test_to_arr (self):
        """."""
        pass
    
    def test_groupby_quanta_tag (self):
        """."""
        pass
    
    def test_oper (self):
        """."""
        pass
    
    def test_mel (self):
        """."""
        pass
    
    def test_expval (self):
        """."""
        pass
    

if __name__ == '__main__':
    unittest.main()