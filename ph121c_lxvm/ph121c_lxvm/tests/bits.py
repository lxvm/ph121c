"""Test bit operations.
"""

import unittest
from itertools import combinations

import numpy as np

from ..basis import bits


class bit_tester (unittest.TestCase):
    def test_bit_swap (self):
        """Test bit swap."""
        n = 4
        indices = np.arange(2 ** n)
        
        # swap test
        for i, j in combinations(np.arange(n), r=2):
            swapped = bits.swap(indices, i, j)
            self.assertTrue(
                (bin())
                ^ np.all(((indices ^ swapped) - (2 ** i + 2 ** j)) == 0)
            )
        # null test
        for i in range(n):
            self.assertTrue(
                np.all(indices == bits.swap(indices, i, i))
            )
            
        
if __name__ == '__main__':
    unittest.main()