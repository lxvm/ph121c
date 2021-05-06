"""Test bit operations.
"""

import unittest

import numpy as np

from ..basis import bits


class bit_tester (unittest.TestCase):
    def test_bit_swap (self):
        """Test bit swap."""
        indices = np.arange(2 ** 4 + 1)
        swapped = bits.swap(indices, 0, 2)
        for i, j in zip(indices, swapped):
            print (np.binary_repr(i, width=8), np.binary_repr(j, width=8))
            
        
if __name__ == '__main__':
    unittest.main()