"""Test interfaces in the tfim subpackage using unittest.
"""

import os
import unittest
from tempfile import mkstemp
from itertools import combinations

import h5py
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

from .. import tfim

def param_sweep (
    L  = set(range(4, 11, 2)),
    h  = set([0.3, 1.0, 1.7]),
    bc = set(['o', 'c']),
):
    """Generate parameter values to sweep over"""
    for i in L:
        for j in h:
            for k in bc:
                yield {
                    "L" : i,
                    "h" : j,
                    "bc": k,
                }

def calc_evals (module, **kwargs):
    """Calculates 6 extremal eigenvalues using all implemented methods"""
    which_six = [0, 1, 2, -3, -2, -1]
    D = module.H_dense(**kwargs)
    S = module.H_sparse(**kwargs)
    O = module.L_vec(**kwargs)
    D_vals = la.eigvalsh(D)
    S_vals = sla.eigsh(S, which='BE',
                       return_eigenvectors=False)
    O_vals = sla.eigsh(O, which='BE',
                       return_eigenvectors=False)
    return (D_vals[which_six], S_vals, O_vals)

class tfim_z_test_case (unittest.TestCase):

    def test_sweep_sparse_against_dense (self):
        """Check eigenvalues match using dense and all sparse methods"""
        for params in param_sweep():
            with self.subTest(**params):
                all_evals = calc_evals(tfim.z, **params)
                for pair in combinations(all_evals, r=2):
                    self.assertTrue(np.allclose(*pair))    
    
class tfim_x_test_case (unittest.TestCase):

    def test_sweep_sparse_against_dense (self):
        """Check eigenvalues match using dense and all sparse methods"""
        for sector in ('+', '-'):
            for params in param_sweep():
                params['sector'] = sector
                with self.subTest(**params):
                    all_evals = calc_evals(tfim.x, **params)
                    for pair in combinations(all_evals, r=2):
                        self.assertTrue(np.allclose(*pair)) 
                        
class tfim_compare_x_z (unittest.TestCase):
                        
    def test_sweep_compare_x_z (self):
        """Check eigenvalues from z and x bases match"""
        return NotImplemented
            
class tfim_data_storage(unittest.TestCase):
    
    def setUp (self):
        """Set testing parameters"""
        self.func = tfim.z.H_sparse
        self.params_a = {
            'L'  : set([4, 5]),
            'h'  : set([0.3, 1.0, 1.7]),
            'bc' : set(['o', 'c']),
        }
        self.params_b = {
            'L'  : set([4, 6]),
            'h'  : set([0.3, 1.0, 1.7]),
            'bc' : set(['o', 'c']),
        }
        self.solver = sla.eigsh
        self.job = lambda params : { 'A' : self.func(**params) }
        self.archive = mkstemp()[1]
        
    def test_hashable_naming (self):
        """Check naming is unique and repeatable"""
        for p_a, p_b in zip(
            param_sweep(**self.params_a),
            param_sweep(**self.params_b),
        ):
            a = tfim.data.name(self.func, p_a, self.solver, self.job(p_a))
            b = tfim.data.name(self.func, p_b, self.solver, self.job(p_b))
            if p_a == p_b:
                self.assertTrue(a == b)
            else:
                self.assertTrue(a != b)
            
    def test_hdf5_interface (self):
        """Make sure writing to archive works."""
        for p in param_sweep(**self.params_a):
            # should calculate a new dataset
            data = tfim.data.job(
                self.func, p, self.solver, self.job(p), self.archive
            )
            # should fetch the calculated dataset
            fetch = tfim.data.job(
                self.func, p, self.solver, self.job(p), self.archive
            )
        
    def tearDown(self):
        """Delete temp file"""
        os.remove(self.archive)

if __name__ == '__main__':
    unittest.main()