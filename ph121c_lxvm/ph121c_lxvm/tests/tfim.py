"""Test interfaces in the tfim subpackage using unittest.
"""

import os
import unittest
from tempfile import mkstemp
from itertools import combinations

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

from .. import tfim, data

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
        self.oper = tfim.z.H_sparse
        self.oper_params = [
            {
                'L'  : set([4, 5]),
                'h'  : set([0.3, 1.0, 1.7]),
                'bc' : set(['o', 'c']),
            },
            {
                'L'  : set([4, 6]),
                'h'  : set([0.3, 1.0, 1.7]),
                'bc' : set(['o', 'c']),
            },
        ]
        self.solver = sla.eigsh
        self.solver_params = [
            {
                'k' : 6,
                'which' : 'BE',
            }
        ]
        self.archive = mkstemp()[1]
        
    def test_hashable_naming (self):
        """Check naming is unique and repeatable"""
        for solver_params in self.solver_params:
            for oper_params_tuple in zip(
                *[ param_sweep(**p) for p in self.oper_params ]
            ):
                names = [
                    tfim.data.job_name(
                        self.oper,   oper_params,
                        self.solver, solver_params,
                    ) 
                    for oper_params in oper_params_tuple
                ]
                for pair in combinations(
                    zip(names, oper_params_tuple), r=2
                ):
                    if pair[0][1] == pair[1][1]:
                        self.assertTrue(pair[0][0] == pair[1][0])
                    else:
                        self.assertTrue(pair[0][0] != pair[1][0])

    def test_hdf5_interface (self):
        """Make sure reading and writing to archive works."""
        for solver_params in self.solver_params:
            for oper_params in param_sweep(**self.oper_params[0]):
                job = [self.oper, oper_params, self.solver, solver_params]
                # should calculate a new dataset
                _ = tfim.data.obtain(
                    *job, archive=self.archive, batch=True,
                )
                self.assertTrue(
                    tfim.data.LAST_EXIT_MODE == tfim.data.EXIT_MODES[1]
                )
                # should fetch the calculated dataset
                fetch = tfim.data.obtain(
                    *job, archive=self.archive, batch=False,
                )
                self.assertTrue(
                    tfim.data.LAST_EXIT_MODE == tfim.data.EXIT_MODES[0]
                )
                # should store/fetch metadata correctly
                metadata = data.hdf5.inquire(
                    self.archive, path=tfim.data.job_name(*job),
                )
                for k, v in tfim.data.job_metadata(*job).items():
                    self.assertTrue(v == metadata[k])

    def tearDown(self):
        """Delete temp file"""
        os.remove(self.archive)

if __name__ == '__main__':
    unittest.main()