"""Test interfaces in the data subpackage using unittest.

Modules and procedures tested:
jobs: obtain, and by extension the hdf5 module
"""

import os
import unittest
from tempfile import mkstemp
from itertools import combinations

from .. import data, tfim

    
class jobs_test_case (unittest.TestCase):
    """Test the data storage and retrieval backend."""
    
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
                    data.job_name(
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
                with self.subTest(name='calculates new dataset on first time'):
                    _ = data.jobs.obtain(
                        *job, archive=self.archive, batch=True,
                    )
                    self.assertTrue(
                        data.jobs.LAST_EXIT_MODE == data.jobs.EXIT_MODES[1]
                    )
                with self.subTest(name='fetches previously calculated data'):
                    fetch = data.jobs.obtain(
                        *job, archive=self.archive, batch=False,
                    )
                    self.assertTrue(
                        data.jobs.LAST_EXIT_MODE == data.jobs.EXIT_MODES[0]
                    )
                with self.subTest(name='correctly stores/retrieves metadata'):
                    metadata = data.hdf5.inquire(
                        self.archive, path=data.jobs.job_name(*job),
                    )
                    for k, v in data.jobs.job_metadata(*job).items():
                        self.assertTrue(v == metadata[k])

    def tearDown(self):
        """Delete temp file"""
        os.remove(self.archive)

        
if __name__ == '__main__':
    unittest.main()