#!/usr/bin/env python

"""Compute all wavefunctions of interest in one go.

This is useful to populate the HDF5 archive.
"""

import scipy.sparse.linalg as sla

from ph121c_lxvm import tfim, tests

L = range(6, 21, 2)
h = [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 1.7]
bc = ['o', 'c']

def main ():
    """Sweep a wide range of parameters to generate datasets"""
    for oper_params in tests.tfim.param_sweep(
        L = L, h = h, bc= bc,
    ):
        tfim.data.obtain(
            oper=tfim.z.H_sparse,
            oper_params=oper_params,
            solver=sla.eigsh,
            solver_params={
                'k' : 6,
                'which' : 'BE',
            },
            batch=True,
        ) 

if __name__ == '__main__':
    main()