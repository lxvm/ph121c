"""Performs computational storage/retrieval for TFIM data.

Results are stored in an HDF5 archive for eventual reuse.
The files in the archive are grouped by the modules and functions
that call them and identified by a unique job hash for lookup.
Attributes about the job such as simulation parameters are also stored.
The main interface provided by this module is the `job` function.
"""

import time
from hashlib import md5

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

from .. import tfim
from ..data import hdf5


ARCHIVE = '/home/lxvm/Documents/repos/ph121c/data/main.hdf5'

def tuplify (obj):
    """Make an iterable into a tuple with hashed parts."""
    if isinstance(obj, dict):
        return tuple(sorted((k, hashify(v)) for k, v in obj.items()))
    elif isinstance(obj, (set, frozenset)):
        return tuple(sorted(hashify(e) for e in obj))
    elif isinstance(obj, (tuple, list)):
        return tuple(hashify(e) for e in obj)
    else:
        return obj

def hashify (obj):
    """Extract a reproducible ascii hash from an object."""
    # Using this idea https://stackoverflow.com/a/42151923 for nonhashables
    try:
        return md5(obj).digest()
    except TypeError:
        return hashify(repr(tuplify(obj)).encode())
        
def name (func, params, solver, job):
    """Make a name for a dataset.
    
    /<func_module>/<func_name>/<solver_name>/<job+params_hash>
    """
    return '/'.join(['',
                     func.__module__,
                     func.__name__,
                     solver.__name__,
                     hashify(dict(**job, **params)).hex(),
                    ])
    
def job (func, params, solver, job, archive=ARCHIVE):
    """Retrieve the data set from archive or compute if not available.
    
    Uses a hash function to name individual data sets by the contents of jobs.
    Actual parameters for jobs go into the dataset attributes.
    """
    dset = name(func, params, solver, job)
    try:
        return hdf5.find(dset, archive)
    except Exception:
        tic = time.perf_counter()
        data = solver(**job)
        toc = time.perf_counter() - tic
        metadata = params.copy()
        metadata['runtime'] = toc
        hdf5.save(dset, data, metadata, archive)
        return data
        