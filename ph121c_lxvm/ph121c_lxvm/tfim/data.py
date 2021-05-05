"""Performs computational storage/retrieval for TFIM data.

Results are stored in an HDF5 archive for eventual reuse.
The files in the archive are identified by a unique job hash.
Attributes about the job such as simulation parameters are stored as metadata.
The main interface provided by this module is the `obtain` function.
"""

import time
from hashlib import md5

from ..data import hdf5


ARCHIVE = '/home/lxvm/Documents/repos/ph121c/data/main.hdf5'

EXIT_MODES = ['found', 'ran_job']

LAST_EXIT_MODE = None

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
        
def job_name (oper, oper_params, solver, solver_params):
    """Make a hdf5 path for a job using a unique hash, e.g. /<hash>."""
    return '/'.join(['',
        hashify(dict(
            oname=oper.__module__,
            omod=oper.__name__,
            sname=solver.__module__,
            smod=solver.__name__,
            **oper_params,
            **solver_params,
        )).hex(),
    ])

def job_metadata (oper, oper_params, solver, solver_params):
    """Make some metadata to save with a job."""
    metadata = dict(
        oper='.'.join([oper.__module__, oper.__name__]),
        solver='.'.join([solver.__module__, solver.__name__]),
        **oper_params,
        **solver_params,
    )
    return metadata

def job (oper, oper_params, solver, solver_params):
    """Compute job and return data + metadata."""
    tic = time.perf_counter()
    operator = oper(**oper_params)
    tac = time.perf_counter()
    data = solver(operator, **solver_params)
    toe = time.perf_counter()
    metadata = job_metadata(oper, oper_params, solver, solver_params)
    metadata['opertime']   = tac - tic
    metadata['solvertime'] = toe - tac
    metadata['walltime']   = toe - tic
    return (data, metadata)

    
def obtain (oper, oper_params, solver, solver_params, archive=ARCHIVE, batch=False):
    """Obtain a dataset from archive or by computation.
    
    The idea is that oper(**oper_params) is the matrix that is the first
    argument to solver, with additional arguments in solver_params.
    
    Uses a hash function to name individual data sets by the contents of jobs.
    Actual parameters for jobs go into the dataset attributes.
    """
    global LAST_EXIT_MODE
    dset = job_name(oper, oper_params, solver, solver_params)
    try:
        data = hdf5.find(dset, archive)
        LAST_EXIT_MODE = EXIT_MODES[0]
        if batch:
            return
        return data
    except Exception:
        data, metadata = job(oper, oper_params, solver, solver_params)
        hdf5.save(dset, data, metadata, archive)
        LAST_EXIT_MODE = EXIT_MODES[1]
        if batch:
            return
        return data