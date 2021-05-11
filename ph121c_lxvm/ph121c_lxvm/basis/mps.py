"""Implement matrix-product state factorization and operations.

Conventional to use left-canonical MPS.
For reference: https://arxiv.org/pdf/1008.3477.pdf.
"""

import numpy as np
import scipy.sparse.linalg as sla

from .. import tests, tfim


def tr_merge (s, w, r, d):
    """Truncates virtual index and merges the next available physical index."""
    index = np.repeat(np.arange(r), d) \
        + np.repeat(np.arange(0, d*r, r).reshape(d*r, 1), r, axis=0).ravel()
    return (s[:r, None] * w[:r, :]).reshape((r*d, w.shape[1] // d), order='F')[index, :]
    
def convert_vec_to_mps (v, L, r=2, d=2):
    """Return a MPS representation of a wavefunction."""
    assert 1 <= r <= d
    assert v.size == d ** L
    nrow = d
    ncol = d ** (L - 1)
    A    = np.zeros(d ** L)
    a, s, w = np.linalg.svd(
        v.reshape((nrow, ncol), order='F'),
        full_matrices=False
    )
    A[:d*r] = a[:, :r].ravel()
    
    while ncol > d:
        w    = tr_merge(s, w, r, d)
        nrow = w.shape[0]
        ncol = w.shape[1]
        a, s, w = np.linalg.svd(w, full_matrices=False)
        A[:d*r] = a[:, :r].ravel()
    return my_mps(v, L, r, d, A)

class my_mps:
    """Class for and operating on MPS representations of wavefunctions.
    
    Ok with qudits already in the computational basis.
    """
    def __init__ (self, v, L, r, d, A):
        # desired feature: argument: order=[perm(range(L))]
        # to construct the mps from a different order of the virtual indices
        assert 1 <= r <= d
        assert v.size == d ** L
        self.v = v
        self.d = d
        self.r = r
        self.L = L
        self.A = A

    def lower_rank (self, r):
        """Return a new my_mps with lower rank."""
        assert 1 <= r < self.r
        return my_mps(
            self.v,
            self.L,
            r,
            self.d,
            self.A.reshape((self.A.size // self.d, self.d))[:, :r].ravel(),
        )
    
    def get_component (self):
        pass
    
    
if __name__ == '__main__':
#     for oper_params in tests.tfim.param_sweep(
#         L = [20],
#         h = [1],
#         bc= ['o'],
#     ):
    L = 20
    h = 1
    bc = 'o'
    oper_params={
        'L' : L,
        'h' : h,
        'bc': bc,
    }
    job = dict(
        oper=tfim.z.H_sparse,
        oper_params=oper_params,
        solver=sla.eigsh,
        solver_params={ 
            'k' : 6, 
            'which' : 'BE',
        },
    )
    evals, evecs = tfim.data.obtain(**job)
    
    gs = evecs[:, 0]
    
    def mps_index (L, X, r, bondN, indexA, indexB=None):
        """Return a slice to retrieve one Schmidt vector based on its index."""
        assert L >= bondN
        assert X >= indexA
        #if 
        start = 0
        end = 0

    X = 2
    r = 2