from datetime import datetime

import numpy as np

from ph121c_lxvm import models, data


def main():
    """Generate data for 4.1.1."""
    hx, hz = (-1.05, 0.5)
    bc = 'c'
    for L in [8, 10, 12, 14]:
        job = dict(
            oper=models.tfim_z.H_dense,
            oper_params={
                'L' : L,
                'h' : hx,
                'hz': hz,
                'bc': bc,
            },
            solver=np.linalg.eigh,
            solver_params={},
        )
        print(f'Starting L={L} at {datetime.now()}')
        data.jobs.obtain(batch=True, **job)

if __name__ == '__main__':
    main()
