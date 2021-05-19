from datetime import datetime

import numpy as np

from ph121c_lxvm import models, data


def main():
    """Generate data for 4.1.1."""
    rng = np.random.default_rng(seed=935)
    W = 3.0
    Navg = 5
    bc = 'c'
    for L in [8, 10, 12]:
        for i in range(Navg):
            job = dict(
                oper=models.tfim_z.H_dense,
                oper_params={
                    'L' : L,
                    'h' : rng.uniform(low=-W, high=W, size=L),
                    'hz': rng.uniform(low=-W, high=W, size=L),
                    'bc': bc,
                },
                solver=np.linalg.eigh,
                solver_params={},
            )
            print(f'Starting L={L} sample={i}/{Navg-1} at {datetime.now()}')
            data.jobs.obtain(batch=True, **job)

if __name__ == '__main__':
    main()
