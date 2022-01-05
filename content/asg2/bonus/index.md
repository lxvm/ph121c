---
title: "Assignment 2"
date: 2021-05-13T20:35:17-07:00
draft: false
---
# Bonus

$\require{physics}$

## Verifying entanglement

I wanted to check something about entanglement entropy in this assignment:
are the Ising symmetry sectors of the Hamiltonian, which are uncoupled, not
entangled?
Since my code can handle building the Hamiltonian in the full x-basis as well
as permutations to and from the Ising symmetry sectors, let's test it!

We will find the ground state in the x basis and then sift out the system
into a + symmetry sector and - symmetry sector, do an svd and look at the values.


```python
import numpy as np
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
%matplotlib inline

from ph121c_lxvm import basis, tfim, tests, measure, data, tensor
```


```python
for oper_params in tests.tfim_sweep(
    L = [14],
    h = [1],
    bc= ['c'],
    sector=['f']
):
    job = dict(
        oper=tfim.x.H_sparse,
        oper_params=oper_params,
        solver=sla.eigsh,
        solver_params={ 
            'k' : 6, 
            'which' : 'BE',
        },
    )
    evals, evecs = data.jobs.obtain(**job)
    
```


```python
vals = basis.schmidt.values(
    # permute the gs so the the + and - sectors are in block forms
    evecs[basis.unitary.Ising(oper_params['L']), 0],
    # perform matricization and svd with respect to the +/- subsystems
    np.array([oper_params['L'] - 1]),
    oper_params['L']
)
print('singular values: ', vals)
print('entanglement entropy:', measure.entropy.entanglement(vals))
```

    singular values:  [1.00000000e+00 2.14532468e-14]
    entanglement entropy: -2.2204460492213453e-15


I'm rather confident that the code is correct because it passes all of my TFIM
consistency and basis interconsistency tests.
I might conclude that the existence of degenerate eigenspaces in the Hamiltonian
with respect to some symmetry operator produces no entanglement entropy
across the disjoint subsystems.

## Verifying local and nonlocal mps operators

Brenden helped me test my mps code by posing the challenge of verifying the
action of the Ising symmetry operator and of 1-point correlation functions
in the ferromagnetic $h < 1$ and paramagnetic $h > 1$ domains.
In both cases, the 1-point correlation operator $C^z_i = \ev{\sigma^z_i}$
should vanish in expectation, 
Also, the Ising symmetry operator $U = \prod_i \sigma^x_i$ should yield the sign
of the symmetry sector.
In the following calculation, we test these ideas in the mps formalism after
obtaining the ground states in the symmetry sectors and converting them to the
z basis.


```python
for oper_params in tests.tfim_sweep(
    L = [14],
    h = [0.3, 1.7],
    bc= ['o'],
    sector=['+', '-'],
):
    job = dict(
        oper=tfim.x.H_sparse,
        oper_params=oper_params,
        solver=sla.eigsh,
        solver_params={ 
            'k' : 6, 
            'which' : 'BE',
        },
    )
    evals, evecs = data.jobs.obtain(**job)
    gs = np.zeros(2 ** oper_params['L']) # in sector basis
    # expand to full x basis
    ## Insert into diagonal
    gs[((oper_params['sector'] == '-') * 2 ** ((oper_params['L'] - 1))) + np.arange(2 ** (oper_params['L'] - 1))] = evecs[:, 0]
    ## rotate diagonal into full
    gs = gs[basis.unitary.Ising(oper_params['L'], inverse=True)]
    # rotate to z basis
    gs = basis.unitary.Hadamard(oper_params['L']) @ gs
    # DO THE MPS
    chi_max = 5
    rank = tensor.bond_rank(chi_max, oper_params['L'], d=2)
    A = tensor.mps(gs, rank, oper_params['L'], 2)
    # find the expectation values of the operators
    sx = np.array([[0, 1], [1, 0]], dtype='float64')
    sz = np.array([[1, 0], [0, -1]], dtype='float64')
    U  = tensor.mpo(oper_params['L'], d=2)
    for i in range(U.L):
        U[i] = sx
    C = tensor.mpo(oper_params['L'], d=2)
    C[0] = sz
    print(
        'L =', oper_params['L'],
        ': h =', oper_params['h'],
        ': sector =', oper_params['sector'],
    )
    print('1-point correlation expval:', A.expval(C) / A.inner(A))
    print('Ising operator expval     :', A.expval(U) / A.inner(A))
    print('')
```

    L = 14 : h = 0.3 : sector = +
    1-point correlation expval: -1.3322676295501896e-15
    Ising operator expval     : 1.0
    
    L = 14 : h = 0.3 : sector = -
    1-point correlation expval: -1.942890293094023e-15
    Ising operator expval     : -0.9999999999999996
    
    L = 14 : h = 1.7 : sector = +
    1-point correlation expval: -2.2784508654782915e-16
    Ising operator expval     : 0.9999999999999998
    
    L = 14 : h = 1.7 : sector = -
    1-point correlation expval: 8.673617379884047e-17
    Ising operator expval     : -0.9999999999999998
    


That seems about right.
Thank god that the implementation is working.


```python

```
