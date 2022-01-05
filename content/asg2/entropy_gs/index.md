---
title: "Assignment 2: Part 2"
date: 2021-05-06T01:27:05-07:00
draft: false
---
# Entanglement entropy of highly excited states

## Introduction

We are interested in calculating the entanglement entropy for states at the
middle of the spectrum, which sparse solvers do not do well (though it is
ameliorated by transformations like shift-invert mode).
Instead, I will obtain states via dense diagonalization

## Program

- Calculate wavefunctions at center of spectrum using dense solver
- Repeat entropy calculations as in part 2


```python
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
%matplotlib inline

from ph121c_lxvm import tfim, basis, tests, measure, data
```


```python
%%time
entropies = {
    'L' : [],
    'h' : [],
    'l' : [],
    'S' : [],
    'bc': [],
}

for oper_params in tests.tfim_sweep(
    L = list(range(4,15)),
    h = [0.3],
    bc= ['c'],
):
    job = dict(
        oper=tfim.z.H_dense,
        oper_params=oper_params,
        solver=la.eigh,
        solver_params={ 
            'subset_by_index' : list(np.array([-2, 3]) + 2 ** (oper_params['L'] - 1)),
        },
    )
    evals, evecs = data.jobs.obtain(**job)
        
    es = evecs[:, 5]
    print('using state with eigenvalue:', evals[5])
    A = []
    for l in range(oper_params['L']-1):
        A.append(l)
        entropies['S'].append(
            measure.entropy.entanglement(
                basis.schmidt.values(es, A, oper_params['L'])
            )
        )
        entropies['l'].append(l)
        entropies['L'].append(oper_params['L'])
        entropies['h'].append(oper_params['h'])
        entropies['bc'].append(oper_params['bc'])

df = pd.DataFrame(entropies)
```

    using state with eigenvalue: 0.6
    using state with eigenvalue: -0.255107460434849
    using state with eigenvalue: 1.4196736589527976
    using state with eigenvalue: 0.7000000000000011
    using state with eigenvalue: 1.3541254488577767e-16
    using state with eigenvalue: -0.5651924049830526
    using state with eigenvalue: 0.5813865554288129
    using state with eigenvalue: 0.5809819774954872
    using state with eigenvalue: 1.227076422501945e-15
    using state with eigenvalue: -0.5150533998116924
    using state with eigenvalue: 0.00034525367950669615
    CPU times: user 236 ms, sys: 12 ms, total: 248 ms
    Wall time: 53.7 ms



```python
def plot_script(df, bc):
    """Make the display plots."""
    L = sorted(set(df.L))
    h = sorted(set(df.h))
    w = 2

    fig_l, axes_l = plt.subplots(len(L)//w+len(L)%w, w)
    for i, row in enumerate(axes_l):
        for j, ax in enumerate(row):
            if w*i + j < len(L):
                for s in h:
                    sub = df[(df.h==s) & (df.L==L[w*i+j]) & (df.bc==bc)]
                    ax.plot(sub.l.values, sub.S.values, label='h='+str(s))
                ax.set_xlabel(f'$l$ at $L={L[w*i+j]}$')
                ax.set_ylabel('$S$')
                handles, labels = ax.get_legend_handles_labels()
            else:
                ax.set_axis_off()
                ax.legend(handles, labels, loc='center')
    st = fig_l.suptitle('Entanglement entropies')
    fig_l.set_size_inches(5, 6)
    fig_l.tight_layout()
    st.set_y(.95)
    fig_l.subplots_adjust(top=.9)

    fig_L, ax_L = plt.subplots()
    for s in h:
        sub = df[(df.h==s) & (df.l==df.L//2) & (df.bc==bc)]
        ax_L.plot(sub.L.values, sub.S.values, label='h='+str(s))
    ax_L.set_title('Entanglement entropy at half-subsystem')
    ax_L.set_xlabel('$L$')
    ax_L.set_ylabel('$S$')
    ax_L.legend()
    fig_L.set_size_inches(3, 3)

    return (fig_l, fig_L)
```


```python
%%capture
figs = plot_script(df, 'c')
```

## Results

The main goal is to observe a different scaling of entropy for a highly-excited
state which is not a band state.
In fact this is what we observe for the Hamiltonian in the ferromagnetic phase
and with periodic boundary conditions.


```python
figs[0]
```




    
![png](output_7_0.png)
    



It really is different behavior than the constant for the area law in the band
states.
From smaller $L=4$ to larger systems, $L=14$, the same pattern emerges where
the entanglement entropy has a maximum at $\ell = L/2$ and appears to decay
linearly away from that maximum.

Next we how a summary of the growth of the entanglement entropy at half
subsystems for various $L$:


```python
figs[1]
```




    
![png](output_9_0.png)
    



## Discussion

Instead of observing a constant entanglement entropy with respect to subsystems,
as in the extremal eigenstates, the states at the center of the spectrum display
new behaviors: increasing entropy with respect to system size and also with
respect to subsystem size.
One might expect that the entropy increases at equipartition, and indeed, this
is what we are observing.
From this perspective, the excited states are more random than those at the bands.


```python

```
