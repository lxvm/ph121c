"""1D Transverse-Field Ising Model (TFIM) galore.

This package makes available functions for calculating Hamiltonians of 1D TFIMs
of the form shown: https://en.wikipedia.org/wiki/Transverse-field_Ising_model.
We take the liberty of reducing the model to be a function of one critical
parameter, h, which modulates the strength of the transverse field terms.

We implement functions that generate the Hamiltonian as a function of the
system length, L, the critical parameter, h, and open 'o' or closed 'c' boundary
conditions. These Hamiltonians are available in either the z basis or the x
basis in the corresponding modules z and x. Further, in the x module, we make
use of the Ising symmetry of the model to construct matrices in the + and -
symmetry sectors of the Hamiltonian, offering the possibility of computational
accelerations because of the reduced size of the symmetry sector. Thus when
using the x module, one must also specify the sector for full 'f', '+', or '-'
basis.

Each of these Hamiltonians can be utilized in three ways in both x and z modules.
To obtain the dense representation, call the H_dense function. There are 2
sparse representations of the Hamiltonian, one as a scipy sparse CSR matrix
available in the H_sparse function, and another as a scipy sparse LinearOperator
available in the H_oper function.

In terms of performance, these functions are implemented with (L + 1) * (2 ** L)
operations, which is exactly the number of matrix elements, and should run 

This module uses a computational backend written in Fortran, available in the
ph121c_lxvm_fortran_tfim package, also imported in ph121c_lxvm.fortran.
Additionally, if one desires, it is possible to calculate individual matrix
elements in this Fortran backend through the H_z_mel and H_x_mel functions.

Deprecated functions are prefixed by 'old' and are retained for reference as an
implementation in Python.

Refer to the docstrings for more details.
"""

__all__ = [
    'z',
    'x',
]

from . import *
