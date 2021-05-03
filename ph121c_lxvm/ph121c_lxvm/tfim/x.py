"""Provides Transverse-Field Ising Model Hamiltonians in x basis.

Exploits the Ising symmetry sectors to reduce the computational cost.

Supports closed and open (default) boundary conditions.
Supports a single parameter h for tuning transver field.
Supports the following formats for storing the Hamiltonian:
- numpy.ndarray
- scipy.sparse.linalg.LinearOperator
- scipy.sparse.linalg.csr_matrix
"""

