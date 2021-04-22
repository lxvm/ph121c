! This module implements some solvers for sparse Hamiltonians
! and only those in CSR formats

include "mkl_spblas.f90"
include "mkl_solvers_ee.f90"

module solvers_sparse
    use mkl_spblas
    use mkl_solvers_ee
    implicit none

    private
    public mkl_sparse_ev, feast_sparse_ev

contains

    subroutine mkl_sparse_ev (sHam, L, m, evals, evecs)
        ! Use sparse Hamiltonian function to get some real eigenthings
        type (sparse_matrix_t)                    :: sHam
        integer,                   intent (in   ) :: L, m
        real, dimension (m),       intent (  out) :: evals
        real, dimension (2**L, m), intent (  out) :: evecs

        evals = 0
        evecs = 0
        stop 'not implemented'
    end subroutine mkl_sparse_ev

    
    subroutine feast_sparse_ev (sHam, L, m, evals, evecs)
        ! Use sparse Hamiltonian function to get some real eigenthings
        type (sparse_matrix_t)                    :: sHam
        integer,                   intent (in   ) :: L, m
        real, dimension (m),       intent (  out) :: evals
        real, dimension (2**L, m), intent (  out) :: evecs

        evals = 0
        evecs = 0
        stop 'not implemented'
    end subroutine feast_sparse_ev
    
end module solvers_sparse
