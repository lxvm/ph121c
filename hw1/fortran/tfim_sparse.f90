! This module creates sparse TFIM Hamiltonians
! in Sparse BLAS CSR format
! It's goal is to provide the same functionality as
! the tfim_dense module 

include "mkl_spblas.f90"
include "mkl_solvers_ee.f90"

module tfim_sparse
    use mkl_spblas
    use mkl_solvers_ee
    implicit none

    private
    public H_open_kron,  H_closed_kron,  &
           H_vec_closed, H_vec_open,     &
           H_closed_vec, H_open_vec

contains

    subroutine def_matrices
        ! Define the identity and Pauli z and x matrices

        stop "not implemented"
    end subroutine def_matrices


    function kron
        ! Get the Kronecker product of two matrices
        
        stop "not implemented"
    end function kron


    recursive function kron_left
        ! Take left-handed Kronecker product
        ! of A with B n times: A (*) ... (*) A (*) B

        stop "not implemented"
    end function kron_left

    
    recursive function kron_right
        ! Take right-handed Kronecker product
        ! of A with B n times: A (*) B (*) ... (*) B

        stop "not implemented"
    end function kron_right

    
    function H_open_kron
        ! Construct H using Kronecker product method
        ! with open boundary conditions

        stop "not implemented"
    end function H_open_kron


    function H_closed_kron
        ! Construct H using Kronecker product method
        ! with closed boundary conditions

        stop "not implemented"
    end function H_closed_kron


    function H_vec_open
        ! calculate H|v> in computational basis
        ! with open boundary conditions

        stop "not implemented"
    end function H_vec_open


    function H_vec_closed
        ! calculate H|v> in computational basis
        ! with closed boundary conditions

        stop "not implemented"
    end function H_vec_closed


    subroutine H_open_vec
        ! Construct H using vector method
        ! with open boundary conditions
        
        stop "not implemented"
    end subroutine H_open_vec


    subroutine H_closed_vec
        ! Construct H using vector method
        ! with closed boundary conditions

        stop "not implemented"
    end subroutine H_closed_vec

end module tfim_sparse
