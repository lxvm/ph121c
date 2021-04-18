include "mkl_spblas.f90"
include "mkl_solvers_ee.f90"

module sparse_solvers
    use lapack95
    use mkl_spblas
    use mkl_solvers_ee
    implicit none

    private
    public my_lanczos

contains

    subroutine mkl_sparse_exe ()
        integer, dimension (128) :: pm
        integer stat
        stat = mkl_sparse_ee_init(pm)        
    end subroutine mkl_sparse_exe 

    subroutine my_lanczos (fHam, L, m, evals, evecs)
        ! Use my Hamiltonian function to get some real eigenthings
        abstract interface
            function H_vec (v) result (w)
                ! calculate H|v> in computational basis
                real, dimension (:), intent (in) :: v
                real, dimension (size(v))        :: w
            end function H_vec
        end interface
        procedure (H_vec)                         :: fHam
        integer,                   intent (in   ) :: L, m
        real, dimension (m),       intent (  out) :: evals
        real, dimension (2**L, m), intent (  out) :: evecs
        ! Local variables
        real, dimension (m, 2)    :: T
        real, dimension (2**L, m) :: V
        real, dimension (m, m)    :: T_evecs
        real, dimension (2**L)    :: tmpa, tmpb
        integer i, j

        if (m > 2**L) stop "requested too many eigenvectors"

        ! Symmetric tridiagonal matrix T stored as 2 columns
        ! First is m diagonal entries
        ! Second is m-1 off-diagonal entries with mth for work
        T = 0
        print *, 'T mat'
        print *, T(:5, :)
        ! initial vector
        V       = 0
        V(1, 1) = 1
        ! initial iteration step
        tmpa    = fHam(V(:, 1))
        print *, 'tmpa'
        print *, tmpa(:10)
        T(1, 1) = sum(tmpa * V(:, 1))
        print *, 'T11'
        print *, T(1, 1)
        tmpb    = tmpa - T(1, 1) * V(:, 1)
        print *, 'tmpb'
        print *, tmpb(:10)

        ! general loop
        do j = 2, m
            ! Off-diagonal
            T(j-1, 2) = sum(tmpb * tmpb)
            print *, 'Tjj-1'
            print *, T(j-1, 2)
            ! Choose new vector v
            if (T(j-1, 2) /= 0) then
                V(:, j) = tmpb / T(j-1, 2)
            else
                ! Choose a new vector orthogonal to those in V
                V(j, j) = 1
                if (any(matmul(transpose(V(:, :j-1)), V(:, j)) /= 0)) then
                    V(:, j) = get_null_vector(V(:, :j))
                end if
            end if
            ! Find transformation
            tmpa    = fHam(V(:, j))
            print *, 'tmpa'
            print *, tmpa(:10)
            ! Diagonal
            T(j, 1) = sum(tmpa * V(:, j))
            print *, 'Tjj'
            print *, T(j, 1)
            ! Next iterate
            tmpb    = tmpa - T(j, 1) * V(:, j) - T(j-1, 2) * V(:, j-1)
            print *, 'tmpb'
            print *, tmpb(:10)
        end do

        print *, 'T mat'
        print *, T(:5, :)
        ! Diagonalize T
        evals = T(:, 1)
        call stev(evals, T(:, 2), T_evecs)
        evecs = matmul(V, T_evecs)
        
    end subroutine my_lanczos

    function get_null_vector (A) result (x)
        ! Find a vector in the null space of A
        real, dimension (:, :),       intent (in) :: A
        real, dimension (size(A, 1))              :: x
        x=0

        stop 'not implemented and not necessary for full-rank Hermitian matrix'
        ! According to
        ! https://scicomp.stackexchange.com/questions/10185/solving-for-null-space-of-a-matrix-with-mkl-lapack
        ! It is possible to find a vector in the kernel of a matrix
        ! via the QR decomposition
    end function get_null_vector

end module sparse_solvers
