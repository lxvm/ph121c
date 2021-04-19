! This module implements some solvers for dense Hamiltonians
! or those which use dense representations

module solvers_dense
    use lapack95
    use tfim_dense
    implicit none

    private
    public my_lanczos

contains

    subroutine my_lanczos (fHam, L, m, evals, evecs, job)
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
        real, dimension (2**L, m), intent (  out), target :: evecs
        character, intent (in) :: job
        ! Local variables
        real, dimension (:, :), pointer :: V
        real, dimension (m, 2)    :: T
        real, dimension (m, m)    :: T_evecs
        real, dimension (2**L)    :: tmp
        integer i, j

        if (m > 2**L) stop "requested too many eigenvectors"
        ! Symmetric tridiagonal matrix T stored as 2 columns
        ! First is m diagonal entries
        ! Second is m-1 off-diagonal entries with mth for work
        T = 0
        ! initial vector
        call seed(1293)
        V       => evecs
        V       = 0
        V(:, 1) = get_rand_vector(size(V, 1))
        ! initial iteration step
        tmp     = fHam(V(:, 1))
        T(1, 1) = sum(tmp * V(:, 1))
        tmp     = tmp - T(1, 1) * V(:, 1)
        
        ! general loop
        do j = 2, m
            ! Off-diagonal
            T(j-1, 2) = sqrt(sum(tmp * tmp))
            ! Choose new vector v
            if (T(j-1, 2) /= 0) then
                V(:, j) = tmp / T(j-1, 2)
            else
                ! Choose a new vector orthogonal to those in V
                V(j, j) = 1
                if (any(matmul(transpose(V(:, :j-1)), V(:, j)) /= 0)) then
                    V(:, j) = get_null_vector(V(:, :j))
                end if
            end if
            ! Find transformation
            tmp     = fHam(V(:, j))
            ! Diagonal
            T(j, 1) = sum(tmp * V(:, j))
            ! Next iterate
            tmp     = tmp - T(j, 1) * V(:, j) - T(j-1, 2) * V(:, j-1)
        end do

        ! Diagonalize T (segfault on L=10, m=2**10  with evecs)
        evals = T(:, 1)
        if (job == 'N') then
            call stev(evals, T(:, 2))
        else if (job == 'V') then
            ! increase the stack limit, otherwise this will cause segfault
            call stev(evals, T(:, 2), T_evecs)
            evecs = matmul(V, T_evecs)
        end if
    end subroutine my_lanczos


    function get_rand_vector (n) result (v)
        ! sample a vector of length n randomly on U[-.5, .5] and normalize it
        integer, intent (in) :: n
        real, dimension (n)  :: v
        integer i
        
        do i = 1, n
           call random_number(v(i))
        end do
        ! Center
        v = v - 0.5
        ! Normalize
        v = v / sqrt(sum(v * v))
    end function get_rand_vector


    function get_null_vector (A) result (x)
        ! Find a vector in the null space of A
        real, dimension (:, :),       intent (in) :: A
        real, dimension (size(A, 1))              :: x
        x=0

        stop 'not implemented'
        ! According to
        ! https://scicomp.stackexchange.com/questions/10185/solving-for-null-space-of-a-matrix-with-mkl-lapack
        ! It is possible to find a vector in the kernel of a matrix
        ! via the QR decomposition
        ! It can also be done via Gaussian elimination
    end function get_null_vector

end module solvers_dense
