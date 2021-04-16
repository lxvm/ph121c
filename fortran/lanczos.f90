module lanczos
    use blas95
    use tfim
    implicit none

    private
    public f

contains

    real function f (x)
        real, intent (in) :: x
        f = x
    end function f

    subroutine real_lanczos(fHam, L, h, m, evals)
        abstract interface
            function H_vec (L, h, v) result (w)
                ! calculate |Hv> in computational basis
                integer, intent (in) :: L
                real,    intent (in) :: h
                real, dimension(0:), intent (in) :: v
                real, dimension(0:((2**L)-1)) :: w
            end function H_vec
        end interface
        procedure (H_vec) :: fHam
        integer, intent (in) :: L, m
        real,    intent (in) :: h
        real, dimension(size(M, 1)), intent (out) :: evals
        real, dimension(0:((2 ** L) - 1), 0:m) :: w
        real, dimension(0:((2 ** L) - 1), 0:m) :: tmp
        integer i, j

        if (size(A, 1) /= size(A, 2)) stop "matrix not square"
        ! initial iteration step
        w = 0
        w(1, 0) = 1
        tmp     = fHam(w(:, 0))
        w(:, 1) = tmp - sum(tmp * w(:, 0)) * w(:, 0)
        do j = 2, m
            
        end do
        
    end subroutine

end module lanczos
