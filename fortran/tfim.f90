!include
module tfim
    !use
    implicit none

    private
    public def_matrices

contains

    subroutine def_matrices (id, sz, sx)
        real, dimension(2, 2), intent(inout) :: id, sz, sx
        id(:, 1) = (/ 1 , 0 /)
        id(:, 2) = (/ 0 , 1 /)
        sz(:, 1) = (/ 1 , 0 /)
        sz(:, 2) = (/ 0 ,-1 /)
        sx(:, 1) = (/ 0 , 1 /)
        sx(:, 2) = (/ 1 , 0 /)
    end subroutine def_matrices

    function kron (A, B) result (C)
        real, dimension (:, :), intent (in)       :: A, B
        real, dimension (size(A, 1) * size(B, 1), &
                         size(A, 2) * size(B, 2)) :: C
        integer i, j, sizeB1, sizeB2
        
        sizeB1 = size(B, 1)
        sizeB2 = size(B, 2)
        do j = 1, size(A, 2)
            do i = 1, size(A, 1)
                C(((i - 1) * sizeB1 + 1):(i * sizeB1), &
                  ((j - 1) * sizeB2 + 1):(j * sizeB2)) &
                    = A(i, j) * B
            end do
        end do         
    end function kron

    function H_dense_open_kron (L, h) result (Ham)
        integer, intent (in) :: L
        real,    intent (in) :: h
        real, dimension(0:(2 ** L - 1), 0:(2 ** L - 1)) :: Ham
        real, dimension(2, 2) :: id, sz, sx
        integer i

        call def_matrices(id, sz, sx)
        Ham = 0
        do i = 0, L
            Ham = Ham &
                ! Contribution from sigma z (*) sigma z terms
                + kron(id, sz) &
                ! Contribution from h * sigma x terms
                + h *kron(id, sx)
        end do
    end function H_dense_open_kron
        
end module tfim
