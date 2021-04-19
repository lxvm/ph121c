! This module contains some helper functions for dense matrices

module matrix_helpers_dense
    implicit none

    private
    public save_matrix, print_matrix, pack_matrix

contains

    subroutine save_matrix (M)
        real, dimension (:, :), intent (in) :: M
        integer i, j
        ! use default file name fort.1
        open (1)
        do i = 1, size(M, 1)
            do j = 1, size(M, 2)
                write (1, '(f25.15, " ")', advance='no') M(i, j)
            end do
            write (1, '(a)')
        end do
        close (1)
    end subroutine save_matrix


    subroutine print_matrix (M)
        real, dimension (:, :), intent (in) :: M
        integer i, j
        do i = 1, size(M, 1)
            do j = 1, size(M, 2)
                write (*, '(f8.3, " ")', advance='no') M(i, j)
            end do
            write (*, '(a)')
        end do
    end subroutine print_matrix


    function pack_matrix (M) result (A)
        ! This packs a real symmetric matrix's
        ! upper diagonal for use in lapack routines
        real, dimension (:, :), intent (in) :: M
        real, dimension (size(M, 1) + size(M, 2) * (size(M, 2) - 1) / 2) :: A
        integer i, j

        if (size(M, 1) /= size(M, 2)) stop 'matrix not square'
        A = 0
        do j = 1, size(M, 2)
            do i = 1, j
                A(i + j * (j - 1) / 2) = M(i, j)
            end do
        end do
    end function pack_matrix

end module matrix_helpers_dense
