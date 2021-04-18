!include
module tfim
    !use
    implicit none

    private
    public save_matrix,  print_matrix,    &
           H_dense_kron_open, H_dense_kron_closed,  &
           H_vec_closed, H_vec_open,      &
           H_dense_closed,     H_dense_open,          &
           pack_matrix

contains

    subroutine save_matrix (M)
        real, dimension (:, :), intent (in) :: M
        integer i, j
        ! use default file name fort.1
        open (1)
        do i = 1, size(M, 1)
            do j = 1, size(M, 2)
                write (1, '(f8.5, " ")', advance='no') M(i, j)
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
                write (*, '(f8.5, " ")', advance='no') M(i, j)
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

    subroutine def_matrices (id, sz, sx)
        real, dimension (2, 2), intent (inout) :: id, sz, sx
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

    recursive function l_Nkron (n, A, B) result (C)
        ! take left-handed kronecker product
        ! of A with B n times: A (*) ... (*) A (*) B
        integer, intent (in) :: n
        real, dimension (:, :), intent (in)       :: A, B
        real, dimension ((size(A, 1) ** n) * size(B, 1), &
                         (size(A, 2) ** n) * size(B, 2)) :: C
        if (n >= 1) then
            C = l_Nkron(n-1, A, kron(A, B))
        else
            C = B
        end if
    end function l_Nkron
    
    recursive function r_Nkron (A, B, n) result (C)
        ! take right-handed kronecker product
        ! of A with B n times: A (*) B (*) ... (*) B
        integer, intent (in) :: n
        real, dimension (:, :), intent (in)       :: A, B
        real, dimension (size(A, 1) * (size(B, 1) ** n), &
                         size(A, 2) * (size(B, 2) ** n)) :: C
        if (n >= 1) then
            C = r_Nkron(kron(A, B), B,  n-1)
        else
            C = A
        end if
    end function r_Nkron
    
    function H_dense_kron_open (L, h) result (Ham)
        integer, intent (in) :: L
        real,    intent (in) :: h
        real, dimension (0:(2 ** L - 1), 0:(2 ** L - 1)) :: Ham
        real, dimension (2, 2) :: id, sz, sx
        integer i

        call def_matrices(id, sz, sx)
        Ham = 0
        do i = 0, L-2
            Ham = Ham &
                ! Contribution from sigma z (*) sigma z terms
                - kron(l_Nkron(i, id, sz), r_Nkron(sz, id, L-1-(i+1))) &
                ! Contribution from h * sigma x terms
                - h * l_Nkron(i, id, r_Nkron(sx, id, L-(i+1)))
        end do
        Ham = Ham - h * l_Nkron(L-1, id, sx)
    end function H_dense_kron_open

    function H_dense_kron_closed (L, h) result (Ham)
        integer, intent (in) :: L
        real,    intent (in) :: h
        real, dimension (0:(2 ** L - 1), 0:(2 ** L - 1)) :: Ham
        real, dimension (2, 2) :: id, sz, sx

        call def_matrices(id, sz, sx)
        Ham = H_dense_kron_open(L, h) - kron(sz, l_Nkron(L-2, id, sz))
    end function H_dense_kron_closed

    function H_vec_open (L, h, v) result (w)
        ! calculate H|v> in computational basis
        integer, intent (in) :: L
        real,    intent (in) :: h
        real, dimension (0:), intent (in) :: v
        real, dimension (0:((2**L)-1)) :: w
        integer i, j

        w = 0
        do i = 0, ((2**L)-1)
            if (v(i) /= 0) then
                ! flip bits per sigma x
                do j = 0, (L-1)
                    w(ibchng(i, j)) = w(ibchng(i, j)) + h * v(i)
                end do
                ! sign changes from sigma z
                do j = 0, (L-2)
                    if (xor(btest(i, j + 1), btest(i, j))) then
                        w(i) = w(i) - v(i)
                    else
                        w(i) = w(i) + v(i)
                    end if
                end do
            end if
        end do
        w = -w
    end function H_vec_open

    function H_vec_closed (L, h, v) result (w)
        ! calculate H|v> in computational basis
        integer, intent (in) :: L
        real,    intent (in) :: h
        real, dimension (0:), intent (in) :: v
        real, dimension (0:((2**L)-1)) :: w
        integer i, j
        w = 0
        do i = 0, ((2**L)-1)
            if (v(i) /= 0) then
                ! flip bits per sigma x
                do j = 0, (L-1)
                    w(ibchng(i, j)) = w(ibchng(i, j)) + h * v(i)
                end do
                ! sign changes from sigma z
                do j = 0, (L-2)
                    if (xor(btest(i, j + 1), btest(i, j))) then
                        w(i) = w(i) - v(i)
                    else
                        w(i) = w(i) + v(i)
                    end if
                end do
                if (xor(btest(i, 0), btest(i, L - 1))) then
                    w(i) = w(i) - v(i)
                else
                    w(i) = w(i) + v(i)
                end if
            end if
        end do
        w = -w
    end function H_vec_closed

    subroutine H_dense_open (L, h, Ham)
        integer,                     intent (in   ) :: L
        real,                        intent (in   ) :: h
        real,    dimension (0:, 0:), intent (inout) :: Ham
        real,    dimension (0:(2**L-1))             :: v
        integer i
        
        if (size(Ham, 1) < 2**L) stop 'input ham too small'
        if (size(Ham, 2) < 2**L) stop 'input ham too small'

        do i = 0, ((2**L) - 1)
            v    = 0
            v(i) = 1
            Ham(0:((2**L)-1), i) = H_vec_open(L, h, v)
        end do
    end subroutine H_dense_open

    subroutine H_dense_closed (L, h, Ham)
        integer,                     intent (in   ) :: L
        real,                        intent (in   ) :: h
        real,    dimension (0:, 0:), intent (inout) :: Ham
        real,    dimension (0:(2**L-1))             :: v
        integer i
        
        if (size(Ham, 1) < 2**L) stop 'input ham too small'
        if (size(Ham, 2) < 2**L) stop 'input ham too small'

        do i = 0, ((2**L) - 1)
            v    = 0
            v(i) = 1
            Ham(0:((2**L)-1), i) = H_vec_closed(L, h, v)
        end do
    end subroutine H_dense_closed

end module tfim
