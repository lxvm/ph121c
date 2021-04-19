! This module creates dense TFIM Hamiltonians

module tfim_dense
    implicit none

    private
    public H_open_kron,  H_closed_kron,  &
           H_vec_closed, H_vec_open,     &
           H_closed_vec, H_open_vec,     &
           corr_sigma_z, var_magnetization
           
contains

    subroutine def_matrices (id, sz, sx)
        ! Define the identity and Pauli z and x matrices
        real, dimension (2, 2), intent (out) :: id, sz, sx

        id(:, 1) = (/ 1 , 0 /)
        id(:, 2) = (/ 0 , 1 /)
        sz(:, 1) = (/ 1 , 0 /)
        sz(:, 2) = (/ 0 ,-1 /)
        sx(:, 1) = (/ 0 , 1 /)
        sx(:, 2) = (/ 1 , 0 /)
    end subroutine def_matrices


    function kron (A, B) result (C)
        ! Get the Kronecker product of two matrices
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


    recursive function kron_left (n, A, B) result (C)
        ! Take left-handed Kronecker product
        ! of A with B n times: A (*) ... (*) A (*) B
        integer, intent (in) :: n
        real, dimension (:, :), intent (in)       :: A, B
        real, dimension ((size(A, 1) ** n) * size(B, 1), &
                         (size(A, 2) ** n) * size(B, 2)) :: C

        if (n >= 1) then
            C = kron_left(n-1, A, kron(A, B))
        else
            C = B
        end if
    end function kron_left

    
    recursive function kron_right (A, B, n) result (C)
        ! Take right-handed Kronecker product
        ! of A with B n times: A (*) B (*) ... (*) B
        integer, intent (in) :: n
        real, dimension (:, :), intent (in)       :: A, B
        real, dimension (size(A, 1) * (size(B, 1) ** n), &
                         size(A, 2) * (size(B, 2) ** n)) :: C

        if (n >= 1) then
            C = kron_right(kron(A, B), B,  n-1)
        else
            C = A
        end if
    end function kron_right

    
    function H_open_kron (L, h) result (Ham)
        ! Construct H using Kronecker product method
        ! with open boundary conditions
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
                - kron(kron_left(i, id, sz), kron_right(sz, id, L-1-(i+1))) &
                ! Contribution from h * sigma x terms
                - h * kron_left(i, id, kron_right(sx, id, L-(i+1)))
        end do
        Ham = Ham - h * kron_left(L-1, id, sx)
    end function H_open_kron


    function H_closed_kron (L, h) result (Ham)
        ! Construct H using Kronecker product method
        ! with closed boundary conditions
        integer, intent (in) :: L
        real,    intent (in) :: h
        real, dimension (0:(2 ** L - 1), 0:(2 ** L - 1)) :: Ham
        real, dimension (2, 2) :: id, sz, sx

        call def_matrices(id, sz, sx)
        Ham = H_open_kron(L, h) - kron(sz, kron_left(L-2, id, sz))
    end function H_closed_kron


    function H_vec_open (L, h, v) result (w)
        ! calculate H|v> in computational basis
        ! with open boundary conditions
        integer, intent (in) :: L
        real,    intent (in) :: h
        real, dimension (0:), intent (in) :: v
        real, dimension (0:((2**L)-1)) :: w
        integer i, j

        w = 0
        do i = 0, ((2**L)-1)
            if (v(i) /= 0) then
                do j = 0, (L-2)
                    ! flip bits per sigma x
                    w(ibchng(i, j)) = w(ibchng(i, j)) + h * v(i)
                    ! sign changes from sigma z
                    if (xor(btest(i, j + 1), btest(i, j))) then
                        w(i) = w(i) - v(i)
                    else
                        w(i) = w(i) + v(i)
                    end if
                end do
                ! boundary cases
                ! Last sigma x
                w(ibchng(i, L-1)) = w(ibchng(i, L-1)) + h * v(i)
            end if
        end do
        w = -w
    end function H_vec_open


    function H_vec_closed (L, h, v) result (w)
        ! calculate H|v> in computational basis
        ! with closed boundary conditions
        integer, intent (in) :: L
        real,    intent (in) :: h
        real, dimension (0:), intent (in) :: v
        real, dimension (0:((2**L)-1)) :: w
        integer i, j

        w = 0
        do i = 0, ((2**L)-1)
            if (v(i) /= 0) then
                do j = 0, (L-2)
                    ! flip bits per sigma x
                    w(ibchng(i, j)) = w(ibchng(i, j)) + h * v(i)
                    ! sign changes from sigma z
                    if (xor(btest(i, j + 1), btest(i, j))) then
                        w(i) = w(i) - v(i)
                    else
                        w(i) = w(i) + v(i)
                    end if
                end do
                ! boundary cases
                ! last sigma x
                w(ibchng(i, L-1)) = w(ibchng(i, L-1)) + h * v(i)
                ! closed boundary term sigma z
                if (xor(btest(i, 0), btest(i, L - 1))) then
                    w(i) = w(i) - v(i)
                else
                    w(i) = w(i) + v(i)
                end if
            end if
        end do
        w = -w
    end function H_vec_closed


    subroutine H_open_vec (L, h, Ham)
        ! Construct H using vector method
        ! with open boundary conditions
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
    end subroutine H_open_vec


    subroutine H_closed_vec (L, h, Ham)
        ! Construct H using vector method
        ! with closed boundary conditions
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
    end subroutine H_closed_vec


    function corr_sigma_z (i, j, v) result (corr)
        ! Compute the correlation C^{zz} between sites i, j
        ! This function is zero-indexed
        integer,                intent (in) :: i, j
        real,   dimension (0:), intent (in) :: v
        real,   dimension (0:(size(v)-1))   :: w
        real corr
        integer k

        w = 0
        do k = 0, (size(v)-1)
            if (v(k) /= 0) then
                ! sign changes from sigma z
                if (xor(btest(k, i), btest(k, j))) then
                    w(k) = w(k) - v(k)
                else
                    w(k) = w(k) + v(k)
                end if
            end if
        end do
        corr = sum(w * v)
    end function corr_sigma_z


    function var_magnetization (L, v) result (mag)
        ! Compute the magnetization variance of a spin chain per site
        integer,                intent (in) :: L
        real,   dimension (0:), intent (in) :: v
        real mag
        integer i

        mag = 0
        do i = 0, L-1
            mag = mag + corr_sigma_z(0, i, v)
        end do
    end function var_magnetization

end module tfim_dense
