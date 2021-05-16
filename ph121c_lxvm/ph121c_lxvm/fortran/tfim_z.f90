module types
    ! https://stackoverflow.com/questions/12523524/f2py-specifying-real-precision-in-fortran-when-interfacing-with-python
    implicit none
    
    integer, parameter :: dp=kind(0.d0)
    
end module types


module tfim_z
    ! Transverse-Field Ising Model Hamiltonians (and their derivates) in z basis
    use types

    implicit none

    private
    public &
        H_mel, &
        H_coo, &
        H_vec, &
        long_H_vec,&
        mbl_H_vec, &
        sz_1_val, &
        sz_1_diag,&
        sz_1_mel, &
        sz_2_val, &
        sz_2_diag,&
        sz_2_mel, &
        sx_1_val, &
        sx_1_mel

    ! Suffix 'val' is a scalar & implies use in a do loop over a physical index
    ! Suffix 'mel' means calculation of a matrix element & is loopless
    ! Suffix 'diag' means calculation of a diagonal matrix element
    ! Suffix 'coo' means calculation of a matrix in sparse coordinate format
    ! Suffix 'vec' means calculation of a matrix-vector product
    ! Modifier '_?_' stands for a ?-site/?-local operator 

contains
    
    !!! Define Pauli operators
    
    real(dp) pure function sz_1_val (h)
        ! 1-site sigma z contribution to Hamiltonian in z basis
        real(dp), intent (in) :: h
        sz_1_val = -h
    end function sz_1_val

    real(dp) pure function sz_1_diag (i, L, h)
        ! 1-site sigma z contribution to Hamiltonian in z basis
        integer,  intent (in) :: i, L
        real(dp), intent (in) :: h
        sz_1_diag = sz_1_val(h) * (L - 2 * popcnt(i))
    end function sz_1_diag
    
    real(dp) pure function sz_1_mel (i, j, L, h)
        ! 1-site sigma z contribution to Hamiltonian in z basis
        integer,  intent (in) :: i, j, L
        real(dp), intent (in) :: h
        
        if (i == j) then
            sz_1_mel = sz_1_diag(i, L, h)
        else
            sz_1_mel = 0
        end if
    end function sz_1_mel
    
    real(dp) pure function sz_2_val ()
        sz_2_val = -1
    end function sz_2_val

    real(dp) pure function sz_2_diag (i, L, bc)
        ! 2-site sigma z in z basis contribution on diagonal
        ! for each bit in system:
        ! if same as neighbor: +1
        ! else: -1
        character,intent (in) :: bc
        integer,  intent (in) :: i, L

        if (bc == 'c') then
            ! in closed system, do a cyclic shift
            sz_2_diag = sz_2_val() &
                * (L - 2 * popcnt(ieor(ishftc(i, 1, L), i)))
        else if (bc == 'o') then
            ! in open system, shift non-cyclically
            sz_2_diag = sz_2_val() &
                * (L - 1 - 2 * popcnt(ieor(ishft(i, -1), ibclr(i, L - 1))))
        else
            ! invalid boundary condition
            sz_2_diag = 0
        end if
    end function sz_2_diag
    
    real(dp) pure function sz_2_mel (i, j, L, bc)
        ! 2-site sigma z in z basis matrix element
        ! for each bit in system:
        ! if same as neighbor: +1
        ! else: -1
        character,intent (in) :: bc
        integer,  intent (in) :: i, j, L

        if (i == j) then
            sz_2_mel = sz_2_diag(i, L, bc)
        else
            sz_2_mel = 0
        end if
    end function sz_2_mel
    
    real(dp) pure function sx_1_val (h)
        real(dp), intent (in) :: h
        sx_1_val = -h
    end function sx_1_val
    
    real(dp) pure function sx_1_mel (i, j, h)
        ! Calculate matrix element of 1-site sigma x in z basis
        integer,  intent (in) :: i, j
        real(dp), intent (in) :: h
        
        if (popcnt(ieor(i, j)) == 1) then
            ! sigma x contribution for Hamming distance of 1
            sx_1_mel = sx_1_val(h)
        else 
            sx_1_mel = 0
        end if
    end function sx_1_mel

    !!! Define Hamiltonian operators

    real(dp) pure function H_mel (i, j, L, h, bc)
        character,intent (in) :: bc
        integer,  intent (in) :: i, j, L
        real(dp), intent (in) :: h
        ! This function is a reference for how to calculate matrix elements.
        H_mel = sz_2_mel(i, j, L, bc) + sx_1_mel(i, j, h)
    end function H_mel

    pure subroutine H_coo (elem, rows, cols, N, L, h, bc)
        ! Build H as sparse matrix in COO format
        ! Note: N = (L + 1) * (2 ** L)
        character,intent (in   ) :: bc
        integer,  intent (in   ) :: L, N
        real(dp), intent (in   ) :: h
        real(dp), intent (  out) :: elem(0:(N - 1))
        integer,  intent (  out) :: rows(0:(N - 1)), &
                                    cols(0:(N - 1))
        integer i, j, k
        k = 0

        do i = 0, ((2 ** L) - 1)
            ! spin flips, sigma x
            do j = 0, L - 1
                elem(k) = sx_1_val(h)
                rows(k) = i
                cols(k) = ieor(i, 2 ** j)
                k = k + 1
            end do
            ! sign flips, sigma z
            elem(k) = sz_2_diag(i, L, bc)
            rows(k) = i
            cols(k) = i
            k = k + 1
        end do
    end subroutine H_coo
        
    pure subroutine H_vec (v, N, L, h, bc, w)
        ! calculate H|v> in computational basis
        ! Note: N = 2 ** L
        character, intent (in   ) :: bc
        integer,   intent (in   ) :: L, N
        real(dp),  intent (in   ) :: h
        real(dp),  intent (in   ) :: v(0:(N - 1))
        real(dp),  intent (  out) :: w(0:(N - 1))
        integer i, j, m
        
        w = 0
        do i = 0, ((2**L)-1)
            if (v(i) /= 0) then
                ! spin flips from sigma x
                do j = 0, (L-1)
                    m = ieor(i, 2 ** j)
                    w(m) = w(m) + v(i) * sx_1_val(h)
                end do
                ! sign flips from sigma z
                w(i) = w(i) + v(i) * sz_2_diag(i, L, bc)
            end if
        end do
    end subroutine H_vec
    
    pure subroutine long_H_vec (v, N, L, hx, hz, bc, w)
        ! calculate H|v> in the computational z basis
        ! including a longitudintal 1-site sigma z term at each site
        ! Note: N = 2 ** L
        character, intent (in   ) :: bc
        integer,   intent (in   ) :: L, N
        real(dp),  intent (in   ) :: hx, hz
        real(dp),  intent (in   ) :: v(0:(N - 1))
        real(dp),  intent (  out) :: w(0:(N - 1))
        integer i
        
        call H_vec(v, N, L, hx, bc, w)
        do i = 0, ((2**L)-1)
            if (v(i) /= 0) then
                ! sign flips from sigma z
                w(i) = w(i) + v(i) * sz_1_diag(i, L, hz)
            end if
        end do
    end subroutine long_H_vec

    pure subroutine mbl_H_vec (v, N, L, hx, hz, bc, w)
        ! calculate H|v> in computational z basis using 
        ! disordered coefficients to introduce many-body localization (mbl)
        ! Note: N = 2 ** L
        character, intent (in   ) :: bc
        integer,   intent (in   ) :: L, N
        real(dp),  intent (in   ) :: hx(0:(L - 1)), hz(0:(L - 1))
        real(dp),  intent (in   ) :: v(0:(N - 1))
        real(dp),  intent (  out) :: w(0:(N - 1))
        integer i, j, m
        
        w = 0
        do i = 0, ((2**L)-1)
            if (v(i) /= 0) then
                do j = 0, (L-1)
                    ! spin flips from sigma x
                    m = ieor(i, 2 ** j)
                    w(m) = w(m) + v(i) * sx_1_val(hx(j))
                    ! sign flips from 1-site sigma z
                    if (btest(i, j)) then
                        w(i) = w(i) - v(i) * sz_1_val(hz(j))
                    else
                        w(i) = w(i) + v(i) * sz_1_val(hz(j))
                    end if
                end do
                ! sign flips from 2-site sigma z
                w(i) = w(i) + v(i) * sz_2_diag(i, L, bc)
            end if
        end do
    end subroutine mbl_H_vec
    
end module tfim_z
