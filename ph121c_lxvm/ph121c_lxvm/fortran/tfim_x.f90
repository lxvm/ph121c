module types
    ! https://stackoverflow.com/questions/12523524/f2py-specifying-real-precision-in-fortran-when-interfacing-with-python
    implicit none
    
    integer, parameter :: dp=kind(0.d0)
    
end module types


module tfim_x
    ! Transverse-Field Ising Model Hamiltonians (and their derivates) in x basis
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
        sz_1_mel, &
        sz_2_val, &
        sz_2_mel, &
        sx_1_val, &
        sx_1_diag,&
        sx_1_mel, &
        set_bounds,   &
        convert_full, &
        convert_sector

    ! Suffix 'val' is a scalar & implies use in a do loop over a physical index
    ! Suffix 'mel' means calculation of a matrix element & is loopless
    ! Suffix 'diag' means calculation of a diagonal matrix element
    ! Suffix 'coo' means calculation of a matrix in sparse coordinate format
    ! Suffix 'vec' means calculation of a matrix-vector product
    ! Modifier '_?_' stands for a ?-site/?-local operator at coordinate i 

contains
    
    !!! Define Pauli operators
    
    real(dp) pure function sz_1_val (h)
        real(dp), intent (in) :: h
        sz_1_val = -h
    end function sz_1_val

    real(dp) pure function sz_1_mel (i, j, h)
        ! 1-site sigma z contribution to Hamiltonian in z basis
        integer,  intent (in) :: i, j
        real(dp), intent (in) :: h
        
        if (popcnt(ieor(i, j)) == 1) then
            ! sigma x contribution for Hamming distance of 1
            sz_1_mel = sz_1_val(h)
        else 
            sz_1_mel = 0
        end if
    end function sz_1_mel
    
    real(dp) pure function sz_2_val ()
        sz_2_val = -1
    end function sz_2_val

    real(dp) pure function sz_2_mel (i, j, L, bc)
        ! 2-site sigma z contribution for Hamming distance of 2
        ! but only for adjacent bits (if closed, match ends too)
        character,intent (in) :: bc
        integer,  intent (in) :: i, j, L
        
        if (popcnt(ieor(i, j)) == 2) then
            if (bc == 'c') then
                ! popcnt(ieor(ishftc(ieor(i, j), 1, L), ieor(i, j))) == 1
                if (ieor(ishftc(ieor(i, j), 1, L), ieor(i, j)) /= 0) then
                    sz_2_mel = sz_2_val() 
                else
                    sz_2_mel = 0
                end if
            else if (bc == 'o') then
                ! popcnt(ieor(ishft(ieor(i, j), 1), ieor(i, j))) == 1
                if (ieor(ishft(ieor(i, j), 1), ieor(i, j)) /= 0) then
                    sz_2_mel = sz_2_val()
                else
                    sz_2_mel = 0
                end if
            else
                ! not valid bc
                sz_2_mel = 0
            end if
        else
            sz_2_mel = 0
        end if
    end function sz_2_mel

    real(dp) pure function sx_1_val (h)
        ! 1-site sigma x coefficient
        real(dp), intent (in) :: h
        sx_1_val = -h
    end function sx_1_val

    real(dp) pure function sx_1_diag (i, L, h)
        ! 1-site sigma x contribution where for each spin
        ! add 1 if bit position at position is zero (spin down, + sector)
        ! else subtract 1
        integer,  intent (in) :: i, L
        real(dp), intent (in) :: h
        sx_1_diag = sx_1_val(h) * (L - 2 * popcnt(i))
    end function sx_1_diag
    
    real(dp) pure function sx_1_mel (i, j, L, h)
        ! 1-site sigma x contribution where for each spin
        ! add 1 if bit position at position is zero (spin down, + sector)
        ! else subtract 1
        integer,  intent (in) :: i, j, L
        real(dp), intent (in) :: h
        
        if (i == j) then
            sx_1_mel = sx_1_diag(i, L, h)
        else
            sx_1_mel = 0
        end if
    end function sx_1_mel
    
    !!! Define utilities to optimizing array size and convert sector indices
    
    pure subroutine set_bounds (lz, dim, L, bc, sector)
        character,intent (in   ) :: sector, bc
        integer,  intent (in   ) :: L
        integer,  intent (  out) :: lz, dim
        
        if (bc == 'c') then
            lz = L
        else if (bc == 'o') then
            lz = L - 1
        end if
        if ((sector == '+') .or. (sector == '-')) then
            dim = 2 ** (L - 1)
        else if (sector == 'f') then
            dim = 2 ** L
        end if
    end subroutine set_bounds
    
    integer pure function convert_full (i, sector)
        ! Convert indices from x sector basis to full basis
        character,intent (in) :: sector
        integer,  intent (in) :: i
        
        if (sector == '-') then
            convert_full = ((2 * i) + ieor(poppar(i), 1))
        else if (sector == '+') then
            convert_full = ((2 * i) + poppar(i))
        else if (sector == 'f') then
            convert_full = i
        else
            ! sector not allowed
            convert_full = 0
        end if
    end function convert_full

    pure subroutine convert_sector (i, sector)
        ! Convert indices from x full basis to sector basis
        character,intent (in   ) :: sector
        integer,  intent (inout) :: i
        
        if ((sector == '+') .or. (sector == '-')) then
            i = (i - mod(i, 2)) / 2
        end if
    end subroutine convert_sector

    !!! Define Hamiltonian operators
    
    real(dp) pure function H_mel (i, j, L, h, bc)
        character,intent (in) :: bc
        integer,  intent (in) :: i, j, L
        real(dp), intent (in) :: h
        ! This function is a reference on how to calculate matrix elements
        H_mel = sz_2_mel(i, j, L, bc) + sx_1_mel(i, j, L, h)
    end function H_mel
    
    pure subroutine H_coo (elem, rows, cols, N, L, h, bc, sector)
        ! calculate COO format matrix elements of H in x basis
        ! sector == '+' or '-' calculates in a sector, 'f' does full basis
        ! Note:
        ! when sector == '+' or '-' and bc == 'o', N = L * (2**(L-1))
        ! when sector == '+' or '-' and bc == 'c', N = (L + 1) * (2**(L-1))
        ! when sector == 'f' and bc == 'o', N = L * (2**L)
        ! when sector == 'f' and bc == 'c', N = (L + 1) * (2**L)
        character,intent (in   ) :: sector, bc
        integer,  intent (in   ) :: L, N
        real(dp), intent (in   ) :: h
        real(dp), intent (  out) :: elem(0:(N - 1))
        integer,  intent (  out) :: rows(0:(N - 1)), &
                                    cols(0:(N - 1))
        integer i, j, k, m, p, lz, dim
        k = 0
        call set_bounds(lz, dim, L, bc, sector)

        do i = 0, (dim - 1)
            ! find index in full basis
            p = convert_full(i, sector)
            ! sign flips, sigma x
            elem(k) = sx_1_diag(p, L, h)
            rows(k) = i
            cols(k) = i
            k = k + 1
            ! spiin flips, sigma z
            do j = 0, (lz - 1)
                m = ieor(ieor(p, 2 ** j), 2 ** mod(j + 1, L))
                call convert_sector(m, sector)
                elem(k) = sz_2_val()
                rows(k) = i
                cols(k) = m
                k = k + 1 
            end do
        end do
    end subroutine H_coo
    
    pure subroutine H_vec (v, N, L, h, bc, sector, w)
        ! calculate H|v> in x basis
        ! sector == '+' or '-' calculates in a sector, 'f' does full basis
        ! Note:
        ! when sector == '+' or '-' and bc == 'o', N = L * (2**(L-1))
        ! when sector == '+' or '-' and bc == 'c', N = (L + 1) * (2**(L-1))
        ! when sector == 'f' and bc == 'o', N = L * (2**L)
        ! when sector == 'f' and bc == 'c', N = (L + 1) * (2**L)
        character, intent (in   ) :: bc, sector
        integer,   intent (in   ) :: L, N
        real(dp),  intent (in   ) :: h
        real(dp),  intent (in   ) :: v(0:(N - 1))
        real(dp),  intent (  out) :: w(0:(N - 1))
        integer i, j, m, p, lz, dim
        call set_bounds(lz, dim, L, bc, sector)
        w = 0

        do i = 0, (dim - 1)
            if (v(i) /= 0) then
                ! find index in full basis
                p = convert_full(i, sector)
                ! sign flips from sigma x
                w(i) = w(i) + v(i) * sx_1_diag(p, L, h)
                ! spiin flips from sigma z
                do j = 0, (lz - 1)
                    m = ieor(ieor(p, 2 ** j), 2 ** mod(j + 1, L))
                    call convert_sector(m, sector)
                    w(m) = w(m) + v(i) * sz_2_val()
                end do
            end if
        end do
    end subroutine H_vec
        
    pure subroutine long_H_vec (v, N, L, hx, hz, bc, w)
        ! calculate H|v> in the computational x basis
        ! including a longitudintal 1-site sigma z term at each site
        ! Note: N = 2 ** L, and there is no longer Ising symmetry sectors
        character, intent (in   ) :: bc
        integer,   intent (in   ) :: L, N
        real(dp),  intent (in   ) :: hx, hz
        real(dp),  intent (in   ) :: v(0:(N - 1))
        real(dp),  intent (  out) :: w(0:(N - 1))
        integer i, j, m
        
        call H_vec(v, N, L, hx, bc, 'f', w)
        do i = 0, ((2**L)-1)
            if (v(i) /= 0) then
                ! spin flips from 1-site sigma z
                do j = 0, (L - 1)
                    m = ieor(i, 2 ** j)
                    w(m) = w(m) + v(i) * sz_1_val(hz)
                end do
            end if
        end do
    end subroutine long_H_vec
    
    pure subroutine mbl_H_vec (v, N, L, hx, hz, bc, w)
        ! calculate H|v> in computational x basis using
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
                    ! sign flips from sigma x
                    if (btest(i, j)) then
                        w(i) = w(i) - v(i) * sx_1_val(hx(j))
                    else
                        w(i) = w(i) + v(i) * sx_1_val(hx(j))
                    end if
                    ! spin flips from 1-site sigma z
                    m = ieor(i, 2 ** j)
                    w(m) = w(m) + v(i) * sz_1_val(hz(j))
                    ! spin flips from 2-site sigma z
                    if ((bc == 'o') .and. (j == (L - 1))) cycle
                    m = ieor(m, 2 ** mod(j + 1, L))
                    w(m) = w(m) + v(i) * sz_2_val()
                end do
            end if
        end do
    end subroutine mbl_H_vec
    
end module tfim_x
