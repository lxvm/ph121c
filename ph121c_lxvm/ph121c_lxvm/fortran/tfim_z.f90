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
        H_z_mel, &
        H_z_coo, &
        H_z_vec, &
        long_H_z_vec,&
        mbl_H_z_vec, &
        sigma_z1_z_val, &
        sigma_z1_z_diag,&
        sigma_z1_z_mel, &
        sigma_z2_z_val, &
        sigma_z2_z_diag,&
        sigma_z2_z_mel, &
        sigma_x1_z_val, &
        sigma_x1_z_mel

    ! Suffix 'val' is a scalar & implies use in a do loop over a physical index
    ! Suffix 'mel' means calculation of a matrix element & is loopless
    ! Suffix 'diag' means calculation of a diagonal matrix element
    ! Suffix 'coo' means calculation of a matrix in sparse coordinate format
    ! Suffix 'vec' means calculation of a matrix-vector product
    ! Modifier '_i' means evaluation in the basis of coordinate i
    ! Modifier '_ik_' stands for a k-site/k-local operator at coordinate i 

contains
    
    !!! Define Pauli operators
    
    real(dp) pure function sigma_z1_z_val (h)
        ! 1-site sigma z contribution to Hamiltonian in z basis
        real(dp), intent (in) :: h
        sigma_z1_z_val = -h
    end function sigma_z1_z_val

    real(dp) pure function sigma_z1_z_diag (i, L, h)
        ! 1-site sigma z contribution to Hamiltonian in z basis
        integer,  intent (in) :: i, L
        real(dp), intent (in) :: h
        sigma_z1_z_diag = sigma_z1_z_val(h) * (L - 2 * popcnt(i))
    end function sigma_z1_z_diag
    
    real(dp) pure function sigma_z1_z_mel (i, j, L, h)
        ! 1-site sigma z contribution to Hamiltonian in z basis
        integer,  intent (in) :: i, j, L
        real(dp), intent (in) :: h
        
        if (i == j) then
            sigma_z1_z_mel = sigma_z1_z_diag(i, L, h)
        else
            sigma_z1_z_mel = 0
        end if
    end function sigma_z1_z_mel
    
    real(dp) pure function sigma_z2_z_val ()
        sigma_z2_z_val = -1
    end function sigma_z2_z_val

    real(dp) pure function sigma_z2_z_diag (i, L, bc)
        ! 2-site sigma z in z basis contribution on diagonal
        ! for each bit in system:
        ! if same as neighbor: +1
        ! else: -1
        character,intent (in) :: bc
        integer,  intent (in) :: i, L

        if (bc == 'c') then
            ! in closed system, do a cyclic shift
            sigma_z2_z_diag = sigma_z2_z_val() &
                * (L - 2 * popcnt(ieor(ishftc(i, 1, L), i)))
        else if (bc == 'o') then
            ! in open system, shift non-cyclically
            sigma_z2_z_diag = sigma_z2_z_val() &
                * (L - 1 - 2 * popcnt(ieor(ishft(i, -1), ibclr(i, L - 1))))
        else
            ! invalid boundary condition
            sigma_z2_z_diag = 0
        end if
    end function sigma_z2_z_diag
    
    real(dp) pure function sigma_z2_z_mel (i, j, L, bc)
        ! 2-site sigma z in z basis matrix element
        ! for each bit in system:
        ! if same as neighbor: +1
        ! else: -1
        character,intent (in) :: bc
        integer,  intent (in) :: i, j, L

        if (i == j) then
            sigma_z2_z_mel = sigma_z2_z_diag(i, L, bc)
        else
            sigma_z2_z_mel = 0
        end if
    end function sigma_z2_z_mel
    
    real(dp) pure function sigma_x1_z_val (h)
        real(dp), intent (in) :: h
        sigma_x1_z_val = -h
    end function sigma_x1_z_val
    
    real(dp) pure function sigma_x1_z_mel (i, j, h)
        ! Calculate matrix element of 1-site sigma x in z basis
        integer,  intent (in) :: i, j
        real(dp), intent (in) :: h
        
        if (popcnt(ieor(i, j)) == 1) then
            ! sigma x contribution for Hamming distance of 1
            sigma_x1_z_mel = sigma_x1_z_val(h)
        else 
            sigma_x1_z_mel = 0
        end if
    end function sigma_x1_z_mel

    !!! Define Hamiltonian operators

    real(dp) pure function H_z_mel (i, j, L, h, bc)
        character,intent (in) :: bc
        integer,  intent (in) :: i, j, L
        real(dp), intent (in) :: h
        ! This function is a reference for how to calculate matrix elements.
        H_z_mel = sigma_z2_z_mel(i, j, L, bc) + sigma_x1_z_mel(i, j, h)
    end function H_z_mel

    pure subroutine H_z_coo (elem, rows, cols, N, L, h, bc)
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
                elem(k) = sigma_x1_z_val(h)
                rows(k) = i
                cols(k) = ieor(i, 2 ** j)
                k = k + 1
            end do
            ! sign flips, sigma z
            elem(k) = sigma_z2_z_diag(i, L, bc)
            rows(k) = i
            cols(k) = i
            k = k + 1
        end do
    end subroutine H_z_coo
        
    pure subroutine H_z_vec (v, N, L, h, bc, w)
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
                    w(m) = w(m) + v(i) * sigma_x1_z_val(h)
                end do
                ! sign flips from sigma z
                w(i) = w(i) + v(i) * sigma_z2_z_diag(i, L, bc)
            end if
        end do
    end subroutine H_z_vec
    
    pure subroutine long_H_z_vec (v, N, L, hx, hz, bc, w)
        ! calculate H|v> in the computational z basis
        ! including a longitudintal 1-site sigma z term at each site
        ! Note: N = 2 ** L
        character, intent (in   ) :: bc
        integer,   intent (in   ) :: L, N
        real(dp),  intent (in   ) :: hx, hz
        real(dp),  intent (in   ) :: v(0:(N - 1))
        real(dp),  intent (  out) :: w(0:(N - 1))
        integer i
        
        call H_z_vec(v, N, L, hx, bc, w)
        do i = 0, ((2**L)-1)
            if (v(i) /= 0) then
                ! sign flips from sigma z
                w(i) = w(i) + v(i) * sigma_z1_z_diag(i, L, hz)
            end if
        end do
    end subroutine long_H_z_vec

    pure subroutine mbl_H_z_vec (v, N, L, hx, hz, bc, w)
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
                    w(m) = w(m) + v(i) * sigma_x1_z_val(hx(j))
                    ! sign flips from 1-site sigma z
                    w(i) = w(i) + v(i) * sigma_z1_z_val(hz(j))
                end do
                ! sign flips from 2-site sigma z
                w(i) = w(i) + v(i) * sigma_z2_z_diag(i, L, bc)
            end if
        end do
    end subroutine mbl_H_z_vec
    

end module tfim_z


module tfim_x
    ! Transverse-Field Ising Model Hamiltonians (and their derivates) in x basis
    use types
    
    implicit none
    
    private
    public &
        H_x_mel, &
        H_x_coo, &
        H_x_vec, &
        long_H_x_vec,&
        mbl_H_x_vec, &      
        sigma_z1_x_val, &
        sigma_z1_x_mel, &
        sigma_z2_x_val, &
        sigma_z2_x_mel, &
        sigma_x1_x_val, &
        sigma_x1_x_diag,&
        sigma_x1_x_mel, &
        set_x_bounds,   &
        convert_full_x, &
        convert_sector_x

contains
    
    !!! Define Pauli operators
    
    real(dp) pure function sigma_z1_x_val (h)
        real(dp), intent (in) :: h
        sigma_z1_x_val = -h
    end function sigma_z1_x_val

    real(dp) pure function sigma_z1_x_mel (i, j, h)
        ! 1-site sigma z contribution to Hamiltonian in z basis
        ! equal to sigma_x1_z_mel
        integer,  intent (in) :: i, j
        real(dp), intent (in) :: h
        
        if (popcnt(ieor(i, j)) == 1) then
            ! sigma x contribution for Hamming distance of 1
            sigma_z1_x_mel = sigma_z1_x_val(h)
        else 
            sigma_z1_x_mel = 0
        end if
    end function sigma_z1_x_mel
    
    real(dp) pure function sigma_z2_x_val ()
        sigma_z2_x_val = -1
    end function sigma_z2_x_val

    real(dp) pure function sigma_z2_x_mel (i, j, L, bc)
        ! 2-site sigma z contribution for Hamming distance of 2
        ! but only for adjacent bits (if closed, match ends too)
        character,intent (in) :: bc
        integer,  intent (in) :: i, j, L
        
        if (popcnt(ieor(i, j)) == 2) then
            if (bc == 'c') then
                ! popcnt(ieor(ishftc(ieor(i, j), 1, L), ieor(i, j))) == 1
                if (ieor(ishftc(ieor(i, j), 1, L), ieor(i, j)) /= 0) then
                    sigma_z2_x_mel = sigma_z2_x_val() 
                else
                    sigma_z2_x_mel = 0
                end if
            else if (bc == 'o') then
                ! popcnt(ieor(ishft(ieor(i, j), 1), ieor(i, j))) == 1
                if (ieor(ishft(ieor(i, j), 1), ieor(i, j)) /= 0) then
                    sigma_z2_x_mel = sigma_z2_x_val()
                else
                    sigma_z2_x_mel = 0
                end if
            else
                ! not valid bc
                sigma_z2_x_mel = 0
            end if
        else
            sigma_z2_x_mel = 0
        end if
    end function sigma_z2_x_mel

    real(dp) pure function sigma_x1_x_val (h)
        ! 1-site sigma x coefficient
        real(dp), intent (in) :: h
        sigma_x1_x_val = -h
    end function sigma_x1_x_val

    real(dp) pure function sigma_x1_x_diag (i, L, h)
        ! 1-site sigma x contribution where for each spin
        ! add 1 if bit position at position is zero (spin down, + sector)
        ! else subtract 1
        integer,  intent (in) :: i, L
        real(dp), intent (in) :: h
        sigma_x1_x_diag = sigma_x1_x_val(h) * (L - 2 * popcnt(i))
    end function sigma_x1_x_diag
    
    real(dp) pure function sigma_x1_x_mel (i, j, L, h)
        ! 1-site sigma x contribution where for each spin
        ! add 1 if bit position at position is zero (spin down, + sector)
        ! else subtract 1
        integer,  intent (in) :: i, j, L
        real(dp), intent (in) :: h
        
        if (i == j) then
            sigma_x1_x_mel = sigma_x1_x_diag(i, L, h)
        else
            sigma_x1_x_mel = 0
        end if
    end function sigma_x1_x_mel
    
    !!! Define utilities to optimizing array size and convert sector indices
    
    pure subroutine set_x_bounds (lz, dim, L, bc, sector)
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
    end subroutine set_x_bounds
    
    integer pure function convert_full_x (i, sector)
        ! Convert indices from x sector basis to full basis
        character,intent (in) :: sector
        integer,  intent (in) :: i
        
        if (sector == '-') then
            convert_full_x = ((2 * i) + ieor(poppar(i), 1))
        else if (sector == '+') then
            convert_full_x = ((2 * i) + poppar(i))
        else if (sector == 'f') then
            convert_full_x = i
        else
            ! sector not allowed
            convert_full_x = 0
        end if
    end function convert_full_x

    pure subroutine convert_sector_x (i, sector)
        ! Convert indices from x full basis to sector basis
        character,intent (in   ) :: sector
        integer,  intent (inout) :: i
        
        if ((sector == '+') .or. (sector == '-')) then
            i = (i - mod(i, 2)) / 2
        end if
    end subroutine convert_sector_x

    !!! Define Hamiltonian operators
    
    real(dp) pure function H_x_mel (i, j, L, h, bc)
        character,intent (in) :: bc
        integer,  intent (in) :: i, j, L
        real(dp), intent (in) :: h
        ! This function is a reference on how to calculate matrix elements
        H_x_mel = sigma_z2_x_mel(i, j, L, bc) + sigma_x1_x_mel(i, j, L, h)
    end function H_x_mel
    
    pure subroutine H_x_coo (elem, rows, cols, N, L, h, bc, sector)
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
        call set_x_bounds(lz, dim, L, bc, sector)

        do i = 0, (dim - 1)
            ! find index in full basis
            p = convert_full_x(i, sector)
            ! sign flips, sigma x
            elem(k) = sigma_x1_x_diag(p, L, h)
            rows(k) = i
            cols(k) = i
            k = k + 1
            ! spiin flips, sigma z
            do j = 0, (lz - 1)
                m = ieor(ieor(p, 2 ** j), 2 ** mod(j + 1, L))
                call convert_sector_x(m, sector)
                elem(k) = sigma_z2_x_val()
                rows(k) = i
                cols(k) = m
                k = k + 1 
            end do
        end do
    end subroutine H_x_coo
    
    pure subroutine H_x_vec (v, N, L, h, bc, sector, w)
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
        call set_x_bounds(lz, dim, L, bc, sector)
        w = 0

        do i = 0, (dim - 1)
            if (v(i) /= 0) then
                ! find index in full basis
                p = convert_full_x(i, sector)
                ! sign flips from sigma x
                w(i) = w(i) + v(i) * sigma_x1_x_diag(p, L, h)
                ! spiin flips from sigma z
                do j = 0, (lz - 1)
                    m = ieor(ieor(p, 2 ** j), 2 ** mod(j + 1, L))
                    call convert_sector_x(m, sector)
                    w(m) = w(m) + v(i) * sigma_z2_x_val()
                end do
            end if
        end do
    end subroutine H_x_vec
        
    pure subroutine long_H_x_vec (v, N, L, hx, hz, bc, w)
        ! calculate H|v> in the computational x basis
        ! including a longitudintal 1-site sigma z term at each site
        ! Note: N = 2 ** L, and there is no longer Ising symmetry sectors
        character, intent (in   ) :: bc
        integer,   intent (in   ) :: L, N
        real(dp),  intent (in   ) :: hx, hz
        real(dp),  intent (in   ) :: v(0:(N - 1))
        real(dp),  intent (  out) :: w(0:(N - 1))
        integer i, j, m
        
        call H_x_vec(v, N, L, hx, bc, 'f', w)
        do i = 0, ((2**L)-1)
            if (v(i) /= 0) then
                ! spin flips from 1-site sigma z
                do j = 0, (L - 1)
                    m = ieor(i, 2 ** j)
                    w(m) = w(m) + v(i) * sigma_z1_x_val(hz)
                end do
            end if
        end do
    end subroutine long_H_x_vec
    
    pure subroutine mbl_H_x_vec (v, N, L, hx, hz, bc, w)
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
                    w(i) = w(i) + v(i) * sigma_x1_x_val(hx(j))
                    ! spin flips from 1-site sigma z
                    m = ieor(i, 2 ** j)
                    w(m) = w(m) + v(i) * sigma_z1_x_val(hz(j))
                    ! spin flips from 2-site sigma z
                    if ((bc == 'o') .and. (j == (L - 1))) cycle
                    m = ieor(m, 2 ** mod(j + 1, L))
                    w(m) = w(m) + v(i) * sigma_z2_x_val()
                end do
            end if
        end do
    end subroutine mbl_H_x_vec
    
end module tfim_x


module scars
    ! A spin 1/2 Hamiltonian that is a toy model of Rydberg atoms
    use types
    
    implicit none
    
    private
    public &
        H_z_vec
    
contains

    pure subroutine H_z_vec (v, N, L, O, w)
        ! compute H|v> in the z basis
        ! Note: N = 2 ** L
        integer,   intent (in   ) :: L, N
        real(dp),  intent (in   ) :: O
        real(dp),  intent (in   ) :: v(0:(N - 1))
        real(dp),  intent (  out) :: w(0:(N - 1))
        w = 0
    end subroutine
    
end module scars