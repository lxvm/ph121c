module types
    ! https://stackoverflow.com/questions/12523524/f2py-specifying-real-precision-in-fortran-when-interfacing-with-python
    implicit none
    
    integer, parameter :: dp=kind(0.d0)
    
end module types

module tfim
    
    use types

    implicit none

    private
    public &
        H_z_mel, &
        H_z_coo, &
        H_z_vec, &
        H_x_mel, &
        H_x_coo, &
        H_x_vec, &
        sigma_z_z, &
        sigma_x_z, &
        sigma_z_x, &
        sigma_x_x, &
        sigma_x_z_val, &
        sigma_z_x_val

contains

    real(dp) pure function sigma_z_z (i, L, bc)
        ! sigma z in z basis contribution on diagonal
        ! for each bit in system:
        ! if same as neighbor: +1
        ! else: -1
        character,intent (in) :: bc
        integer,  intent (in) :: i, L

        if (bc == 'c') then
            ! in closed system, do a cyclic shift
            sigma_z_z = -(L - 2 * popcnt(ieor(ishftc(i, 1, L), i)))
        else if (bc == 'o') then
            ! in open system, shift non-cyclically
            sigma_z_z = -(L - 1 - 2 * popcnt(ieor(ibclr(ishft(i, 1), L), i)))
        else
            ! invalid boundary condition
            sigma_z_z = 0
        end if
    end function sigma_z_z
    
    real(dp) pure function sigma_x_z_val (h)
        real(dp), intent (in) :: h
        
        sigma_x_z_val = -h
    end function sigma_x_z_val
    
    real(dp) pure function sigma_x_z (i, j, h)
        ! Calculate matrix element of sigma x in z basis
        integer,  intent (in) :: i, j
        real(dp), intent (in) :: h
        
        if (popcnt(ieor(i, j)) == 1) then
            ! sigma x contribution for Hamming distance of 1
            sigma_x_z = sigma_x_z_val(h)
        else 
            sigma_x_z = 0
        end if
    end function sigma_x_z

    pure elemental subroutine H_z_mel (elem, i, j, L, h, bc)
        ! In Fortran, this acts elementally on the i, j arrays
        ! In f2py, this isn't recognized so only use this one at a time
        ! So in Fortran, passing i = (/0:(2**L)-1:1/), j = i would return the diagonal
        character,intent (in   ) :: bc
        integer,  intent (inout) :: i, j
        integer,  intent (in   ) :: L
        real(dp), intent (in   ) :: h
        real(dp), intent (  out) :: elem
        ! This function is a reference for how to calculate matrix elements.
        if (i == j) then
            elem = sigma_z_z(i, L, bc)
        else
            elem = sigma_x_z(i, j, h)
        end if
    end subroutine H_z_mel

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
                elem(k) = sigma_x_z_val(h)
                rows(k) = i
                cols(k) = ibchng(i, j)
                k = k + 1
            end do
            ! sign flips, sigma z
            ! test whether adjacent bits match
            elem(k) = sigma_z_z(i, L, bc)
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
        integer i, j
        
        w = 0
        do i = 0, ((2**L)-1)
            if (v(i) /= 0) then
                ! spin flips from sigma x
                do j = 0, (L-1)
                    w(ibchng(i, j)) = w(ibchng(i, j)) + v(i) * sigma_x_z_val(h)
                end do
                ! sign flips from sigma z
                w(i) = sigma_z_z(i, L, bc)
            end if
        end do
    end subroutine H_z_vec
    
    real(dp) pure function sigma_z_x_val ()
        sigma_z_x_val = -1
    end function sigma_z_x_val

    real(dp) pure function sigma_z_x (i, j, L, bc)
        ! sigma z contribution for Hamming distance of 2
        ! but only for adjacent bits (if closed, match ends too)
        character,intent (in) :: bc
        integer,  intent (in) :: i, j, L
        
        if (popcnt(ieor(i, j)) == 2) then
            if (bc == 'c') then
                if (popcnt(ieor(ishftc(ieor(i, j), 1, L), ieor(i, j))) == 1) then
                    sigma_z_x = -1
                else
                    sigma_z_x = 0
                end if
            else if (bc == 'o') then
                if (popcnt(ieor(ishft(ieor(i, j), 1), ieor(i, j))) == 1) then
                    sigma_z_x = -1
                else
                    sigma_z_x = 0
                end if
            else
                ! not valid bc
                sigma_z_x = 0
            end if
        else
            sigma_z_x = 0
        end if
    end function sigma_z_x
    
    real(dp) pure function sigma_x_x (i, L, h)
        ! sigma x contribution where for each spin
        ! add 1 if bit position at position is zero (spin down, + sector)
        ! else subtract 1
        integer,  intent (in) :: i, L
        real(dp), intent (in) :: h
        
        sigma_x_x = -h * (L - 2 * popcnt(i))
    end function sigma_x_x
    
    pure elemental subroutine H_x_mel (elem, i, j, L, h, bc)
        ! In Fortran, this acts elementally on the i, j arrays
        ! In f2py, this isn't recognized so only use this one at a time
        character,intent (in   ) :: bc
        integer,  intent (inout) :: i, j
        integer,  intent (in   ) :: L
        real(dp), intent (in   ) :: h
        real(dp), intent (  out) :: elem
        ! This function is a reference on how to calculate matrix elements
        if (i /= j) then
            elem = sigma_z_x(i, j, L, bc)
        else
            elem = sigma_x_x(i, L, h)
        end if
    end subroutine H_x_mel

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
        k    = 0
        if (bc == 'c') then
            lz = L
        else if (bc == 'o') then
            lz = L - 1
        end if
        if (ior(sector == '+', sector == '-')) then
            dim = 2 ** (L - 1)
        else if (sector == 'f') then
            dim = 2 ** L
        end if

        do i = 0, (dim - 1)
            ! find index in full basis
            if (sector == '-') then
                p = ((2 * i) + ieor(poppar(i), 1))
            else if (sector == '+') then
                p = ((2 * i) + poppar(i))
            else if (sector == 'f') then
                p = i
            end if
            ! sign flips, sigma x
            elem(k) = sigma_x_x(p, L, h)
            rows(k) = i
            cols(k) = i
            k = k + 1
            ! spiin flips, sigma z
            do j = 0, (lz - 1)
                m = ieor(ieor(p, 2 ** j), 2 ** mod(j + 1, L))
                if (ior(sector == '+', sector == '-')) then
                    m = (m - mod(m, 2)) / 2
                end if
                elem(k) = sigma_z_x_val()
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
        if (bc == 'c') then
            lz = L
        else
            lz = L - 1
        end if
        if (ior(sector == '+', sector == '-')) then
            dim = 2 ** (L - 1)
        else if (sector == 'f') then
            dim = 2 ** L
        end if
        w = 0

        do i = 0, (dim - 1)
            if (v(i) /= 0) then
                ! find index in full basis
                if (sector == '-') then
                    p = ((2 * i) + ieor(poppar(i), 1))
                else if (sector == '+') then
                    p = ((2 * i) + poppar(i))
                else if (sector == 'f') then
                    p = i
                end if
                ! sign flips from sigma x
                w(i) = v(i) * sigma_x_x(p, L, h)
                ! spiin flips from sigma z
                do j = 0, (lz - 1)
                    m = ieor(ieor(p, 2 ** j), 2 ** mod(j + 1, L))
                    if (ior(sector == '+', sector == '-')) then
                        m = (m - mod(m, 2)) / 2
                    end if
                    w(m) = w(m) + v(i) * sigma_z_x_val()
                end do
            end if
        end do
    end subroutine H_x_vec

end module tfim
