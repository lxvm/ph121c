module types
    ! https://stackoverflow.com/questions/12523524/f2py-specifying-real-precision-in-fortran-when-interfacing-with-python
    implicit none
    
    private
    public &
        dp
    
    integer, parameter :: dp=kind(0.d0)
    
end module types


module constants

    use types
    
    implicit none
    
    private
    public &
        Im
    
    complex(dp), parameter :: Im = (0, 1)
    
end module constants

module scars
    ! A spin 1/2 Hamiltonian that is a toy model of Rydberg atoms in z basis
    use types
    use constants
    
    implicit none
    
    private
    public &
        H_vec, &
        sz_k_par, &
        sz_k_diag
    
contains

    !!! Define Pauli operator

    recursive pure function sz_k_par (i, L, k, m) result (z)
        ! Returns an integer where bit j has the k-site parity
        ! of bit positions j - m to j - m + k - 1 (bits numbered right to left)
        integer, intent (in) :: i, L, k, m
        integer z
        if (k > 1) then
            z = ieor(sz_k_par(i, L, k - 1, m), ishftc(i, 1 + m - k, L))
        else
            z = i
        end if
    end function sz_k_par

    integer pure function sz_k_diag (i, L, k)
        ! k-site sigma z contribution to Hamiltonian in z basis
        integer,  intent (in) :: i, L, k
        sz_k_diag = L - (2 * popcnt(sz_k_par(i, L, k, 0)))
    end function sz_k_diag
    
    !!! Define Hamiltonian operator
    
    pure subroutine H_vec (v, N, L, O, w)
        ! compute H|v> in the z basis
        ! The Hamiltonian is translation invariant
        ! Note: N = 2 ** L
        integer,    intent (in   ) :: L, N
        real(dp),   intent (in   ) :: O
        real(dp),intent (in   ) :: v(0:(N - 1))
        real(dp),intent (  out) :: w(0:(N - 1))
        integer i, j, m
        w = 0
        
        do i = 0, ((2 ** L) - 1)
            if (v(i) /= 0) then
                ! Diagonal (0 Hamming distance) 1-site - 3-site sigma z terms
                w(i) = w(i) + v(i) * (sz_k_diag(i, L, 1) - sz_k_diag(i, L, 3)) / 4
                ! Off-diagonal terms (3-site and transverse 1-site sigma x)
                do j = 0, (L - 1)                  
                    ! Terms of Hamming-distance 1 at position j (transverse sx)
                    m = ieor(i, 2 ** j)
                    w(m) = w(m) + v(i) * (O / 2)
                    ! Terms of Hamming distance 2 at positions j, j + 1
                    m = ieor(m, 2 ** mod(j + 1, L))
                    ! Set value
                    w(m) = w(m) + v(i) * (-0.25) &
                        ! sx sx sz + sy sy sz terms (1 - 1 = 0)
                        * (2 * poppar(iand(i, (2 ** j) + (2 ** mod(j + 1, L))))) &
                        ! sign of sz
                        * (1 - 2 * poppar(iand(i, 2 ** mod(j + 2, L))))
                end do
            end if
        end do
    end subroutine H_vec
    
end module scars