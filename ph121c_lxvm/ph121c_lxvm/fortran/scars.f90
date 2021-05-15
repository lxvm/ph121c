module types
    ! https://stackoverflow.com/questions/12523524/f2py-specifying-real-precision-in-fortran-when-interfacing-with-python
    implicit none
    
    integer, parameter :: dp=kind(0.d0)
    
end module types


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