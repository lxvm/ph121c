module types
    ! https://stackoverflow.com/questions/12523524/f2py-specifying-real-precision-in-fortran-when-interfacing-with-python
    implicit none
    
    private
    public &
        dp
    
    integer, parameter :: dp=kind(0.d0)
    
end module types


module evolve
    ! Time-evolve expectation values of operators sx, sy, and sz
    use types
    use omp_lib
    ! learning openmp from:
    ! https://curc.readthedocs.io/en/latest/programming/OpenMP-Fortran.html
    ! Lecture notes from MS141 in 2020 by I-Te Lu
    
    implicit none
    
    private
    public &
        set_threads, &
        get_threads, &
        Pauli_ev
        
contains

    subroutine set_threads (n)
        integer, intent (in) :: n
        call omp_set_num_threads(n)
    end subroutine set_threads
    
    integer function get_threads ()
        !$omp parallel
        get_threads = omp_get_num_threads()
        !$omp end parallel
    end function get_threads

    subroutine Pauli_ev (L, k, Nelm, Nstp, which, Cevecs, Tevals, values)
        ! do the time evolution for Pauli x
        character,  intent (in   ) :: which
        integer,    intent (in   ) :: L, k, Nelm, Nstp
        complex(dp),intent (in   ) :: Tevals(0:(Nelm - 1))
        complex(dp),intent (inout) :: Cevecs(0:(Nelm - 1), 0:(Nelm - 1))
        real(dp),   intent (  out) :: values(Nstp)
        complex(dp) partial, lookup(0:(Nelm - 1))
        integer i, n, x
        ! k is the site at which we want to calculate the operator
        x = 2 ** k
        do n = 1, Nstp
            !$omp parallel do private (i)
            do i = 0, ((2 ** L) - 1)
                lookup(i) = sum(Cevecs(:, i))
                Cevecs(:, i) = Cevecs(:, i) * Tevals
            end do
            !$omp end parallel do
            partial = 0
            operator: select case (which)
                case ('x')
                    !$omp parallel do private (i) reduction (+:partial)
                    do i = 0, ((2 ** L) - 1)
                        partial = partial + lookup(i) * conjg(lookup(ieor(i, x)))
                    end do
                    !$omp end parallel do
                case ('y')
                    !$omp parallel do private (i) reduction (+:partial)
                    do i = 0, ((2 ** L) - 1)
                        partial = partial + lookup(i) * conjg(lookup(ieor(i, x))) &
                            * (0, -1.0) * (1 - 2 * poppar(iand(i, x)) - 1)
                    end do
                    !$omp end parallel do
                case ('z')
                    !$omp parallel do private (i) reduction (+:partial)
                    do i = 0, ((2 ** L) - 1)
                        partial = partial + lookup(i) * conjg(lookup(i)) &
                            * (1 - 2 * poppar(iand(i, x)))
                    end do
                    !$omp end parallel do
                case default
                    stop 'chosen operator not implemented'
            end select operator
            values(n) = real(partial)
        end do
    end subroutine Pauli_ev

end module evolve