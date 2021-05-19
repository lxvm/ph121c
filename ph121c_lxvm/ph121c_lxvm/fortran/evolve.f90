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
        get_threads, &
        time_ev, &
        do_time_evx, &
        do_time_evy, &
        do_time_evz
        
contains

    function get_threads() result(nt)
        integer :: nt
        nt = 0
        !$omp parallel
        nt = omp_get_num_threads()
        !$omp end parallel
    end function get_threads
    
    subroutine time_ev (L, k, Nelm, Nstp, which, Cevecs, Tevals, values)
        ! Cevecs should have an eigenvector in each row, scaled by the coef
        ! Tevals should be equal to exp(-i * evals * dt) for time stepping
        character,  intent (in   ) :: which
        integer,    intent (in   ) :: L, k, Nelm, Nstp
        complex(dp),intent (in   ) :: Tevals(0:(Nelm - 1))
        complex(dp),intent (inout) :: Cevecs(0:(Nelm - 1), 0:(Nelm - 1))
        real(dp),   intent (  out) :: values(Nstp)
        operator: select case (which)
            case ('x')
                call do_time_evx(L, k, Nelm, Nstp, Cevecs, Tevals, values)
            case ('y')
                call do_time_evy(L, k, Nelm, Nstp, Cevecs, Tevals, values)
            case ('z')
                call do_time_evz(L, k, Nelm, Nstp, Cevecs, Tevals, values)
            case default
                stop 'operator not supported: only Pauli x, y, z'
        end select operator
    end subroutine time_ev
    
    subroutine do_time_evx (L, k, Nelm, Nstp, Cevecs, Tevals, values)
        ! do the time evolution for Pauli z
        integer,    intent (in   ) :: L, k, Nelm, Nstp
        complex(dp),intent (in   ) :: Tevals(0:(Nelm - 1))
        complex(dp),intent (inout) :: Cevecs(0:(Nelm - 1), 0:(Nelm - 1))
        real(dp),   intent (  out) :: values(Nstp)
        complex(dp) partial, lookup(0:(Nelm - 1))
        integer i, n, x
        ! k is the site at which we want to calculate the operator
        x = 2 ** k
        !$omp parallel private(partial)
        do n = 1, Nstp
            ! partial sum computed on each thread
            partial = 0
            ! Marginalize out the energy basis, leaving the physical index
            lookup = sum(Cevecs, dim=1)
            !$omp do
            do i = 0, ((2 ** L) - 1)
                ! Find the contribution to the sum
                partial = partial + lookup(i) * conjg(lookup(ieor(i, x)))
                ! Meanwhile, update time step for next loop
                Cevecs(:, i) = Cevecs(:, i) * Tevals
            end do
            !$omp end do
            !$omp atomic
            values(n) = values(n) + real(partial)
            !$omp end atomic
        end do
        !$omp end parallel
    end subroutine do_time_evx
    
    subroutine do_time_evy (L, k, Nelm, Nstp, Cevecs, Tevals, values)
        ! do the time evolution for Pauli y
        integer,    intent (in   ) :: L, k, Nelm, Nstp
        complex(dp),intent (in   ) :: Tevals(0:(Nelm - 1))
        complex(dp),intent (inout) :: Cevecs(0:(Nelm - 1), 0:(Nelm - 1))
        real(dp),   intent (  out) :: values(Nstp)
        complex(dp) partial, lookup(0:(Nelm - 1))
        integer i, n, x
        ! k is the site at which we want to calculate the operator
        x = 2 ** k
        !$omp parallel private(partial)
        do n = 1, Nstp
            ! partial sum computed on each thread
            partial = 0
            ! Marginalize out the energy basis, leaving the physical index
            lookup = sum(Cevecs, dim=1)
            !$omp do
            do i = 0, ((2 ** L) - 1)
                ! Find the contribution to the sum
                partial = partial + lookup(i) * conjg(lookup(ieor(i, x))) &
                    * (0, -1.0) * (1 - 2 * poppar(iand(i, x)) - 1)
                ! Meanwhile, update time step for next loop
                Cevecs(:, i) = Cevecs(:, i) * Tevals
            end do
            !$omp end do
            !$omp atomic
            values(n) = values(n) + real(partial)
            !$omp end atomic
        end do
        !$omp end parallel
    end subroutine do_time_evy
    
    subroutine do_time_evz (L, k, Nelm, Nstp, Cevecs, Tevals, values)
        ! do the time evolution for Pauli z
        integer,    intent (in   ) :: L, k, Nelm, Nstp
        complex(dp),intent (in   ) :: Tevals(0:(Nelm - 1))
        complex(dp),intent (inout) :: Cevecs(0:(Nelm - 1), 0:(Nelm - 1))
        real(dp),   intent (  out) :: values(Nstp)
        complex(dp) partial, lookup(0:(Nelm - 1))
        integer i, n, x
        ! k is the site at which we want to calculate the operator
        x = 2 ** k
        !$omp parallel private(partial)
        do n = 1, Nstp
            ! partial sum computed on each thread
            partial = 0
            ! Marginalize out the energy basis, leaving the physical index
            lookup = sum(Cevecs, dim=1)
            !$omp do
            do i = 0, ((2 ** L) - 1)
                ! Find the contribution to the sum
                partial = partial + lookup(i) * conjg(lookup(i)) &
                    * (1 - 2 * poppar(iand(i, x)))
                ! Meanwhile, update time step for next loop
                Cevecs(:, i) = Cevecs(:, i) * Tevals
            end do
            !$omp end do
            !$omp atomic
            values(n) = values(n) + real(partial)
            !$omp end atomic
        end do
        !$omp end parallel
    end subroutine do_time_evz
    
end module evolve