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
        time_ev, get_threads
        
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
        complex(dp) partial, tmp
        integer i, n, x
        ! k is the site we want to calculate the operator at
        x = 2 ** k
        
        !$omp parallel private(partial, tmp)
        do n = 1, Nstp
            partial = 0
            !$omp do
            do i = 0, ((2 ** L) - 1)
                if (which == 'x') then
                ! Sigma x expectation value
                    partial = partial &
                        + sum(Cevecs(:, i)) * conjg(sum(Cevecs(:, ieor(i, x))))
                else if (which == 'y') then
                ! Sigma y expectation value
                    partial = partial &
                        + (0, -1.0) * (1 - 2 * poppar(iand(i, x))) &
                        * sum(Cevecs(:, i)) * conjg(sum(Cevecs(:, ieor(i, x))))
                else if (which == 'z') then
                    ! Sigma z expectation value
                    tmp = sum(Cevecs(:, i))
                    partial = partial &
                        + tmp * conjg(tmp) * (1 - 2 * poppar(iand(i, x)))
                end if
            end do
            !$omp end do
            !$omp atomic
            values(n) = values(n) + real(partial)
            !$omp end atomic
            ! update for next time step
            if (n < Nstp) then
                !$omp do
                do i = 0, (Nelm - 1)
                    Cevecs(:, i) = Cevecs(:, i) * Tevals
                end do
                !$omp end do
            end if
        end do
        !$omp end parallel
    end subroutine time_ev
    
end module evolve