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
    
    implicit none
    
    private
    public &
        time_ev
        
contains

    pure subroutine time_ev (L, k, Nelm, Nstp, which, Cevecs, Tevals, values)
        ! Cevecs should have an eigenvector in each row, scaled by the coef
        ! Tevals should be equal to exp(-i * evals * dt) for time stepping
        character,  intent (in   ) :: which
        integer,    intent (in   ) :: L, k, Nelm, Nstp
        complex(dp),intent (inout) :: Cevecs(0:(Nelm - 1), 0:(Nelm - 1)), &
                                      Tevals(0:(Nelm - 1))
        real(dp),intent (  out) :: values(Nstp)
        complex(dp) Oij, tmp
        integer i, n, x
        ! k is the site we want to calculate the operator at
        x = 2 ** k
        
        if (which == 'x') then
            do n = 1, Nstp
                ! Sigma x expectation value
                do i = 0, ((2 ** L) - 1)
                    values(n) = values(n) &
                        + sum(Cevecs(:, i)) * conjg(sum(Cevecs(:, ieor(i, x))))
                end do
                ! update for next time step
                if (n < Nstp) then
                    do i = 0, (Nelm - 1)
                        Cevecs(:, i) = Cevecs(:, i) * Tevals
                    end do
                end if
            end do
        else if (which == 'y') then
            do n = 1, Nstp
                ! Sigma y expectation value
                do i = 0, ((2 ** L) - 1)
                    values(n) = values(n) &
                        + sum(Cevecs(:, i)) * conjg(sum(Cevecs(:, ieor(i, x))))&
                        * (0, -1.0) * (1 - 2 * poppar(iand(i, x)))
                end do
                ! update for next time step
                if (n < Nstp) then
                    do i = 0, (Nelm - 1)
                        Cevecs(:, i) = Cevecs(:, i) * Tevals
                    end do
                end if
            end do
        else if (which == 'z') then
            do n = 1, Nstp
                ! Sigma z expectation value
                 do i = 0, ((2 ** L) - 1)
                    tmp = sum(Cevecs(:, i))
                    values(n) = values(n) &
                        + tmp * conjg(tmp) &
                        * (1 - 2 * poppar(iand(i, x)))
                    ! update for next time step
                    if (n < Nstp) then
                        Cevecs(:, i) = Cevecs(:, i) * Tevals
                    end if
                end do
            end do
        else
            values = 0
        end if        
    end subroutine time_ev
    
end module evolve