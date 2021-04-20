! This is my answer to the dense ED question
! Here I compute ground state energies using
! dense diagonalization

program dense_ed
    use lapack95
    use tfim_dense
    implicit none

    integer, dimension (4), parameter :: L = (/ 8, 10, 12, 14 /)
    real,    dimension (9), parameter :: h = (/ 0.1, 0.18, 0.32, 0.56, 1.0, 1.78, 3.16, 5.62, 10.0 /)
    real time_beg, time_end
    integer i, j

    open (1, file='data/dense_ed.dat', access='append')
    write (1, *) 'L     h bc         lowest_eigenvalue'
    do i = 1, size(L)
        ! Calculate eigenvalues
        do j = 1, size(h)
            ! open bc
            call cpu_time(time_beg)
            call dense_ev(L(i), h(j), 'N', 'o')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.2, g0)', 'dense_ed ', &
                time_end - time_beg, ' seconds to run L=', L(i), ' with h=', h(j), ' open'
            ! closed bc
            call cpu_time(time_beg)
            call dense_ev(L(i), h(j), 'N', 'c')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.2, g0)', 'dense_ed ', &
                time_end - time_beg, ' seconds to run L=', L(i), ' with h=', h(j), ' closed'
        end do
    end do
    write (1, '(a)')
    write (1, '(a)')
    close (1)
    
contains

    subroutine dense_ev (L, h, job, bc)
        ! Create and diagonalize a dense Hamiltonian
        ! and save the ground state eigenvalues
        integer,   intent (in) :: L
        real,      intent (in) :: h
        character, intent (in) :: job
        ! if job = 'N' then eigenvalues
        ! if job = 'V' then eigenpair
        character, intent (in) :: bc
        ! bc must be 'o' for open or 'c' for closed 
        ! local variables
        real, dimension (2**L)       :: evals
        real, dimension (2**L, 2**L) :: evecs

        ! First fill evecs with Hamiltonian
        if (bc == 'o') then
            call H_open_vec(L, h,  evecs)
        else if (bc == 'c') then
            call H_closed_vec(L, h, evecs)
        else
            stop 'unknown bc'
        end if
        ! diagonalize
        call syev(evecs, evals, job)
        ! print ground state energy
        write (1, '(i2, " ")', advance='no') L
        write (1, '(f5.2, " ")', advance='no') h
        write (1, '(a2, " ")', advance='no') bc
        write (1, '(f25.15)') evals(1)
    end subroutine dense_ev

end program dense_ed
