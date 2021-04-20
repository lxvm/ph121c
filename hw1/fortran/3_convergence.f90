! This is my answer to the convergence question
! Here I use my lanczos implementation for larger systems
! and calculate the ground state energy per site

program convergence
    use tfim_dense
    use solvers_dense
    implicit none

    real,    dimension (2), parameter :: h = (/ 0.3, 1.7 /)
    integer, dimension (9), parameter :: L = (/ 8, 10, 12, 14, 16, 18, 20, 22, 24 /)
    ! For each L, how many eigenvalues to compute
    integer, dimension (9), parameter :: m = (/ 32, 32, 32, 32, 32, 32, 32, 32, 32 /)
    real time_beg, time_end
    integer i, j

    open (1, file='data/convergence.dat', access='append')
    write (1, *) 'L     h bc         ground_state_E/L'
    do i = 1, size(L)
        do j = 1, size(h)
            call cpu_time(time_beg)
            call lanczos_extremes(L(i), h(j), 'N', 'c')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.2, g0)', 'convergence ', &
                time_end - time_beg, ' seconds to run L=', L(i), ' with h=', h(j), ' closed'
            call cpu_time(time_beg)
            call lanczos_extremes(L(i), h(j), 'N', 'o')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.2, g0)', 'convergence ', &
                time_end - time_beg, ' seconds to run L=', L(i), ' with h=', h(j), ' open'
        end do
    end do
    write (1, '(a)')
    write (1, '(a)')    
    close (1)

contains

    function wrap_H_vec_closed (v) result (w)
        real, dimension (:), intent (in) :: v
        real, dimension (size(v))        :: w
        w = H_vec_closed(L(i), h(j), v)
    end function wrap_H_vec_closed
    
    function wrap_H_vec_open (v) result (w)
        real, dimension (:), intent (in) :: v
        real, dimension (size(v))        :: w
        w = H_vec_open(L(i), h(j), v)
    end function wrap_H_vec_open

    subroutine lanczos_extremes (L, h, job, bc)
        ! Compute just the extremal eigenvalues of the Hamiltonian
        ! using the Lanczos algorithm
        integer,   intent (in) :: L
        real,      intent (in) :: h
        character, intent (in) :: job
        character, intent (in) :: bc
        ! local variables
        real, dimension (m(i))       :: evals
        real, dimension (2**L, m(i)) :: evecs

        if (bc == 'o') then
            call my_lanczos(wrap_H_vec_open, L, m(i), evals, evecs, job)
        else if (bc == 'c') then
            call my_lanczos(wrap_H_vec_closed, L, m(i), evals, evecs, job)
        else
            stop 'unknown bc'
        end if
        write (1, '(i2, " ")', advance='no') L
        write (1, '(f5.2, " ")', advance='no') h
        write (1, '(a2, " ")', advance='no') bc
        write (1, '(f25.15)') evals(1)/L
    end subroutine lanczos_extremes    

end program convergence
