! This is part 2 of my answer to the transition question
! Here I use my lanczos implementation for largest system
! and calculate the excitation gaps of low-energy states

program fidelity
    use tfim_dense
    use solvers_dense
    implicit none

    ! Use this value of dh to calculate fidelities
    real, parameter              :: dh = 0.01
    real,    dimension (9), parameter :: h = (/ 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4 /)
    integer, dimension (1), parameter :: L = (/ 24 /)
    ! For each L, how many eigenvalues to compute
    integer, dimension (1), parameter :: m = (/ 32 /)
    real time_beg, time_end
    integer i, j

    open (1, file='data/fidelity.dat', access='append')
    write (1, *) 'L     h bc                 fidelity'
    do i = 1, size(L)
        do j = 1, size(h)
            call cpu_time(time_beg)
            call lanczos_fidelity(L(i), h(j), 'V', 'c')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.2, g0)', 'fidelity ', &
                time_end - time_beg, ' seconds to run L=', L(i), ' with h=', h(j), ' closed'
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

    function wrap_H_vec_closed_dh (v) result (w)
        real, dimension (:), intent (in) :: v
        real, dimension (size(v))        :: w
        w = H_vec_closed(L(i), h(j)+dh, v)
    end function wrap_H_vec_closed_dh

    subroutine lanczos_fidelity (L, h, job, bc)
        ! Compute the fidelity of the ground state for small deviations dh
        integer,   intent (in) :: L
        real,      intent (in) :: h
        character, intent (in) :: job
        character, intent (in) :: bc
        ! local variables
        real, dimension (m(i))       :: evals
        real, dimension (2**L, m(i), 2) :: evecs

        if (bc == 'c') then
            call my_lanczos(wrap_H_vec_closed, L, m(i), evals, evecs(:, :, 1), job)
            call my_lanczos(wrap_H_vec_closed_dh, L, m(i), evals, evecs(:, :, 2), job)
        else
            stop 'unknown bc'
        end if
        ! print the overlap in the ground states
        write (1, '(i2, " ")', advance='no') L
        write (1, '(f5.2, " ")', advance='no') h
        write (1, '(a2, " ")', advance='no') bc
        write (1, '(f25.15)') ip(evecs(:, 1, 1), evecs(:, 1, 2))
    end subroutine lanczos_fidelity


    real function ip (v, w)
        real, intent (in) :: v(:), w(:)
        ip = abs(sum(v * w) / ((sum(v * v) * sum(w * w)) ** 0.5))
    end function ip
    
end program fidelity
