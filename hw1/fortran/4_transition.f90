! This is my answer to the transition question
! Here I use my lanczos implementation for largest system
! and calculate the excitation gaps of low-energy states

program transition
    use tfim_dense
    use solvers_dense
    implicit none

    real,    dimension (9), parameter :: h = (/ 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7 /)
    integer, dimension (1), parameter :: L = (/ 24 /)
    ! For each L, how many eigenvalues to compute
    integer, dimension (1), parameter :: m = (/ 32 /)
    real time_beg, time_end
    integer i, j

    open (1, file='data/transition.dat', access='append')
    write (1, *) 'L     h bc k               eigenvalue'
    do i = 1, size(L)
        do j = 1, size(h)
            call cpu_time(time_beg)
            call lanczo_gs_gap(L(i), h(j), 'N', 'c')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.2, g0)', 'transition ', &
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


    subroutine lanczo_gs_gap (L, h, job, bc)
        ! Compute just the extremal eigenvalues of the Hamiltonian
        ! using the Lanczos algorithm
        integer,   intent (in) :: L
        real,      intent (in) :: h
        character, intent (in) :: job
        character, intent (in) :: bc
        ! local variables
        real, dimension (m(i))       :: evals
        real, dimension (2**L, m(i)) :: evecs
        integer k

        if (bc == 'c') then
            call my_lanczos(wrap_H_vec_closed, L, m(i), evals, evecs, job)
        else
            stop 'unknown bc'
        end if
        ! since lanczos tends to spread eigenvalues uniformly,
        ! I will only print the lowest 4 of 32
        do k = 1, 4
            write (1, '(i2, " ")', advance='no') L
            write (1, '(f5.2, " ")', advance='no') h
            write (1, '(a2, " ")', advance='no') bc
            write (1, '(i1, " ")', advance='no') k
            write (1, '(f25.15)') evals(k)
        end do
    end subroutine lanczo_gs_gap    

end program transition
