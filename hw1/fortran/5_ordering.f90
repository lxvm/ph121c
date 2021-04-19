! This is my answer to the ordering question
! Here I use my lanczos implementation to 
! calculate the long-range order of the ground state

program ordering
    use tfim_dense
    use solvers_dense
    implicit none

    real,    dimension (3), parameter :: h = (/ 0.3, 1.0, 1.7 /)
    integer, dimension (7), parameter :: L = (/ 12, 14, 16, 18, 20, 22, 24 /)
    ! For each L, how many eigenvalues to compute
    integer, dimension (7), parameter :: m = (/ 32, 32, 32, 32, 32, 32, 32 /)
    real time_beg, time_end
    integer i, j

    open (1, file='data/ordering.dat', access='append')
    write (1, *) 'L     h bc  r              expectation'
    do i = 1, size(L)
        do j = 1, size(h)
            call cpu_time(time_beg)
            call lanczos_ordering(L(i), h(j), 'V', 'c')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.1, g0)', 'ordering ', &
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


    subroutine lanczos_ordering (L, h, job, bc)
        ! Compute just the extremal eigenpairs of the Hamiltonian
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
        ! find the correlator at all sites and magnetization of the gs
        do k = 0, (L-1)
            write (1, '(i2, " ")', advance='no') L
            write (1, '(f5.1, " ")', advance='no') h
            write (1, '(a2, " ")', advance='no') bc
            write (1, '(i2, " ")', advance='no') k
            write (1, '(f25.15)') corr_sigma_z(0, k, evecs(:, 1))
        end do
        write (1, '(i2, " ")', advance='no') L
        write (1, '(f5.1, " ")', advance='no') h
        write (1, '(a2, " ")', advance='no') bc
        write (1, '(a2, " ")', advance='no') ' m'
        write (1, '(f25.15)') var_magnetization(L, evecs(:, 1))        
    end subroutine lanczos_ordering

end program ordering
