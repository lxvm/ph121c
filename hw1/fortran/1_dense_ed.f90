program dense_ed
    use lapack95
    use tfim_dense
    implicit none

    integer, dimension (3), parameter     :: L = (/ 8, 10, 12 /)
    real,    dimension (7), parameter     :: h = (/ 0.0, 0.1, 0.7, 1.0, 7.0, 10.0, 100.0 /)
    real,    dimension (2**L(size(L)), 2*size(h))     :: evals
    real,    dimension (2**L(size(L)), 2**L(size(L))) :: Ham
    real time_beg, time_end
    integer i, j, k, info

    open (1, file='data/dense_8_10_12.dat')
    do i = 1, size(L)
        evals = 0
        Ham   = 0
    
        ! Calculate eigenvalues
        do j = 1, size(h)
            ! open bc
            call cpu_time(time_beg)
            call H_open_vec(L(i), h(j), Ham(:2**L(i), :2**L(i)))
            call syev(Ham(:2**L(i), :2**L(i)), evals(:2**L(i), 2*j-1), info=info)
            call cpu_time(time_end)
            print *, time_end - time_beg, 'seconds to run L=', L(i), 'with h=', h(j), 'open'
            if (info /= 0) print *, 'did not converge'
            ! closed bc
            call cpu_time(time_beg)
            call H_closed_vec(L(i), h(j), Ham(:2**L(i), :2**L(i)))
            call syev(Ham(:2**L(i), :2**L(i)), evals(:2**L(i), 2*j), info=info)
            call cpu_time(time_end)
            print *, time_end - time_beg, 'seconds to run L=', L(i), 'with h=', h(j), 'closed'
            if (info /= 0) print *, 'did not converge'
        end do

        ! Write ground state energies to disk
        write (1, *) 'L,   h,            open_E_gs,           closed_E_gs'
        do j = 1, size(h)
            write (1, '(i2, " ")', advance='no') L(i)
            write (1, '(f10.5, " ")', advance='no') h(j)
            write (1, '(f25.15, " ")', advance='no') evals(1, 2*j-1)
            write (1, '(f25.15, " ")', advance='no') evals(1, 2*j)
            write (1, '(a)')
        end do
        write (1, '(a)')
        write (1, '(a)')
    end do
    close (1)

end program dense_ed
