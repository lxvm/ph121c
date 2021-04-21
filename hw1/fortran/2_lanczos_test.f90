! This is my answer to the sparse ED question
! Here I verify my implementation of the
! Lanczos algorithm against the dense diagonalization
! routine and then continue to test the implementation
! for larger systems

program lanczos_test
    use lapack95
    use tfim_dense
    use solvers_dense
    implicit none

    real,    dimension (2), parameter :: h = (/ 0.3, 1.7 /)
    integer, dimension (1), parameter :: L = (/ 8 /)
    ! For each L, how many eigenvalues to compute
    integer, dimension (7), parameter :: m = (/ 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8 /)
    real time_beg, time_end
    integer i, j

    open (1, file='data/lanczos_test.dat', access='append')
    write (1, *) 'L     h bc    k keff    m method                eigenvalue'
    do i = 1, size(L)
        do j = 1, size(h)
            call cpu_time(time_beg)
            call test_lanczos_dense(L(i), h(j), 'N', 'c')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.2, g0)', 'lanczos_test ', &
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
    
    subroutine test_lanczos_dense (L, h, job, bc)
        ! Compute all the eigenvalues using Lanczos and dense
        ! methods and report their values
        integer,   intent (in) :: L
        real,      intent (in) :: h
        character, intent (in) :: job
        character, intent (in) :: bc
        ! local variables
        real, dimension (2**L)       :: evals_dense, evals_lanczos 
        real, dimension (2**L, 2**L) :: Ham
        integer j, k

        if (bc /= 'c') stop 'bc not implemented'
        ! Diagonalize Ham
        call H_closed_vec(L, h, Ham)
        call syev(Ham, evals_dense, job)
        ! print eigenvalues
        do k = 1, size(evals_dense)
            write (1, '(i2, " ")', advance='no') L
            write (1, '(f5.2, " ")', advance='no') h
            write (1, '(a2, " ")', advance='no') bc
            write (1, '(i4, " ")', advance='no') k
            write (1, '(i4, " ")', advance='no') k
            write (1, '(i4, " ")', advance='no') 2**L
            write (1, '(a6, " ")', advance='no') 'dense'
            write (1, '(f25.15)') evals_dense(k)
        end do
        do j = 1, size(m)
            call my_lanczos(wrap_H_vec_closed, L, m(j), evals_lanczos, Ham, job)
            do k = 1, m(j)
                write (1, '(i2, " ")', advance='no') L
                write (1, '(f5.2, " ")', advance='no') h
                write (1, '(a2, " ")', advance='no') bc
                write (1, '(i4, " ")', advance='no') k
                write (1, '(i4, " ")', advance='no') ((k - 1) * (2**L - 1) / (m(j) - 1)) + 1 
                write (1, '(i4, " ")', advance='no') m(j)
                write (1, '(a6, " ")', advance='no') 'lanczo'
                write (1, '(f25.15, " ")') evals_lanczos(k)
            end do
        end do
    end subroutine test_lanczos_dense

end program lanczos_test
