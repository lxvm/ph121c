! This is my answer to the sparse ED question
! Here I verify my implementation of the
! Lanczos algorithm against the dense diagonalization
! routine and then continue to test the implementation
! for larger systems

program sparse_ed
    use lapack95
    use tfim_dense
    use solvers_dense
    implicit none

    real,    dimension (2), parameter :: h = (/ 0.3, 1.7 /)
    integer, dimension (7), parameter :: L = (/ 8, 10, 12, 14, 16, 18, 20 /)
    ! For each L, how many eigenvalues to compute
    integer, dimension (7), parameter :: m = (/ 2**8, 2**10, 32, 32, 32, 32, 32 /)
    real time_beg, time_end
    integer i, j

    open (1, file='data/sparse_ed.dat', access='append')
    write (1, *) 'L     h bc    k        eigenvalue_lanczos          eigenvalue_dense'
    do i = 1, 2
        do j = 1, size(h)
            call cpu_time(time_beg)
            call verify_lanczos_dense(L(i), h(j), 'N', 'o')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.1, g0)', 'sparse_ed_verify ', &
                time_end - time_beg, ' seconds to run L=', L(i), ' with h=', h(j), ' open'
            call cpu_time(time_beg)
            call verify_lanczos_dense(L(i), h(j), 'N', 'c')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.1, g0)', 'sparse_ed_verify ', &
                time_end - time_beg, ' seconds to run L=', L(i), ' with h=', h(j), ' closed'
        end do
    end do
    write (1, '(a)')
    write (1, '(a)')
    write (1, *) 'L     h bc    k        eigenvalue_lanczos'
    do i = 3, size(L)
        do j = 1, size(h)
            call cpu_time(time_beg)
            call lanczos_extremes(L(i), h(j), 'N', 'c')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.1, g0)', 'sparse_ed_extreme ', &
                time_end - time_beg, ' seconds to run L=', L(i), ' with h=', h(j), ' closed'
            call cpu_time(time_beg)
            call lanczos_extremes(L(i), h(j), 'N', 'o')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.1, g0)', 'sparse_ed_extreme ', &
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

    subroutine verify_lanczos_dense (L, h, job, bc)
        ! Compute all the eigenvalues using Lanczos and dense
        ! methods and report their values
        integer,   intent (in) :: L
        real,      intent (in) :: h
        character, intent (in) :: job
        character, intent (in) :: bc
        ! local variables
        real, dimension (2**L)       :: evals_dense, evals_lanczos 
        real, dimension (2**L, 2**L) :: Ham
        integer k

        ! run lanczos and prepare Ham for dense ED
        if (bc == 'o') then
            call my_lanczos(wrap_H_vec_open, L, m(i), evals_lanczos, Ham, job)
            call H_open_vec(L, h, Ham)
        else if (bc == 'c') then
            call my_lanczos(wrap_H_vec_closed, L, m(i), evals_lanczos, Ham, job)
            call H_closed_vec(L, h, Ham)
        else
            stop 'unknown bc'
        end if
        call syev(Ham, evals_dense, job)
        ! print eigenvalues
        do k = 1, m(i)
            write (1, '(i2, " ")', advance='no') L
            write (1, '(f5.1, " ")', advance='no') h
            write (1, '(a2, " ")', advance='no') bc
            write (1, '(i4, " ")', advance='no') k
            write (1, '(f25.15, " ")', advance='no') evals_lanczos(k)
            write (1, '(f25.15)') evals_dense(k)
        end do
    end subroutine verify_lanczos_dense


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
        integer k

        if (bc == 'o') then
            call my_lanczos(wrap_H_vec_open, L, m(i), evals, evecs, job)
        else if (bc == 'c') then
            call my_lanczos(wrap_H_vec_closed, L, m(i), evals, evecs, job)
        else
            stop 'unknown bc'
        end if
        do k = 1, m(i)
            write (1, '(i2, " ")', advance='no') L
            write (1, '(f5.1, " ")', advance='no') h
            write (1, '(a2, " ")', advance='no') bc
            write (1, '(i4, " ")', advance='no') k
            write (1, '(f25.15)') evals(k)
        end do
    end subroutine lanczos_extremes    

end program sparse_ed
