! This is my answer to the symmetry question
! Here I use my lanczos implementation
! on restricted symmetry sectors of the tfim Hamiltonian

program symmetry
    use tfim_dense
    use solvers_dense
    implicit none

    real,    dimension (3), parameter :: h = (/ 0.3, 1.0, 1.7 /)
    integer, dimension (13), parameter :: L = (/ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25 /)
    ! For each L, how many eigenvalues to compute
    integer, dimension (13), parameter :: m = (/ 2, 8, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 /)
    real time_beg, time_end
    integer i, j

    open (1, file='data/symmetry.dat', access='append')
    write (1, *) 'L     h bc k s             eigenvalue'
    do i = 1, size(L)
        do j = 1, size(h)
            call cpu_time(time_beg)
            call lanczos_psector(L(i), h(j), 'N', 'c')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.2, g0)', 'symmetry+++ ', &
                time_end - time_beg, ' seconds to run L=', L(i), ' with h=', h(j), ' closed'
            call cpu_time(time_beg)
            call lanczos_msector(L(i), h(j), 'N', 'c')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.2, g0)', 'symmetry--- ', &
                time_end - time_beg, ' seconds to run L=', L(i), ' with h=', h(j), ' closed'
        end do
    end do
    write (1, '(a)')
    write (1, '(a)')    
    close (1)

contains

    function wrap_Hxp_vec_closed (v) result (w)
        real, dimension (:), intent (in) :: v
        real, dimension (size(v))        :: w
        w = Hxp_vec_closed(L(i), h(j), v)
    end function wrap_Hxp_vec_closed


    function wrap_Hxm_vec_closed (v) result (w)
        real, dimension (:), intent (in) :: v
        real, dimension (size(v))        :: w
        w = Hxm_vec_closed(L(i), h(j), v)
    end function wrap_Hxm_vec_closed


    subroutine lanczos_psector (L, h, job, bc)
        ! Compute just the extremal eigenpairs of the Hamiltonian
        ! using the Lanczos algorithm on the restricted symmetry sectors
        integer,   intent (in) :: L
        real,      intent (in) :: h
        character, intent (in) :: job
        character, intent (in) :: bc
        ! local variables
        real, dimension (m(i))           :: evals
        real, dimension (2**(L-1), m(i)) :: evecs
        integer k

        if (bc == 'c') then
            call my_lanczos(wrap_Hxp_vec_closed, L-1, m(i), evals, evecs, job)
        else
            stop 'unknown bc'
        end if
        ! Report 4 lowest eigenvalues
        do k = 1, 4
            write (1, '(i2, " ")', advance='no') L
            write (1, '(f5.2, " ")', advance='no') h
            write (1, '(a2, " ")', advance='no') bc
            write (1, '(i1, " ")', advance='no') k
            write (1, '(a1, " ")', advance='no') 'p'
            write (1, '(f25.15)') evals(k)
        end do
    end subroutine lanczos_psector


    subroutine lanczos_msector (L, h, job, bc)
        ! Compute just the extremal eigenpairs of the Hamiltonian
        ! using the Lanczos algorithm on the restricted symmetry sectors
        integer,   intent (in) :: L
        real,      intent (in) :: h
        character, intent (in) :: job
        character, intent (in) :: bc
        ! local variables
        real, dimension (m(i))       :: evals
        real, dimension (2**(L-1), m(i)) :: evecs
        integer k

        if (bc == 'c') then
            call my_lanczos(wrap_Hxm_vec_closed, L-1, m(i), evals, evecs, job)
        else
            stop 'unknown bc'
        end if
        ! Report 4 lowest eigenvalues
        do k = 1, 4
            write (1, '(i2, " ")', advance='no') L
            write (1, '(f5.2, " ")', advance='no') h
            write (1, '(a2, " ")', advance='no') bc
            write (1, '(i1, " ")', advance='no') k
            write (1, '(a1, " ")', advance='no') 'm'
            write (1, '(f25.15)') evals(k)
        end do
    end subroutine lanczos_msector

end program symmetry
