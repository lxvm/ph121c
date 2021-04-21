! This is my additional answer to the convergence question
! Here I use my lanczos implementation for larger systems
! and calculate the ground state energy per site

program trick
    use tfim_dense
    use solvers_dense
    implicit none

    real,    dimension (2), parameter :: h = (/ 0.3, 1.7 /)
    integer, dimension (9), parameter :: L = (/ 8, 10, 12, 14, 16, 18, 20, 22, 24 /)
    ! For each L, how many eigenvalues to compute
    integer, dimension (9), parameter :: m = (/ 32, 32, 32, 32, 32, 32, 32, 32, 32 /)
    real time_beg, time_end
    integer i, j

    open (1, file='data/trick.dat', access='append')
    write (1, *) 'L1 L2    h bc       ground_state_Ediff'
    do i = 1, (size(L)-1)
        do j = 1, size(h)
            call cpu_time(time_beg)
            call trick_bc(i, h(j), 'N', 'o')
            call cpu_time(time_end)
            print '(a20, f10.6, g0, i2, g0, f5.2, g0)', 'trick ', &
                time_end - time_beg, ' seconds to run L=', L(i), ' with h=', h(j), ' open'
        end do
    end do
    write (1, '(a)')
    write (1, '(a)')    
    close (1)

contains

    function wrap_H_vec_open_small (v) result (w)
        real, dimension (:), intent (in) :: v
        real, dimension (size(v))        :: w
        w = H_vec_closed(L(i), h(j), v)
    end function wrap_H_vec_open_small
    
    function wrap_H_vec_open_large (v) result (w)
        real, dimension (:), intent (in) :: v
        real, dimension (size(v))        :: w
        w = H_vec_open(L(i+1), h(j), v)
    end function wrap_H_vec_open_large

    subroutine trick_bc (i, h, job, bc)
        ! Compute just the extremal eigenvalues of the Hamiltonian
        ! using the Lanczos algorithm
        integer,   intent (in) :: i
        real,      intent (in) :: h
        character, intent (in) :: job
        character, intent (in) :: bc
        ! local variables
        real                         :: eval
        real, dimension (m(i))       :: evals
        real, dimension (2**L(i+1), m(i)) :: evecs

        if (bc == 'o') then
            call my_lanczos(wrap_H_vec_open_small, L(i), m(i), evals(:2**L(i)), evecs, job)
            eval = evals(1)
            evals= 0
            call my_lanczos(wrap_H_vec_open_large, L(i+1), m(i+1), evals, evecs, job)
        else
            stop 'unknown bc'
        end if
        write (1, '(i2, " ")', advance='no') L(i)
        write (1, '(i2, " ")', advance='no') L(i+1)
        write (1, '(f5.2, " ")', advance='no') h
        write (1, '(a2, " ")', advance='no') bc
        write (1, '(f25.15)') (evals(1)/L(i+1) - eval / L(i)) / 2
    end subroutine trick_bc

end program trick
