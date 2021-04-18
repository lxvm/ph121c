program sparse_ed
    use tfim_dense
    use solvers_dense
    implicit none

    integer, dimension (1), parameter     :: LL = (/ 23 /)
    real,    dimension (1), parameter     :: hh = (/ 1.0 /)
    ! For each L, how many eigenpairs to compute
    integer, dimension (1), parameter     :: mm = (/ 2**5 /)
    real,    dimension (   mm(size(mm)))  :: evals
    real,    dimension (2**LL(size(LL)), mm(size(mm))) :: evecs
    integer i, j, L, m
    real    h

    do i = 1, size(LL)
        L = LL(i)
        m = mm(i)
        do j = 1, size(hh)
            h = hh(j)
            print *, 'L=', L, 'h=', h
            call my_lanczos(wrap_H_vec_closed, L, m, evals(:m), evecs(:(2**L), :m))
        end do
    end do

!    print *, 'eigenvectors'
!    call print_matrix(evecs)

    print *, 'eigenvalues of ham'
    print *, evals(:m)
    
contains

    function wrap_H_vec_closed (v) result (w)
        real, dimension (:), intent (in) :: v
        real, dimension (size(v))        :: w
        w = H_vec_closed(L, h, v)
    end function wrap_H_vec_closed
    
    function wrap_H_vec_open (v) result (w)
        real, dimension (:), intent (in) :: v
        real, dimension (size(v))        :: w
        w = H_vec_open(L, h, v)
    end function wrap_H_vec_open
    
end program sparse_ed
