program sparse
    use tfim
    use sparse_solvers
    implicit none

    integer, dimension (1), parameter     :: LL = (/ 8  /)
    real,    dimension (1), parameter     :: hh = (/ 1.0 /)
    ! For each L, how many eigenpairs to compute
    integer, dimension (1), parameter     :: mm = (/ 1 /)
    real,    dimension (2**LL(size(LL)))  :: evals
    real,    dimension (2**LL(size(LL)), mm(size(mm))) :: evecs
    real,    dimension (2**LL(size(LL)), 2**LL(size(LL))) :: Ham
    integer i, j, L, m
    real    h

    do i = 1, size(LL)
        L = LL(i)
        m = mm(i)
        do j = 1, size(hh)
            h = hh(j)
            call my_lanczos(wrap_H_vec_closed, L, m, evals(:2**L), evecs(:2**L, :m))
        end do
    end do

    print *, 'eigenvalues of ham'
    print *, evals(:10)

    print *, 'ham'
    call H_dense_closed(L, h, Ham)
    call print_matrix(Ham(:10, :10))

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
    
end program sparse
