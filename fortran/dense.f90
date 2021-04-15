program dense
    use tfim
    implicit none
    real, dimension(2, 2) :: id, sz, sx
    call def_matrices(id, sz, sx)
    call save_matrix(H_dense_open_kron(4, 1.0))
end program dense
