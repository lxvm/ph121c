def dim_mps (i, L, d):
    """Calculate the maximal rank for an svd at bond index i in mps"""
    if i <= L / 2:
        return d ** i
    else:
        return d ** (L - i)

def bond_rank (chi, L, d):
    """Return a function to calculate the bond ranks with a constant truncation."""
    return lambda i: max(1, min(chi, dim_mps(i, L, d)))

def test_valid_scheme (r, L, d):
    """Tests whether a rank function is a valid mps approximation scheme."""
    prev_val = 1
    for i in range(L + 1):
        if not (1 <= r(i) <= min(prev_val, dim_mps(i, L, d))):
            return False
        prev_val = d * r(i)
    return True
