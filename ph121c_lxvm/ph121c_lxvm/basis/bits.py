"""Ordinary bit operations.
"""


def swap (n, i, j):
    """Swap bit positions i, j in n."""
    # Do nothing if bits i, j are the same
    # xor flip if bits i, j are not the same
    matched_mask = ((n >> i) & 1) == ((n >> j) & 1) 
    return (n * matched_mask + (n ^ ((1 << i) + (1 << j))) * (matched_mask ^ 1))