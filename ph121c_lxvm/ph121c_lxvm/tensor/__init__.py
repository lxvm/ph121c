"""Implement elementary tensor network construction and operations.

Currently performs matrix product state (MPS) representations and operations.
A good overview of the topic is available here:
https://tensornetwork.org/mps/

This module defines two classes to do MPS operations:
mps: instantiation of a matrix product state representation of a vector
    with methods for index contraction and calculating products
mpo: representation of matrix product operators
    (basically a numpy array with no notable methods)
"""

__all__ = [
    'mps',
    'mpo',
]

from .utils import *
from .mps import *
from .mpo import *