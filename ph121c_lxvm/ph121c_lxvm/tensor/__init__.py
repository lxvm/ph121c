"""Implement elementary tensor network construction and operations.

Currently performs matrix product state (MPS) representations and operations.
A good overview of the topic is here: https://tensornetwork.org/mps/
A review paper on the topic is here: https://arxiv.org/abs/1008.3477

This module defines two classes to do MPS/MPO operations:
mps: instantiation of a matrix product state representation of a vector
    with methods for index contraction and calculating products
mpo: representation of matrix product operators
    
This module defines classes that are essential to the MPS/MPO classes:
index: a collection of items in an enumeration
site: collection of coefficients stored as a multi-indexed matrix
train: collection of sites, usually connected by bonds
"""

__all__ = [
    'utils',
    'indices',
    'site',
    'train',
    'mps',
    'mpo',
]

from .utils import *
from .indices import *
from .site import *
from .train import *
from .mps import *
from .mpo import *