# Based on:
# https://github.com/scivision/f2py-examples
# Something similar here:
# https://gist.github.com/johntut/1d8edea1fd0f5f1c057c

import setuptools
import site

from numpy.distutils.core import setup, Extension

site.ENABLE_USER_SITE = True

setup(
    ext_modules=[
        Extension(
            name='ph121c_lxvm_fortran_tfim',
            sources=[
                'ph121c_lxvm/fortran/tfim.f90',
            ],
        ),
    ],
)