# Ph 121c

I invite anyone to use this software so long as they follow the Caltech Honor
Code: 

"No member of the Caltech community shall take unfair advantage of any other member of the Caltech community."

If you find any bugs, I encourage you to report an issue, or even better, to fix
it a make a pull request.

## Summary

This is a package of code I wrote to do assignments for a course on computation
in quantum-many body systems. Specifically 1D chains of spins. Herein are
functions that:
- Generate TFIM Hamiltonians in sparse and matrix-vector multiplication formats
- Generate an interface to a tensor manipulation framework for MPS simulations
- Some functions to measure observables and change bases or permute indices

## Python 

### Environment

Since I am using the Intel distribution for Python, you can take the following 
steps to (hopefully) reproduce my environment exactly:

- Install the Intel oneapi basekit
- Using `conda` packaged with the basekit, use the `environment.yml` file to
create a virtual environment with the exact packages
- Install the local package within the virtual environment, i.e. with:
`> cd ph121c_lxvm; python setup.py develop`

Advanced note: the installation builds a Fortran module using whatever compiler
is available on your machine (i.e. `gfortran`, `ifort`), so if you have no
Fortran compiler, the code won't build.
You can also override the default compiler by appending
`build_ext --fcompiler=<gnu95, intelem, ...>` to the argument list.
Equivalently, add `[build_ext] \n fcompiler = <...>` to `setup.cfg`.
Look for more help with `python setup.py build_ext --help[-fcompiler]`.

I believe that Intel Python is also available in the `intel` channel on `conda`.
Note this is specific to my Python code and if I use Fortran anywhere then
a Fortran compiler is necessary, and I am using those provided by Intel.

**You could also probably get this to work without Intel Python.**
My code probably will only depend on numpy, scipy, matplotlib, pandas, pillow,
and h5py, and so you could just install my package into a virtual
environment with those and install any additional packages that are needed.
Since I am using `conda` only for Intel Python, you would probably be better
off by installing this into a Python `venv` which is less complicated.

### Package

I made a package to collect my Python code for this project.
If you would like to test the code, you might want to use this package with an 
editable installation.

#### Testing

There are tests, such as the module `ph121c_lxvm.tests.models` that can be
run at the command line via commands such as 
```
> python -m ph121c_lxvm.tests.models
> python -m unittest ph121c_lxvm.tests.models.tfim_test_case
```

#### Status
- Needs argument specifications in all docstrings
- Most of the models and tensor tasks have an associated test -- keep it up!

## Fortran 

### Learning

Don't be afraid to learn Fortran!
But also don't expect to use Fortran for anything: It's strengths currently lie
in rapid and parallel numerics (aka FORmula TRANslation), though there is a
growing standard library for other tasks (however, incomparable to Python).

As a language, it is statically and strongly typed, meaning that there are fewer
gotcha's because every data type needs a type declaration, and every procedure
needs an interface: these traits make Fortran code verbose, but readable.
To learn some Fortran from a book, take a look at a free section of an ebook
[here](https://www.manning.com/books/exploring-modern-fortran-basics).
For historical reasons, you may also be interested in perusing the 1956 manual
[here](http://bitsavers.informatik.uni-stuttgart.de/pdf/ibm/704/704_FortranProgRefMan_Oct56.pdf).

#### Community

Disclaimer: The Fortran community dates to 1954 and continues to represent the
demographics of the scientific computing community. These demographics are
disappointing with respect to gender equity and racial diversity, and currently,
there appear to be no efforts on behalf of the Fortran community to do outreach
in these areas. You might compare this to actions and statements from the
[Julia](https://julialang.org/diversity/) and
[NumFOCUS](https://numfocus.org/programs/diversity-inclusion) communities.
I wonder with whom I should invest my time and these issues matter to me, so I
want to bring attention to this for anyone else who is thinking about it too.
That being said, the Fortran community is small, and all contributors are
welcome according to the Fortran language webpage. This old language has been
seeing a bit of revival since 2020, and new work by the community promises many
good things.

These are the places one can get started with Fortran:
- [_The_ community hub for Fortran](https://fortran-lang.org/)
- [Fortran programming practices](https://www.fortran90.org/)
- [Fortran Wiki](http://fortranwiki.org/fortran/show/Fortran+Wiki)

Look out for these people/blogs/projects in the Fortran community:
- [Dr. Fortran](https://stevelionel.com/drfortran/)
- [Ondřej Čertík](https://ondrejcertik.com/blog/)
- [Milan Curcic](https://milancurcic.com/)
- [Fortran book blog](https://medium.com/modern-fortran)
- [Fortran projects](https://github.com/rabbiabram/awesome-fortran),
notably missing [ARPACK](https://www.caam.rice.edu/software/ARPACK/),
and probably [HDF5](https://www.hdfgroup.org/solutions/hdf5)
- [Fortran proposals](https://github.com/j3-fortran/fortran_proposals)
- [F/OSS](https://github.com/Fortran-FOSS-Programmers)

#### The present is in the the near future (for Fortran)

[LFortran](https://lfortran.org/) is an under-development Fortran compiler
designed to be compiled just-in-time with LLVM technology.
This makes it available for interactive use: LFortran has a Jupyter kernel!
It's not good enough because it doesn't have enough intrinsics to be useful,
but it is cool and I think should be mentioned because it might make learning
Fortran more fun (at the risk of not being complete).

[`fpm`](https://github.com/fortran-lang/fpm) is a Fortran package manager that
aims to make using and building Fortran code similar to Rust's `cargo` -- an
experience much closer to Python's `pip` than the age-old process of building
libraries using `make` via directly linking libraries and such.

[`stdlib`](https://github.com/fortran-lang/stdlib), a growing standard library
for Fortran aimed at providing basic utilities not implement intrinsically by
Fortran compilers (i.e. not accepted by the Fortran Standards Committee).

### Integration with Python

The module `numpy.f2py` allows for integration of Fortran code into Python.
This can be very beneficial, especially for very loopy tasks that are purely
computational, such as building Hamiltonians in COO format.
The actual integration can be a little tedious, but the numpy docs and this
[example package](https://github.com/scivision/f2py-examples)
provide a lot of help to do so.
Ultimately, fast code, even with `numpy`, requires memory allocation.
For simple computational tasks, the activation energy for using Fortran
isn't too large.
For using Fortran-specific modules that don't have a `numpy` or `scipy`
equivalent, I imagine that there begin the typical difficulties one
experiences in Fortran related to compiling and linking libraries.

There are many microscopic bugs that this introduces: one about certain parts
of the Fortran code needing to be understood by C (such as '\*\*'), or 
how to actually distribute the Fortran code as a Python package,
which for me led to an [error](https://github.com/dmlc/xgboost/issues/820).
That only occured when using `pip install -e .` but `python setup.py develop`
works.

The recommended way to interface Fortran and Python is actually via Cython.
What `f2py` does in the background anyway is to port the Fortran to C.
I would recommend using these [best practices
](https://www.fortran90.org/src/best-practices.html#interfacing-with-python),
however after I got started with `f2py`, it was too late to go back.
You can't exactly reuse existing Fortran code with `f2py`.
For example, you cannot use allocatable arrays, and also using functions in 
array size declarations is allowed only when they are understood by C (such as 
pow()) so you may have to write your own Fortran wrapper if you prefer that
over passing the array size as an argument.
On the other hand, Cython requires a lot of setup and understanding
how to interface Fortran to C and C to Python.

I can't assure you that my Fortran code is really portable, but hopefully it
works if you have an Intel compiler.

Update: I've also fixed the Fortran code so that it compiles with `gfortran`
version 10.2 (by accident, because it happened to be installed with R).
The Intel compiler has a number of extensions that are incompatible with other
compilers.

## Misc

Note: I tried to setup this package using `pyscaffold`, which seems pretty 
useful, but I ran into an error with `setuptools_scm` not liking where my
git repository was located.
[Reference this issue](https://github.com/pypa/setuptools_scm/issues/278).
I didn't want to figure this out, so I just went for the bare minimum by
copying another Python package I wrote for something else.

Note: Dashes and underscores are not interchangable in Python.
If a package name contains a dash, the actual directory names must have
underscores instead.
This gave rise to many a ModuleNotFoundError for me.
