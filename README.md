# Ph 121c

## Python environment

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

You could also probably get this to work without Intel Python, as the only
consequence is that it probably won't be as fast.
My code probably will only depend on numpy, scipy, matplotlib, pandas, pillow,
and hdf5, and so you could just install my package into a virtual
environment with those and install any additional packages that are needed.
Since I am using `conda` only for Intel Python, you would probably be better
off by installing this into a Python `venv` which is less complicated.

## Python package

I made a package to collect my Python code for this project.
If you would like to test the code, you might want to use this package with an 
editable installation.

### Testing

There are tests, such as the module `ph121c_lxvm.tests.tfim` that can be
run at the command line via commands such as 
```
> python -m ph121c_lxvm.tests.tfim
> python -m unittest ph121c_lxvm.tests.tfim.test_hashable_naming
```

### Package status
- Needs argument specifications in all docstrings
- Most of the tfim tasks have an associated test -- keep it up!

### Fortran integration

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
[Best practices reference](https://www.fortran90.org/src/best-practices.html#interfacing-with-python).
I would recommend using the best practices, however after I got started with
`f2py`, it was too late to go back.
You can't exactly reuse existing Fortran code with `f2py`.
For example, functions need to be rewritten as subroutines, where the intent
of the arguments then gets translated into a Python function's return value.
You also cannot use allocatable arrays, the only flexibility is if you use a
function with a C equivalent in the array size declaration (such as pow()) but
you may have to write your own Fortran wrapper (so pass an array size argument).
On the other hand, Cython requires a lot of setup and understanding
how to interface Fortran to C and C to Python.

I can't assure you that my Fortran code is really portable, but hopefully it
works if you have an Intel compiler.

Update: I've also fixed the Fortran code so that it compiles with `gfortran`
version 10.2 (by accident, because it happened to be installed with R).
The Intel compiler has a number of extensions

#### Brilliant

[LFortran](https://lfortran.org/) is an under-development Fortran kernel for
Jupyter.
It's not good enough because it doesn't have enough intrinsics to be useful,
but it is cool and I think should be mentioned because it might make learning
Fortran more fun (at the risk of not being complete).
[Another kernel](https://github.com/ZedThree/jupyter-fortran-kernel).

### Misc

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
