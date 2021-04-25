# Ph 121c

## Python environment

I made a package to collect my Python code for this project.
If you would like to test the code, you might want to use this package with an 
editable installation.
Since I am using the Intel distribution for Python, you can take the following 
steps to reproduce my environment:

- Install the Intel oneapi basekit
- Using `conda` packaged with the basekit, use the `environment.yml` file to
create a virtual environment with the exact packages
- Install the local package within the virtual environment, i.e. with:
`$ python -m pip install -e ph121c-lxvm`

I believe that Intel Python is also available in the `intel` channel on `conda`.
Note this is specific to my Python code and if I use Fortran anywhere then
a Fortran compiler is necessary, and I am using those provided by Intel.

You could also probably get this to work without Intel Python, as the only
consequence is that it probably won't be as fast.
My code probably will only depend on numpy, scipy, matplotlib and maybe a
few other packages, so you could just install my package into a virtual
environment with those and install any additional packages that are needed.
Since I am using `conda` only for Intel Python, you would probably be better
off by installing this into a Python `venv` which is less complicated.

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
