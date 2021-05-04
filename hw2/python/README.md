# Python

## `conda` environment

Using Intel Distribution for Python, 2 options:
- With Intel basekit installed (including Python, `conda`) (newer packages)
- With any `conda` installed, from Intel channel (2018 versions from Intel)

Links to respective options:
- [First](https://community.Intel.com/t5/Intel-Distribution-for-Python/How-to-install-use-the-Intel-distribution-of-Python-from-my/td-p/1138999)
- [Second](https://software.Intel.com/content/www/us/en/develop/articles/Intel-distribution-for-python-development-environment-setting-for-jupyter-notebook-and.html)

Slight correction to the first if using installation of oneapi basekit:
- Basekit packages are newer/incompatible with Intel's conda channel
- Basekit only comes with `intelpython3_core` installed, though 
`intelpython3_full` is only available from Intel's conda channel
- To use the basekit installation in a new environment, do the following

```
# Activate oneapi environment
> . /opt/intel/oneapi/setvars.sh
# Activate oneapi conda environment
> conda activate base
# Prioritize the packages already installed with basekit
> conda config --add channels file:///opt/intel/oneapi/conda_channel
> conda env export -f environment.yml    # this saves environment to a file
# Create a new environment from environment.yml file
> conda env create -n <env_name>        # Default: --file environment.yml
# Install more packages as necessary, typically by another conda channel or pip
```

Some guidance using `conda`:
- Don't install anything into base environment
(create a new environment for any project).
- When creating environment, try to install everything at same time to resolve
dependencies (can always try again with a new environment).
- Export and import environments with `conda env export` and `conda env create`
- Don't fret about channel priorities: prioritize the channel you want to
use with `conda config --add channels <channel_name>` right before installation
and change it the next time you want to install something.
- Make `conda` projects self-contained
(i.e. don't use other package managers for software).
- Avoid updating packages in an environment to keep project stable
(create new environment if necessary).
- Don't expect `conda` to be smart.
It doesn't want to be configurable.
It doesn't want to keep things up-to-date for you (it is not `apt`).
When in doubt, make a new environment -- this is what conda does best.
- `python3 -m venv` with pip is a much more easy tool to use to manage python
environments, but Intel's oneapi toolkits are not packaged on PyPI.


## Image processing

I quickly discarded the idea of using Fortran for image processing.
After all, most supported image formats were invented after Fortran,
and there most likely aren't Fortran libraries for reading any of these
formats.

We enter the package zoo wherein we stand on the shoulders of giants ...

In Python, options to read an image input/output as array:
- [`pillow`](https://python-pillow.org/): 
active development of Python Image Library (PIL)
- [`skimage`](https://scikit-image.org/):
image processing in Python
- [`imageio`](https://imageio.github.io/):
image reading and writing in Python
- [`opencv`](https://docs.opencv.org/master/): 
computer vision algorithms
- [`matplotlib.image`](https://matplotlib.org/stable/api/image_api.html#module-matplotlib.image): 
has some functionality but is not as capable

Note that almost every package uses `pillow`, so I will too.

## Runtimes

The file `runtimes.txt` contains some of the diagonalization runtimes for 
smaller systems of up to $L=20$.
I am sad that I trying to make larger matrices results in the IPython kernel
crashing, but I can work around this by using scipy LinearOperator instances.
The issue with that however is that the code is slower because it uses my 
single-threaded LinearOperator much more heavily


## More help

- [Intel forums](https://community.intel.com/)
- [Intel forums: SDKs](https://community.intel.com/t5/Software-Development-SDKs-and/ct-p/software-dev-sdk-libraries)
- [Intel forums: Intel Distribution for Python](https://community.intel.com/t5/Intel-Distribution-for-Python/bd-p/distribution-python)
- [`conda` docs](https://conda.io/projects/conda/en/latest/index.html)
- [`pillow` docs](https://pillow.readthedocs.io/en/latest/index.html)
- [`scipy` docs](https://www.scipy.org/docs.html)
