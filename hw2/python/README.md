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
$ . /opt/intel/oneapi/setvars.sh
# Activate oneapi conda environment
$ conda activate base
# Prioritize the packages already installed with basekit
$ conda config --add channels file:///opt/intel/oneapi/conda_channel
$ conda env export > environment.yml    # this saves environment to a file
# Create a new environment from environment.yml file
$ conda env create -n <env_name>        # Default: --file environment.yml
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

Install `imagemagick` 
(it comes with the command line tools `convert` and `identify`).
Use `imagemagick` to convert your pictures to grayscale: see this
[stackoverflow post](https://stackoverflow.com/questions/13317753/convert-rgb-to-grayscale-in-imagemagick-command-line).
Also, convert all images to `png` format for use with matplotlib`.

In Python, options to read an image into an array:
- [`matplotlib`](https://matplotlib.org/): 
`matplotlib.image` module supports the `png` format with `imread()`
- [`pillow`](https://python-pillow.org/): 
active development of Python Image Library (PIL)
- [`opencv`](https://docs.opencv.org/master/): 
computer vision algorithms

More words on reading images into numpy arrays in this
[stackoverflow post](https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array).

## More help

- [Intel forums](https://community.intel.com/)
- [Intel forums: SDKs](https://community.intel.com/t5/Software-Development-SDKs-and/ct-p/software-dev-sdk-libraries)
- [Intel forums: Intel Distribution for Python](https://community.intel.com/t5/Intel-Distribution-for-Python/bd-p/distribution-python)
- [`conda` docs](https://conda.io/projects/conda/en/latest/index.html)
- [`pillow` docs](https://pillow.readthedocs.io/en/latest/index.html)
- [`scipy` docs](https://www.scipy.org/docs.html)
