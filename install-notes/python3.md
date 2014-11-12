
# Setting up python3 for ipython notebooks

## Installing python

### Installing python3 on OSX using Homebrew

Install the dependencies, then install python3

```shell
brew install xz pkg-config openssl readline sqlite gdbm
brew install python3
```

This installs pip and [pyvenv](https://docs.python.org/3/library/venv.html).

### Installing prerequisites for ipython notebook on OSX using Homebrew

Most of the prerequisites for ipython notebook are probably already installed.

```shell
brew install zmq
```




## Set up a virtual environment

Due to the complex interactions of packages and paths in python, it is advisable to use virtual environments if you want multiple versions of python simultaneously. From python 3.3 onwards, [pyvenv](https://docs.python.org/3/library/venv.html) (py-vee-env) comes with the distribution. To create a virtual environment:

```shell
pyvenv /path/to/environ/py3
```
which will create a folder called py3 in the path above. To use this enviroment (packages, $PYTHONPATH, etc)

```shell
source /path/to/environ/py3/bin/activate
```
To exit the environment
```shell
deactivate
```

## Package management using pip

[pip](https://pip.pypa.io/en/latest/) is the recommended tool for installing and managing python packages. Unless you really need to save disk space, install all packages under [ipython](http://ipython.org/ipython-doc/stable/install/install.html) (which includes the notebook).

```shell
pip install ipython[all]
```

Also install several tools that are commonly used for managing data
```shell
pip install numpy scipy matplotlib pandas
```
