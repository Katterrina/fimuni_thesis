# FI thesis


## Installation

This is organized as an installable project (read more on the rationale [here](https://drivendata.github.io/cookiecutter-data-science/)). After cloning the repository, run the following commands to initialize the environment.

First create a virtual environment. Alternatively use `conda`. Please use Python version at least 3.8.  

```shell
$ python -mvenv env
$ . env/bin/activate
```

Next install the dependencies and the project itself.

```shell
$ pip install --upgrade pip # just in case your system is old
$ pip install -r requirements.txt
$ pip install -e .
```

Follow instructions in `data/external/REAME.md` for initialization of the external datasets.

