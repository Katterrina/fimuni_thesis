# FI thesis: Exploration of network dynamics approaches to description of brain response to stimulation

This repository contains all the codes and figures for the thesis *Exploration of network dynamics approaches to description of brain response to stimulation*. 

## Thesis abstract

This thesis explores methodologies rooted in complex network analysis to study both empirical and simulated EEG recordings of brain responses to transcranial magnetic stimulation (TMS). The work is based on recent results showing that network communication models capture some of the relationship between structural connectivity and stimulus response for intracranial data. 
      
The thesis first replicates the previous results. Following this, the approach utilizing network communication models is applied to the empirical and simulated TMS-EEG data, comparing the results with each other and with the results obtained for intracranial data. In the end, we perform a robustness analysis regarding the variations in structural connectivity datasets and the methods of group-averaging structural connectivity data.

## Installation

This is organized as an installable project (read more on the rationale [here](https://drivendata.github.io/cookiecutter-data-science/)). After cloning the repository, run the following commands to initialize the environment.

First create a virtual environment. Alternatively use `conda`. Please use Python version at least 3.8.  

```shell
$ python -m venv __venv__
$ . __venv__/bin/activate
```

Next install the dependencies and the project itself.

```shell
$ pip install --upgrade pip # just in case your system is old
$ pip install -r requirements.txt
$ pip install -e .
```

Follow instructions in `data/external/REAME.md` for initialization of the external datasets.

## Repository structure

```
├── data
|   └── exernal     # store external data here
|   └── interim     # data created within this project
├── figures         # all figures created are stored to this folder
├── notebooks       # main part of the project
└── src             # functions imported into notebooks (data loaders, ROI ordering, custom plotting, ...)
```
