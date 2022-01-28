# INST0060 Group Project - Group E

This is the code archive by Group E for the *INST0060* Group Project.

## Aim of this Project

This project aim to explore the use of logistic regression model to compare between two countries' average daily COVID-19 deaths. We particularly focus on whether grouping countries by *Continents* or by *Income Group* can help to improve classification accuracy.

## How to Use

### Install Dependencies

The best practice recommended is to create an environment for the project and install dependencies. To do this, navigate to the root directory of the project, create a new environment, and install dependencies via `requirements.txt`. With `conda` as an example:

Create a new environment

```bash
conda create -n covidcomp python=3.7
```

Activate the new environment

```bash
conda activate covidcomp
```

Install dependencies from `requirements.txt`

```bash
conda install -n covidcomp --file requirements.txt
```

> Note: the `requirements.txt` is adapted from the provided requirement file in INST0060 module. `seaborn` is added to the requirements for plotting a heat map of correlation matrix.

### Run the experiment

To conduct the experiment, the [OWID COVID-19 Dataset](https://covid.ourworldindata.org/data/owid-covid-data.csv) needs to be provided.

Start the experiment with `main.py` using command-line arguments. For instance:

```bash
python3 main.py ./covid.csv
```

Prompts of experiment running status will be printed in the console. Plots will be saved to `./output/`

### Playground

A playground jupyter notebook `./playground.ipynb` has been provided alongside with the experiment script to enable interactive experimenting.

## Project Structure

- `./covidcomp/` is the main Python library implemented for the project.

    - `covidcomp.data` module processes the data from the original CSV file to Raw Representation and Derived Representation.

    - `covidcomp.model` module provides an ABC `Model`, with is inherited by all concrete model implementation. `LogisticsRegression` and `L2RegularisedLogisticRegression` are implemented.

    - `covidcomp.experiment` module provides `ExperimentRunner` and `ExperimentResult` to run experiments on partitioned and flat datasets with cross-validation and store results as `ExperimentResult` instances.

    - `covidcomp.plot` module provides a `Plotter` that will plot the required figures in the experiment.

- `./fomlads/` is the supporting Python library provided in INST0060 Foundation of Machine Learning, from which `covidcomp` is developed.
- `./data/` contains all the auxiliary datasets.
- `./output/` is the output directory for plots.
- `./playground.ipynb` see above.
- `./main.py` the entry to the programme, see above.
- The rest are project related configurations and IDE settings

## How to Contribute

### Code Linting

Code linting is done before committing your code through [_pre-commit_](https://pre-commit.com), configured with `.pre-commit-config.yaml`. In order to enable the pre-commit linting hooks, you will need to install `pre-commit` in your environment and install the hooks.

Install `pre-commit` (using `conda` as an example):

```bash
conda install pre-commit
```

Check if `pre-commit` is installed properly:

```bash
pre-commit --version
```

Install hooks as specified in the config file:

```bash
pre-commit install
```
