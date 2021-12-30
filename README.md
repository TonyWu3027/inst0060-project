# INST0060 Group Project - Group E

This is the code archive by Group E for the *INST0060* Group Project.

## How to Use

To conduct the experiment, the [OWID COVID-19 Dataset](https://covid.ourworldindata.org/data/owid-covid-data.csv) needs to be provided.

Start the experiment with `main.py` using command-line arguments. For instance:

```bash
python3 main.py ./covid.csv
```

## How it Works

### Data Preprocessing

First of all, we rule out the countries which lack valid information. By jointing several charts and dividing countries into groups, we obtain a data frame, which for each row represents a particular country and its corresponding index.

Overall, We will work on the daily average of all index within a month period.

### Representation and Basis Function

### Partition

### Algorithm

### Evaluation

In order the evaluate the algorithm, we compare the prediction and actual value by

- Plotting a confusion matrix.
- Calculating its F1 score. The higher F1 score stand for a more accurate algorithm.

## Contributing

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
