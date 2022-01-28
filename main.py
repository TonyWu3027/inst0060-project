import argparse
import os
from argparse import ArgumentParser

from sklearn.decomposition import PCA

from covidcomp.data import DerivedRepresentation, RawRepresentation
from covidcomp.experiment import ExperimentRunner
from covidcomp.model import LogisticRegression
from covidcomp.plot import Plotter


def is_valid_file_path(parser: ArgumentParser, file_path: str) -> str:
    """Determine whether a file_path is valid or not.
    If not, an ArgumentParser error will be promopted.

    Args:
        parser (ArgumentParser): the ArgumentParser
        file_path (str): the given file path

    Returns:
        str: the valid file_path
    """
    if not os.path.exists(file_path):
        parser.error(f"The file {file_path} does not exist")
    else:
        return file_path


def main():
    # Read file path
    parser = argparse.ArgumentParser(description="Train the COVID-19 comparison model.")
    parser.add_argument(
        "file_path",
        type=lambda x: is_valid_file_path(parser, x),
        help="the file path to the OWID COVID-19 dataset",
    )

    args = parser.parse_args()
    file_path = args.file_path

    print("====Generating Raw Representation from CSV====")
    raw = RawRepresentation(file_path)

    print("====Preparing experiment environment====")
    model = LogisticRegression()
    pca = PCA(n_components=8)
    num_folds = 8
    runner = ExperimentRunner(model, num_folds, pca=pca)
    plotter = Plotter()

    print("====Generating Derived Representation for Flat and Partitioned dataset====")
    flat_dict = raw.get_representation()
    partitioned_by_continent_dict = raw.get_representation("continent")
    partitioned_by_income = raw.get_representation("income_group")

    print("Note: the plots will be saved to ./output/")

    print("====Plotting Correlation Heat Map of Data====")
    plotter.plot_data_corr(raw.frame)

    print("====Plotting the Input Data Before Preprocessing===")
    raw_inputs, raw_targets = flat_dict["Flat"]
    flat_derived = DerivedRepresentation(raw_inputs, raw_targets)

    plotter.plot_hist(
        raw_inputs, flat_derived.SKEWED_COLUMNS, title="Input Data Before Preprocessing"
    )

    print("====Plotting the Input Data After Preprocessing===")
    plotter.plot_hist(
        flat_derived.preprocessed_inputs,
        flat_derived.SKEWED_COLUMNS,
        title="Input Data After Preprocessing",
    )

    print("====Plotting PCA Explained Variance====")
    plotter.plot_pca_explained_variance(flat_derived.inputs)

    print("====Running Experiments====")

    # Run experiments on flat framework
    flat_results = runner.run_partition_experiment(
        flat_dict, partitioning_method="Flat"
    )
    print("====Plotting Flat Experiment Accuracies")
    plotter.plot_partitioning_method_results(flat_results)
    print(flat_results.descriptive_statistics)

    # Run experiments on data partitioned by continent
    continent_results = runner.run_partition_experiment(
        partitioned_by_continent_dict, partitioning_method="Continent"
    )
    print("====Plotting Continent Experiment Accuracies")
    plotter.plot_partitioning_method_results(continent_results, flat_results)
    print(continent_results.descriptive_statistics)

    # Run experiments on data partitioned by income group
    income_results = runner.run_partition_experiment(
        partitioned_by_income, partitioning_method="Income Group"
    )
    print("====Plotting Income Group Experiment Accuracies")
    plotter.plot_partitioning_method_results(income_results, flat_results)
    print(income_results.descriptive_statistics)


if __name__ == "__main__":
    main()
