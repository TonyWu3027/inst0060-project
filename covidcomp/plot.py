from os.path import join
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import zscore
from sklearn.decomposition import PCA

from config import OUTPUT_DIR

from .experiment import ExperimentResult
from .model import Model


class Plotter:
    """A plotter that plots all the diagrams in the experiment"""

    FONT_SIZE = 16

    def __init__(self):
        pass

    def plot_data_corr(
        self, flat_raw: DataFrame, title="Correlation Matrix Heatmap"
    ) -> None:
        """Plot the correlation heatmap of the data

        Args:
            flat_raw (DataFrame): raw representation of the data
                in DataFrame
        """
        corr_matrix = flat_raw.corr()
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            corr_matrix, center=0, cmap=sns.diverging_palette(20, 220, n=200)
        )
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )

        plt.savefig(join(OUTPUT_DIR, f"{title}.png"))

    def plot_hist(self, inputs: DataFrame, cols: List[str], title: str = "") -> None:
        """Plot the histogram of data cols in inputs.

        Args:
            inputs (DataFrame): inputs data in DataFrame
            cols (List[str]): the columns to show
            title (str, optional): plot title. Defaults to "".
        """
        plt.figure()
        inputs.hist(column=cols, figsize=(8, 6), layout=(2, 3))

        plt.savefig(join(OUTPUT_DIR, f"{title}.png"))

    def plot_pca_explained_variance(
        self, inputs: ndarray, title="PCA - Ratio of Variance Explained"
    ) -> None:
        """Plot the explained variance ratio against the number
        of components in PCA

        Args:
            inputs (ndarray): inputs data in ndarray
        """
        # Feature scaling
        inputs_scaled = zscore(inputs)

        # Fit PCA
        pca = PCA(n_components=None)
        pca.fit(inputs_scaled)

        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
        plt.xlabel("Number of Components (Dimensions)")
        plt.ylabel("Variance Explained (%)")

        plt.savefig(join(OUTPUT_DIR, f"{title}.png"))

    def plot_partitioning_method_results(
        self,
        result: ExperimentResult,
        control_result: ExperimentResult = None,
        figsize: Tuple[float, float] = (0, 1),
        ylim: Tuple[float, float] = (0, 1),
    ) -> None:
        """Plot the Box-and-Whisker diagram of the results
        in one partitioning method. If control group is given,
        plot control group in highlighted colours.

        Args:
            result (ExperimentResult): experiment group result
            control_result (ExperimentResult, optional):
                control group result if given. Defaulted to None.
            figsize ((float, float)): (width, height) of plot.
            ylim ((float, float)): (bottom, top) for plot y limits.
        """
        # Prepare data
        method = result.partitioning_method
        data, labels = result.get_boxplot_data()

        # Concatenate the control group
        # if control_result is given
        if control_result is not None:
            control_data, control_labels = control_result.get_boxplot_data()
            data = np.append(data, control_data, axis=1)
            labels.append(control_labels[0])

        # Plot data
        plt.figure(figsize=(12, 7))

        bp_dict = plt.boxplot(data, labels=labels, showmeans=True, meanline=True)

        # Set styles
        plt.xlabel("Partition (size)")
        plt.ylabel("Accuracies")
        plt.ylim(ylim)

        # Change the style of the control group
        if control_result is not None:
            control_colour = "#E83E8B"
            bp_dict["boxes"][-1].set(color=control_colour)
            bp_dict["whiskers"][-1].set(color=control_colour)
            bp_dict["caps"][-1].set(color=control_colour)
            bp_dict["fliers"][-1].set(color=control_colour)
            bp_dict["whiskers"][-2].set(color=control_colour)
            bp_dict["caps"][-2].set(color=control_colour)

        # Show legends
        plt.legend(
            [bp_dict["medians"][0], bp_dict["means"][0]],
            ["median", "mean"],
            fontsize="x-large",
        )

        plt.savefig(join(OUTPUT_DIR, f"{method} Partition Accuracies.png"))

    def plot_decision_boundary(
        self, X: ndarray, y: ndarray, model: Model, pca: PCA = None
    ) -> None:
        """Plot the decision boundary of predictions from
        a fitted model.

        Args:
            X (ndarray): N*D inputs data or design matrix
            y (ndarray): N-d vector of true binary labels
            model (Model): the fitted model
            pca (PCA, optional): the PCA to use, if given.
                Defaulted to None.
        """
        raise NotImplementedError()
        # @Harrison
        # TODO: transform X with PCA
        # TODO: predict with model
        # TODO: plot X with true labels
        # TODO: plot prediction regions
