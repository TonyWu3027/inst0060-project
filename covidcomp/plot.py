from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import zscore
from sklearn.decomposition import PCA

from config import OUTPUT_DIR

from .experiment import ExperimentResult


class Plotter:
    """A plotter that plots all the diagrams in the experiment"""

    FONT_SIZE = 20

    def __init__(self):
        pass

    def plot_data_corr(self, flat_raw: DataFrame) -> None:
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
        plt.suptitle("Correlation Matrix Heatmap", fontsize=self.FONT_SIZE)
        plt.savefig(join(OUTPUT_DIR, "correlation-matrix-heatmap.png"))

    def plot_hist(self, inputs: DataFrame, title: str = "") -> None:
        """Plot the histogram of data cols in inputs.

        Args:
            inputs (DataFrame): inputs data in DataFrame
            title (str, optional): plot title. Defaults to "".
        """
        plt.figure()
        inputs.hist(figsize=(20, 10))
        plt.suptitle(title, fontsize=self.FONT_SIZE)
        file_name = title.lower().replace(" ", "-") + ".png"
        plt.savefig(join(OUTPUT_DIR, file_name))

    def plot_pca_explained_variance(self, inputs: ndarray) -> None:
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
        plt.suptitle(
            "Dimensionality Reduction with PCA - Ratio of Variance Explained",
            fontsize=self.FONT_SIZE,
        )

        plt.savefig(join(OUTPUT_DIR, "pca-explained-variance-ratio.png"))

    def plot_partitioning_method_results(self, result: ExperimentResult) -> None:
        """Plot the Box-and-Whisker diagram of the results
        in one partitioning method

        Args:
            result (ExperimentResult): [description]
        """

        method = result.partitioning_method
        data, labels = result.get_boxplot_data()
        plt.figure(figsize=(10, 10))
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.xlabel("Partition and corresponding size")
        plt.ylabel("Accuracies")
        plt.suptitle(
            f"Accuracies in Each Partition, Partitioned by {method}",
            fontsize=self.FONT_SIZE,
        )

        plt.savefig(join(OUTPUT_DIR, f"{method}-partition-accuracies.png"))
