from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn.decomposition import PCA

from fomlads.evaluate.eval_regression import create_cv_folds

from .data import DerivedRepresentation
from .model import Model


class ExperimentRunner:
    def __init__(self, model: Model, num_folds: int, pca: PCA = None):
        """Instantiate an experiment runner

        Args:
            model (Model): the model to use
            num_folds (int): number of folds in cross-validation testing
            pca (PCA, optional): If given, use this `pca` to conduct
                dimensionality reduction on inputs. Defaults to None.
        """
        self.__model = model
        self.__pca = pca
        self.__num_folds = num_folds

    def run_partition_experiment(
        self,
        partitioned_dict: Dict[str, Tuple[DataFrame, DataFrame]],
        partitioning_method: str,
        random_seed: int = None,
    ) -> float:
        """Run the experiment on a particular partitioning method
        (e.g. "Flat", "Continent", or "Income Group" and compute
        the weighted average accuracy of this partitioning method.
        Weighted with the sizes of partitions.

        Args:
            partitioned_dict (Dict[str, Tuple[DataFrame, DataFrame]]):
                `{"partition_name": (raw_inputs, raw_targets)}`
            partitioning_method (str, optional): name of the partition.
                (e.g. "Flat" or "Continent").
            random_seed (int, optional): If given, set the random state of np.random
                to this value. Defaults to None.
        Returns:
            (float): weighted average partitioning method accuracy
        """

        if random_seed is not None:
            np.random.RandomState(random_seed)

        partitioning_method_results = ExperimentResult(partitioning_method)

        for partition in partitioned_dict:
            raw_input, raw_target = partitioned_dict[partition]
            print(f"\nNumber of countries in {partition}: {raw_input.shape[0]}")

            partition_accuracies = self.__run_partition_train_and_test(
                raw_input, raw_target
            )

            partitioning_method_results.add_partition_results(
                partition, raw_input.shape[0], partition_accuracies
            )

            accuracy_mean = np.mean(partition_accuracies, axis=0)

            print(f"Mean accuracy of {partition}: {accuracy_mean}")

        print(
            f"""
            \nWeighted Mean Accuracy by {partitioning_method}:
            {partitioning_method_results.weighted_average_accuracy}
            """
        )

    def __run_partition_train_and_test(
        self, raw_inputs: ndarray, raw_targets: ndarray
    ) -> ndarray:
        """Run the experiment on one partition, compute
        the accuracies in each fold.

        Args:
            raw_inputs (ndarray): raw representation of inputs
            raw_targets (ndarray): raw representation of targets

        Returns:
            ndarray:  the accuracies in each fold
        """
        N, D = raw_inputs.shape

        folds = create_cv_folds(N, self.__num_folds)
        derived = DerivedRepresentation(raw_inputs, raw_targets)
        accuracies = np.empty(self.__num_folds)

        for f, (fold_train_filter, fold_test_filter) in enumerate(folds):
            accuracy = self.__run_single_fold_train_and_test(
                derived, fold_train_filter, fold_test_filter
            )
            accuracies[f] = accuracy

        print(f"Partition accuracies: {accuracies}")
        return accuracies

    def __run_single_fold_train_and_test(
        self,
        data: DerivedRepresentation,
        fold_train_filter: ndarray,
        fold_test_filter: ndarray,
    ) -> float:
        """Run the experiment on one partition in one fold and compute
        the accuracy in this fold.

        Args:
            data (DerivedRepresentation): the derived reprersentation of data
            fold_train_filter (ndarray): a binary 1-d array of countries.
                A country is in training set if it has value `True`
            fold_test_filter (ndarray): a binary 1-d array of countries.
                A country is in testing set if it has value `True`

        Returns:
            (float): the accuracy in this fold
        """
        train_inputs, train_targets, test_inputs, test_targets = data.train_test_split(
            fold_train_filter, fold_test_filter
        )

        if self.__pca is not None:
            train_inputs = self.__pca.fit_transform(train_inputs)
            test_inputs = self.__pca.transform(test_inputs)

        self.__model.fit(train_inputs, train_targets)

        accuracy = self.__model.score(test_inputs, test_targets)
        print(f"Fold accuracy: {accuracy}")

        return accuracy


class ExperimentResult:
    """The results of experiments on one partitioning methods
    """

    def __init__(self, partitioning_method: str):
        """Instantiate an experiment result

        Args:
            partitioning_method (str): the partitioning method.
                (`"Flat" | "Continent" | "Income Group"`)
        """

        self.partitioning_method = partitioning_method
        self.partitions: List[PartitionResults] = []

    def add_partition_results(
        self, partition_name: str, partition_size: int, accuracies: List[float]
    ) -> None:
        """Add the results for a partition to the experiment results

        Args:
            partition_name (str): name of the partition (e.g. `"Asia"`)
            partition_size (int): size of the partition
            accuracies (List[float]): accuracy in each fold
        """
        partition_results = PartitionResults(partition_name, partition_size, accuracies)
        self.partitions.append(partition_results)

    @property
    def weighted_average_accuracy(self) -> float:
        """Get the average accuracy for this partitioning method,
        weighted by the partition sizes

        Returns:
            float: weighted average accuracy for the partitioning method
        """
        total = 0

        partitioning_accuracies: ndarray = np.empty(len(self.partitions))
        partition_sizes: ndarray = np.empty(len(self.partitions))

        for i, partition_results in enumerate(self.partitions):
            total += partition_results.partition_size
            partition_mean_accuracy = np.mean(partition_results.accuracies)

            partitioning_accuracies[i] = partition_mean_accuracy
            partition_sizes[i] = partition_results.partition_size

        weights = partition_sizes / total
        weighted_average_accuracy = np.average(partitioning_accuracies, weights=weights)
        return weighted_average_accuracy


class PartitionResults:
    def __init__(
        self, partition_name: str, partition_size: int, accuracies: List[float]
    ):
        self.partition_name = partition_name
        self.partition_size = partition_size
        self.accuracies = accuracies.copy()
