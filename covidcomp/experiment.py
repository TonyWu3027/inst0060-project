from typing import Dict, Tuple

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
        label: str = "",
        random_seed: int = None,
    ) -> float:
        """Run the experiment on a particular partitioning method
        (e.g. "Flat", "Continent", or "Income Group" and compute
        the weighted average accuracy of this partitioning method.
        Weighted with the sizes of partitions.

        Args:
            partitioned_dict (Dict[str, Tuple[DataFrame, DataFrame]]):
                `{"partition_name": (raw_inputs, raw_targets)}`
            label (str, optional): name of the partition. (e.g. "Flat" or "Continent")
                Defaults to "".
            random_seed (int, optional): If given, set the random state of np.random
                to this value. Defaults to None.
        Returns:
            (float): weighted average partitioning method accuracy
        """

        if random_seed is not None:
            np.random.RandomState(random_seed)

        partition_accuracies = []
        partition_sizes = []

        for partition in partitioned_dict:
            raw_input, raw_target = partitioned_dict[partition]
            print(f"\nNumber of countries in {partition}: {raw_input.shape[0]}")

            partition_accuracy = self.__run_partition_train_and_test(
                raw_input, raw_target
            )

            accuracy_mean = np.mean(partition_accuracy, axis=0)
            accuracy_stderr = np.std(partition_accuracy, axis=0) / np.sqrt(
                self.__num_folds
            )

            print(f"Mean accuracy of {partition}: {accuracy_mean}")
            print(f"S.d. accuracy of {partition}: {accuracy_stderr}")

            partition_accuracies.append(accuracy_mean)

            partition_sizes.append(raw_input.shape[0])

        partition_weights = np.divide(partition_sizes, len(partitioned_dict))
        weighted_avg_accuracy = np.average(
            partition_accuracies, weights=partition_weights
        )

        print(f"\nWeighted Mean Accuracy by {label} : {weighted_avg_accuracy}")
        return weighted_avg_accuracy

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
