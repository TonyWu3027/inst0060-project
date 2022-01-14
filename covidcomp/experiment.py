from typing import Dict, Tuple

import numpy as np
from pandas import DataFrame

from fomlads.evaluate.eval_classification import eval_accuracy

from .data import DerivedRepresentation
from .model import Model


class ExperimentRunner:
    def run_train_and_test(
        partitioned_dict: Dict[str, Tuple[DataFrame, DataFrame]],
        model: Model,
        label: str = "",
        test_fraction: float = 0.5,
    ):
        """Fit a given model and print test scores.

        Args:
            partitioned_dict (Dict[str, Tuple[DataFrame, DataFrame]]):
                `{"partition_name": (raw_inputs, raw_targets)}`
            model (Model): the model to use
            label (str, optional): name of the partition. (e.g. "Flat" or "Continent")
                Defaults to "".
            test_fraction (float, optional): a fraction (between 0 and 1) specifying the
                proportion of the data to use as test data. Defaults to 0.5.
        """

        accuracies = []
        partition_sizes = []

        for partition in partitioned_dict:
            raw_input, raw_target = partitioned_dict[partition]
            print(f"\nNumber of countries in {partition}: {raw_input.shape[0]}")
            derived_continent = DerivedRepresentation(
                raw_input, raw_target, test_fraction
            )

            train_inputs = derived_continent.train_inputs
            train_targets = derived_continent.train_targets
            test_inputs = derived_continent.test_inputs
            test_targets = derived_continent.test_targets

            print(f"Number of training pairs in {partition}: {train_inputs.shape[0]}")

            model.fit(train_inputs, train_targets)
            test_predictions = model.predict(test_inputs)

            accuracy = eval_accuracy(test_targets, test_predictions)

            accuracies.append(accuracy)
            partition_sizes.append(raw_input.shape[0])

            print(f"\n{partition} Accuracy : {accuracy}")

        partition_weights = np.divide(partition_sizes, len(partitioned_dict))
        weighted_avg_accuracy = np.average(accuracies, weights=partition_weights)

        print(f"\nWeighted Mean Accuracy by {label} : {weighted_avg_accuracy}")
