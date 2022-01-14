from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from fomlads.evaluate.eval_classification import eval_accuracy
from fomlads.model.classification import (
    logistic_regression_predict,
    logistic_regression_prediction_probs,
)


class Model(ABC):
    """The Abstract Base Class for models"""

    @abstractmethod
    def fit(self, inputs: ndarray, targets: ndarray) -> ndarray:
        """
        Fits a set of weights to the model

        Args:
            inputs (ndarray): an N*D matrix, each row is a data-point
            targets (ndarray): N-dimension vector of target labels

        Returns:
            (ndarray): - a set of weights for the model
        """
        pass

    @abstractmethod
    def predict(
        self,
        inputs: ndarray,
        decision_threshold: float = 0.5,
        add_bias_term: bool = True,
    ) -> ndarray:
        """
        Get prediction vector from the model.

        Args:
            inputs (ndarray): an N*D matrix of input data (or design matrix)
            decision_threshold (float): the prediction probability above which
                the output prediction is 1. Set to 0.5 for minimum misclassification
            add_bias_term (bool): whether or not a bias term should be added
                to the input or the design matrix
        Returns:
            (ndarray): an N-dimension vector of predictions
        """
        pass

    @abstractmethod
    def score(self, test_inputs: ndarray, test_targets: ndarray) -> float:
        """Return the mean accuracy on the given test inputs and targets.

        Args:
            test_inputs (ndarray): an N*D inputs/design matrix of data points.
            test_targets (ndarray): an D-dimension vector of true targets.
        Returns:
            (float): the mean accuracy on the given test inputs and targets
        """
        pass


class L2RegularisedLogisticRegression(Model):
    """The L-2 Regularised Logistic Regression
    implemented with Iteratively Reweighted Least
    Squares algorithm and Newton-Raphson method
    """

    def __init__(
        self, weights0=None, lambda_=1e3, termination_threshold=1e-8, add_bias_term=True
    ):
        """Instantiate an L-2 Regularised Logistic Regression model

        Args:
            weights0 (ndarray, optional): the initial weights for iteration.
                Defaults to None.
            lambda_ (number, optional): the regularising strength.
                Defaults to 1e3.
            termination_threshold (number, optional): [description].
                Defaults to 1e-8.
            add_bias_term (bool, optional): [description]. Defaults to True.
        """
        self.__weights0 = weights0
        self.__lambda = lambda_
        self.__add_bias_term = add_bias_term
        self.__termination_threshold = termination_threshold

    def fit(self, inputs: ndarray, targets: ndarray) -> ndarray:
        """
        Fits a set of weights to the L-2 regularised logistic regression model
        using the iteratively reweighted least squares (IRLS) method (Rubin, 1983)

        Adapted from `fomlads.model.classification.logistic_regression_fit`
        by Dr Luke Dickens, at the Department of Information Studies, UCL

        Args:
            inputs (ndarray): an N*D matrix, each row is a data-point
            targets (ndarray): N-dimension vector of class IDs 0 and 1

        Returns:
            (ndarray): - a set of weights for the model
        """

        # Reshape the matrix for 1d inputs
        if len(inputs.shape) == 1:
            inputs = inputs.reshape((inputs.size, 1))

        N, D = inputs.shape
        # We should have a bias term (but this may already be included in design
        # matrix). Adding a bias term is equivalent to adding contant first col
        if self.__add_bias_term:
            # we do this by making the first column all ones
            inputs = np.hstack((np.ones((N, 1)), inputs))
            # and increasing the apparent dimension
            D += 1
        targets = targets.reshape((N, 1))
        # initialise the weights
        if self.__weights0 is None:
            weights = np.random.multivariate_normal(np.zeros(D), 0.001 * np.identity(D))
        else:
            weights = self.__weights0

        weights = weights.reshape((D, 1))
        # initially the update magnitude is set as larger than the
        # termination_threshold to ensure the first iteration runs
        update_magnitude = 2 * self.__termination_threshold

        # Initialise the identity matrix
        identity_matrix = np.identity(D)

        while update_magnitude > self.__termination_threshold:
            print(update_magnitude)
            # calculate the current prediction vector for weights
            # we don't want to extend the inputs a second time so add_bias_term
            # is set to False
            predicts = logistic_regression_prediction_probs(
                inputs, weights, add_bias_term=False
            )
            # the diagonal reweighting matrix (easier with predicts as flat array)
            R = np.diag(predicts * (1 - predicts))
            # reshape predicts to be same form as targets
            predicts = predicts.reshape((N, 1))
            # Calculate the Hessian inverse
            H_inv = np.linalg.inv(
                inputs.T @ R @ inputs + self.__lambda * identity_matrix
            )
            # Calculate the gradient
            gradient = inputs.T @ (predicts - targets) + self.__lambda * weights
            # update the weights
            new_weights = weights - H_inv @ gradient
            # calculate the update_magnitude
            update_magnitude = np.sqrt(np.sum((new_weights - weights) ** 2))
            # update the weights
            weights = new_weights
        self.__weights = weights

    def predict(
        self,
        inputs: ndarray,
        decision_threshold: float = 0.5,
        add_bias_term: bool = True,
    ) -> ndarray:
        """
        Get deterministic class prediction vector from the logistic regression model.

        Adapted from `fomlads.model.classification.logistic_regression_predict`
        by Dr Luke Dickens, at the Department of Information Studies, UCL

        Args:
            inputs (ndarray): an N*D matrix of input data (or design matrix)
            decision_threshold (float): the prediction probability above which
                the output prediction is 1. Set to 0.5 for minimum misclassification
            add_bias_term (bool): whether or not a bias term should be added
                to the input or the design matrix
        Returns:
            (ndarray): an N-dimension vector of predictions
        """
        return logistic_regression_predict(
            inputs, self.__weights, decision_threshold, add_bias_term
        )

    def score(self, test_inputs: ndarray, test_targets: ndarray) -> float:
        """Return the mean accuracy on the given test inputs and targets.

        Args:
            test_inputs (ndarray): an N*D inputs/design matrix of data points.
            test_targets (ndarray): an D-dimension vector of true targets.
        Returns:
            (float): the mean accuracy on the given test inputs and targets
        """

        test_predicts = self.predict(test_inputs)
        accuracy = eval_accuracy(test_targets, test_predicts)
        return accuracy
