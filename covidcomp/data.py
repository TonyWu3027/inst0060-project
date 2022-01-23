import os
from datetime import datetime
from os import PathLike
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from config import DATA_DIR, OUTPUT_DIR


class RawRepresentation:
    """The raw data reperesentation generated from
    a series of preprocessing on the
    "Our World in Data" COVID-19 Dataset
    """

    START = datetime(2021, 10, 26)
    END = datetime(2021, 11, 26)

    # Target column
    TARGET_COL = "new_deaths_per_million"

    # Columns that can be used to partition
    PARTITION_COLS = ["continent", "income_group"]

    # Required columns in the OWID dataset
    OWID_COLS = [
        "iso_code",
        "date",
        "continent",
        "new_cases_per_million",
        "new_deaths_per_million",
        "people_vaccinated_per_hundred",
        "population",
        "life_expectancy",
    ]

    # Required columns in the stringency dataset
    STRINGENCY_COLS = [
        "C1_School closing",
        "C2_Workplace closing",
        "C3_Cancel public events",
        "C4_Restrictions on gatherings",
        "C5_Close public transport",
        "C6_Stay at home requirements",
        "C7_Restrictions on internal movement",
        "C8_International travel controls",
    ]

    def __init__(self, file_path: Union[PathLike, str]):

        # Load the "Our World In Data" COVID-19 Dataset with required columns
        owid_frame = self.load_data_frame(file_path, self.OWID_COLS)

        # Compute the daily average in the date range
        self.__covid_frame = self.average_over_date_range(
            owid_frame, "iso_code", "date", self.START, self.END, is_covid=True
        )

        # Join GDP per Capita
        self.__join_auxiliary_dataset(
            os.path.join(DATA_DIR, "gdp_per_capita_wb.csv"),
            ["gdp_per_capita"],
            "Country Code",
            "..",
        )

        # Join Population ages 65+
        self.__join_auxiliary_dataset(
            os.path.join(DATA_DIR, "65_above_share_wb.csv"),
            ["population_ages_65_and_above"],
            "Country Code",
            "..",
        )

        # Join Population Density
        self.__join_auxiliary_dataset(
            os.path.join(DATA_DIR, "pop_den_wb.csv"),
            ["population_density"],
            "Country Code",
            "..",
        )

        # Join the income group
        self.__join_auxiliary_dataset(
            os.path.join(DATA_DIR, "income_group_wb.csv"),
            ["income_group"],
            aux_index_col="Country Code",
            is_numerical=False,
        )

        # Join Stringency index attributes
        stringency_raw = self.load_data_frame(os.path.join(DATA_DIR, "stringency.csv"))
        stringency_daily_avg = self.average_over_date_range(
            stringency_raw,
            "CountryCode",
            "Date",
            self.START,
            self.END,
            date_format="%Y%m%d",
        )
        self.__join_auxiliary_dataset(stringency_daily_avg, self.STRINGENCY_COLS)

    @property
    def frame(self) -> DataFrame:
        """Get a copy of the raw representation
        DataFrame

        Returns:
            DataFrame: a copy of the raw representation DataFrame
        """

        return self.__covid_frame.copy()

    def load_data_frame(
        self, file_path: Union[PathLike, str], columns: List[str] = []
    ) -> DataFrame:
        """Load required columns in a dataset as a DataFrame.
        If no column is given, all columns in the original dataset will be loaded.

        Args:
            file_path (PathLike | str): the file path to the CSV dataset
            columns (List[str], optional): the required columns. Defaults to [].

        Returns:
            DataFrame: the resultant DataFrame
        """

        frame = pd.read_csv(file_path)

        # Select required columns from the raw CSV
        if columns:
            frame = frame[columns]

        return frame

    def average_over_date_range(
        self,
        input: DataFrame,
        index_col: str,
        date_col: str,
        start_date: datetime,
        end_date: datetime,
        date_format: str = "%Y-%m-%d",
        is_covid: bool = False,
    ) -> DataFrame:
        """Calculate the daily average of all data attributes in dataset
        for each country in a given period of time.

        Args:
            input (DataFrame): the input DataFrame
            index_col (str): the index column in `input`
            date_col (str): the column label for date
            start_date (date): start date inclusively
            end_date (date): end date inclusively
            date_format (str): Default to "%Y-%m-%d". The format of date in the dataset
            is_covid (bool): Default to False.
            Whether the input is the OWID COVID-19 Dataset

        Raises:
            ValueError: raised when `end_date` is earlier than the `start_date`
            KeyError: raised when `index_col` is not given

        Returns:
            DataFrame: the resultant DataFrame, indexed with `index_col`
        """

        if end_date < start_date:
            raise ValueError("The end date should not be earlier than the start date")

        if not index_col:
            raise KeyError("Index columns not given")

        # Select the date within the range
        input[date_col] = pd.to_datetime(input[date_col], format=date_format)
        mask = (input[date_col] >= start_date) & (input[date_col] <= end_date)
        result = input.loc[mask]

        by = [index_col]

        # A temporary workaround for the OWID COVID-19 dataset
        if is_covid:
            by.append("continent")

        # Take the daily average for each country and set index column
        result = result.groupby(by, as_index=False).mean()
        result.set_index(index_col, inplace=True)

        return result

    def __join_auxiliary_dataset(
        self,
        auxiliary: Union[str, PathLike, DataFrame],
        columns: List[str],
        aux_index_col: str = "",
        na_value: str = "",
        is_numerical: bool = True,
    ) -> None:
        """Join the required columns in an auxiliary dataset
        to the original DataFrame with empty entries dropped

        Note:
            the auxiliary dataset needs to be indexed with ISO country code

        Args:
            auxiliary (str | DataFrame | PathLike): the path to
            or the DataFrame of the auxiliary dataset.
            Should the DataFrame be given, it should be indexed with `aux_index_col`
            columns (List[str]): the required columns in the auxiliary dataset
            aux_index_col (str): Default to "".
            The index column in the auxiliary DataFrame
            na_value (str): Default to "". The representation for NaN value
            in the auxiliary dataset
            is_numerical (bool): Default to True. Whether the data is numerical or not
        Raises:
            ValueError: raised if `columns` is empty

        Returns:
            DataFrame: the resultant dataset
        """

        if not columns:
            raise ValueError(
                "Required column(s) in the auxiliary dataset is not specified"
            )

        if isinstance(auxiliary, (str, PathLike)):
            # If a file path is given
            # Read the CSV as a DataFrame and set index
            aux_frame = pd.read_csv(auxiliary).set_index(aux_index_col)
        else:
            # If a DataFrame is given,
            # Use a copy of the DataFrame
            aux_frame = auxiliary.copy()

        # Read the required columns in the auxiliary CSV with given index
        aux_frame = aux_frame[columns]

        # Replace the null values with NA and drop the rows
        aux_frame = aux_frame.replace(na_value, pd.NA).dropna()

        # If data is numerical, convert to float
        if is_numerical:
            aux_frame = aux_frame.astype(float)

        # Join the data frames on ISO Country Code and drop empty row entries
        result = self.__covid_frame.join(aux_frame).dropna()

        self.__covid_frame = result

    def export_csv(self) -> None:
        """Export the preprocessed raw dataset to CSV in `output` directory
        """

        self.__covid_frame.to_csv(
            os.path.join(OUTPUT_DIR, f"preprocessed-{datetime.now()}.csv")
        )

    def __get_flat_representation(self) -> Tuple[DataFrame, DataFrame]:
        """Get the flat raw representation of the dataset

        Returns:
            Tuple[DataFrame, DataFrame]: the flat input and the flat target
        """

        raw_input = self.__covid_frame.drop(
            columns=[*self.PARTITION_COLS, self.TARGET_COL]
        )
        raw_target = self.__covid_frame[[self.TARGET_COL]]

        return raw_input, raw_target

    def get_representation(
        self, partition_col: str = ""
    ) -> Dict[str, Tuple[DataFrame, DataFrame]]:
        """Get a raw representation of the dataset. If the
        `partition_col` is not give, a flat represention will be returned

        Args:
            partition_col (str): the partitioning column
                (e.g. `"continent"`, `"income_group"`).
                Defaulted to ""


        Raises:
            ValueError: raised when `partition_col` is not a categorical column

        Returns:
            Dict[str, Tuple[DataFrame, DataFrame]]: {category: (input, target)}.
            For instance, when partitioned by continent, the output might look like:
            `{"Asia": (raw_input_for_asia, raw_target_for_asia), "Europe": (..), ...}`
        """

        if partition_col == "":
            result = {"Flat": self.__get_flat_representation()}
            return result
        elif partition_col not in self.PARTITION_COLS:
            raise ValueError("The partitioning column is invalid")

        # *A WORKAROUND FOR PARTITIONING WITH CONTINENTS*
        # Since the numbers of countries in Oceania, South America,
        # and North America are relatively smaller, we merge
        # the 3 continents as "Other"
        if partition_col == "continent":
            df = self.__covid_frame.replace(
                ["South America", "Oceania", "North America"], "Other"
            )
        # elif partition_col == "income_group":
        #     df = self.__covid_frame.replace(
        #         ["Low income", "Lower middle income"], "Low or lower middle income"
        #     )
        else:
            df = self.__covid_frame.copy()

        # Get a list of categories with the partitioning method
        categories = df[partition_col].unique().tolist()

        result = dict()

        for category in categories:
            partitioned_frame = df.loc[df[partition_col] == category]
            partitioned_input = partitioned_frame.drop(
                columns=[*self.PARTITION_COLS, self.TARGET_COL]
            )
            partitioned_target = partitioned_frame[[self.TARGET_COL]]
            result[category] = (partitioned_input, partitioned_target)

        return result


class DerivedRepresentation:
    """The derived representation generated from
    pairing the raw representation into raw features
    and apply basis function to these raw features
    """

    SKEWED_COLUMNS = [
        "new_cases_per_million",
        "population",
        "life_expectancy",
        "gdp_per_capita",
        "population_density",
        "population_ages_65_and_above",
    ]

    def __init__(self, input: DataFrame, target: DataFrame):
        """Construct a DerivedRepresentation instance

        Args:
            input (DataFrame): the raw representation of the dataset
                from RawRepresentation
            target (DataFrame): the target column of the dataset from RawRepresentation
        """

        # Preprocess inputs
        self.__pre_processed_inputs = self.preprocess_input_representation(input)
        self.__pre_processed_targets = target.copy()

        # All paired inputs and targets
        self.__derived_inputs, self.__derived_targets = self.pair(
            self.__pre_processed_inputs, target
        )

    def pair(
        self, inputs: DataFrame, targets: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """Pair the inputs and targets all drop the pairs where i = j

        Args:
            inputs (DataFrame): raw and preprocessed inputs
            targets (DataFrame): raw target

        Returns:
            Tuple[DataFrame, DataFrame]: the paired inputs and corresponding targets
        """

        # Pair inputs and targets
        paired_inputs = self.__pair_frame(inputs)
        paired_raw_targets = self.__pair_frame(targets)

        # Encode the targets as binary classes
        paired_targets = paired_raw_targets.applymap(self.__binary_mapping).astype(int)

        return paired_inputs, paired_targets

    def __pair_frame(self, frame: DataFrame) -> DataFrame:
        """The basis function that pairs the data points
        in the DataFrame by taking the difference between
        the members in a pair.

        Equation:
            phi_{ij} = x_i - x_j

        Args:
            frame (DataFrame): the input (N*D) or target(N*1) DataFrame

        Returns:
            DataFrame: the paired input (N^2*D) or target(N^2*1) DataFrame
        """
        N, _ = frame.shape

        # Concatenate the vectors for the paired DataFrame
        columns = frame.columns
        paired_frame = (
            frame.assign(key=1).merge(frame.assign(key=1), on="key").drop("key", 1)
        )

        # Take the differences between the vectors
        derived_frame = pd.DataFrame()
        derived_frame[columns] = np.subtract(
            paired_frame[[f"{col}_x" for col in columns]],
            paired_frame[[f"{col}_y" for col in columns]],
        )

        # Drop the case when a country is paired with itself
        duplicated_index = [i for i in range(0, N ** 2, N + 1)]
        derived_frame = derived_frame.drop(duplicated_index)

        return derived_frame

    def log_columns(self, frame: DataFrame, columns: List[str]) -> DataFrame:
        """Take the log of the specified columns
        in the given DataFrame. Rename the columns
        as <column_label>_log.

        Args:
            frame (DataFrame): a given DataFrame
            columns (List[str]): the columns to be taken log

        Returns:
            DataFrame: the resultant DatFrame
        """

        result = frame.copy()

        result[[f"{col}_log" for col in columns]] = np.log(result[columns])

        result.drop(columns=columns, inplace=True)

        return result

    def preprocess_input_representation(self, input: DataFrame) -> DataFrame:
        """Preprocess the input data representation

        Args:
            input (DataFrame): the raw input data presentation

        Returns:
            DataFrame: the preprocessed raw input data presentation
        """

        # Take the log of the skewed columns in the input
        result = self.log_columns(input, self.SKEWED_COLUMNS)

        return result

    def __binary_mapping(self, x: float) -> int:
        """A helper function to map a non-negative number
        to 1 and a negative number to 0

        Args:
            x (float): the number

        Returns:
            int: `1` when x >= 0, `0` when x < 0
        """

        if x >= 0:
            return 1
        else:
            return 0

    @property
    def preprocessed_inputs(self) -> DataFrame:
        """Get a copy of all preprocessed inputs

        Returns:
            DataFrame: the preprocessed inputs
        """
        return self.__pre_processed_inputs

    @property
    def inputs(self) -> ndarray:
        """Get a copy of all derived inputs

        Returns:
            ndarray: the derived inputs in ndarray
        """

        return self.__derived_inputs.to_numpy()

    @property
    def targets(self) -> ndarray:
        """Get a copy of all derived targets

        Returns:
            ndarray: the derived targets in ndarray
        """

        return self.__derived_targets.to_numpy()

    def __min_max_normalisation(self, inputs: ndarray) -> ndarray:
        """Normalise the inputs with min-max normalisation.

        Args:
            inputs (ndarray): N*D input data matrix

        Returns:
            ndarray: N*D min-max-normalised data matrix
        """
        return (inputs - inputs.min()) / (inputs.max() - inputs.min())

    def train_test_split(
        self, train_filter: ndarray, test_filter: ndarray
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Split the countries with given train and test filters
        and pair the countries.

        Args:
            train_filter (ndarray): a binary 1-d array of countries.
                A country is in training set if it has value `True`
            test_filter (ndarray): a binary 1-d array of countries.
                A country is in testing set if it has value `True`

        Returns:
            Tuple[ndarray, ndarray, ndarray, ndarray]:
                (train_inputs, train_targets, test_inputs, test_targets)
        """

        raw_train_inputs = self.__pre_processed_inputs[train_filter]
        raw_test_inputs = self.__pre_processed_inputs[test_filter]

        raw_train_targets = self.__pre_processed_targets[train_filter]
        raw_test_targets = self.__pre_processed_targets[test_filter]

        # Pair the inputs and targets
        derived_train_inputs, derived_train_targets = self.pair(
            raw_train_inputs, raw_train_targets
        )
        derived_test_inputs, derived_test_targets = self.pair(
            raw_test_inputs, raw_test_targets
        )

        return (
            self.__min_max_normalisation(derived_train_inputs.to_numpy()),
            derived_train_targets.to_numpy(),
            self.__min_max_normalisation(derived_test_inputs.to_numpy()),
            derived_test_targets.to_numpy(),
        )
