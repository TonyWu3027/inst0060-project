import os
from datetime import datetime
from os import PathLike
from typing import Dict, List, Tuple, Union

import pandas as pd
from pandas import DataFrame, Series

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
        "total_vaccinations_per_hundred",
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
        # TODO: general solution for all categorical columns
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

    def get_flat_representation(self) -> Tuple[DataFrame, Series]:
        """Get the flat raw representation of the dataset

        Returns:
            Tuple[DataFrame, Series]: the flat input and the flat target
        """

        raw_input = self.__covid_frame.drop(
            columns=[*self.PARTITION_COLS, self.TARGET_COL]
        )
        raw_target = self.__covid_frame[self.TARGET_COL]

        return raw_input, raw_target

    def get_partitioned_representation(
        self, partition_col: str
    ) -> Dict[str, Tuple[DataFrame, Series]]:
        """Get a raw representation of the dataset partitioned
        by the specified column

        Args:
            partition_col (str): the partitioning column
            (e.g. `"continent"`, `"income_group"`)

        Raises:
            ValueError: raised when `partition_col` is not a categorical column

        Returns:
            Dict[str, Tuple[DataFrame, Series]]: {category: (input, target)}.
            For instance, when partitioned by continent, the output might look like:
            `{"Asia": (raw_input_for_asia, raw_target_for_asia), "Europe": (..), ...}`
        """

        if partition_col not in self.PARTITION_COLS:
            raise ValueError("The partitioning column is invalid")

        # Get a list of categories with the partitioning method
        categories = self.__covid_frame[partition_col].unique().tolist()

        result = dict()

        for category in categories:
            partitioned_frame = self.__covid_frame.loc[
                self.__covid_frame[partition_col] == category
            ]
            partitioned_input = partitioned_frame.drop(
                columns=[*self.PARTITION_COLS, self.TARGET_COL]
            )
            partitioned_target = partitioned_frame[self.TARGET_COL]
            result[category] = (partitioned_input, partitioned_target)

        return result
