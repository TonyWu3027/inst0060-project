from datetime import datetime
from os import PathLike
from typing import List, Union

import pandas as pd
from pandas import DataFrame


def load_covid_frame(
    file_path: Union[PathLike, str], columns: List[str] = []
) -> DataFrame:
    """Load required columns in the "Our World In Data"
    COVID-19 dataset as a DataFrame. If no column is given,
    all columns in the original dataset will be loaded.

    Args:
        file_path (PathLike | str): the file path to the CSV dataset
        columns (List[str], optional): the required columns. Defaults to [].

    Returns:
        DataFrame: the resultant DataFrame
    """

    covid_frame = pd.read_csv(file_path)

    # Select required columns from the raw CSV
    if columns:
        covid_frame = covid_frame[columns]

    return covid_frame


def join_auxiliary_dataset(
    input: DataFrame,
    auxiliary: Union[str, PathLike, DataFrame],
    columns: List[str],
    aux_index_col: str = "",
    na_value: str = "",
    is_numerical: bool = True,
) -> DataFrame:
    """Join the required columns in an auxiliary dataset
    to the original DataFrame with empty entries dropped

    Note:
        the auxiliary dataset needs to be indexed with ISO country code

    Args:
        input (DataFrame): the input DataFrame, indexed with ISO country code
        auxiliary (str | DataFrame | PathLike): the path to or the DataFrame of the auxiliary dataset.
        Should the DataFrame be given, it should be indexed with `aux_index_col`
        columns (List[str]): the required columns in the auxiliary dataset
        aux_index_col (str): Default to "". the index column in the auxiliary DataFrame
        na_value (str): Default to "". The representation for NaN value in the auxiliary dataset
        is_numerical (bool): Default to True. Whether the data is numerical or not
    Raises:
        ValueError: raised if `columns` is empty

    Returns:
        DataFrame: the resultant dataset
    """

    if not columns:
        raise ValueError("Required column(s) in the auxiliary dataset is not specified")

    if isinstance(auxiliary, (str, PathLike)):
        aux_frame = pd.read_csv(auxiliary).set_index(aux_index_col)
    else:
        aux_frame = auxiliary

    # Read the required columns in the auxiliary CSV with given index
    aux_frame = aux_frame[columns]

    # Replace the null values with NaN and drop the rows
    aux_frame = aux_frame.replace(na_value, pd.NA).dropna()

    # If data is numerical, convert to float 65
    if is_numerical:
        aux_frame = aux_frame.astype(float)

    # Join the aux_frame on ISO and drop empty row entries
    result = input.join(aux_frame).dropna()

    return result


def average_over_date_range(
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
        date_format (str): Defualt to "%Y-%m-%d". The format of date in the dataset
        is_covid (bool): Default to False. Whether the input is the OWID COVID-19 Dataset

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


if __name__ == "__main__":
    from preprocessing import *

    COVID_COLUMNS = [
        "iso_code",
        "date",
        "continent",
        "new_cases_per_million",
        "new_deaths_per_million",
        "new_vaccinations_smoothed_per_million",
        "new_people_vaccinated_smoothed_per_hundred",
        "population",
        "life_expectancy",
    ]

    raw = load_covid_frame("./covid.csv", COVID_COLUMNS)

    START = datetime(2021, 10, 26)
    END = datetime(2021, 11, 26)

    country_raw = average_over_date_range(
        raw, "iso_code", "date", START, END, is_covid=True
    )

    # Join GDP per Capita
    country_raw = join_auxiliary_dataset(
        country_raw,
        "./data/gdp_per_capita_wb.csv",
        ["gdp_per_capita"],
        "Country Code",
        "..",
    )

    # Join Population ages 65+
    country_raw = join_auxiliary_dataset(
        country_raw,
        "./data/65_above_share_wb.csv",
        ["population_ages_65_and_above"],
        "Country Code",
        "..",
    )

    # Join Population Density
    country_raw = join_auxiliary_dataset(
        country_raw,
        "./data/pop_den_wb.csv",
        ["population_density"],
        "Country Code",
        "..",
    )

    # Join Stringency index attributes
    stringency_raw = pd.read_csv("./data/stringency.csv")
    stringency_daily_avg = average_over_date_range(
        stringency_raw, "CountryCode", "Date", START, END, date_format="%Y%m%d"
    )
    country_raw = join_auxiliary_dataset(
        country_raw,
        stringency_daily_avg,
        [
            "C1_School closing",
            "C2_Workplace closing",
            "C3_Cancel public events",
            "C4_Restrictions on gatherings",
            "C5_Close public transport",
            "C6_Stay at home requirements",
            "C7_Restrictions on internal movement",
            "C8_International travel controls",
        ],
    )

    print(country_raw)
    print(country_raw.dtypes)
