from datetime import datetime
from os import PathLike
from typing import List

import pandas as pd
import numpy as np


def load_covid_frame(file_path: PathLike, columns: List[str] = []) -> pd.DataFrame:
    """Load required columns in the "Our World In Data"
    COVID-19 dataset as a DataFrame. If no column is given,
    all columns in the original dataset will be loaded.

    Args:
        file_path (PathLike): the file path to the CSV dataset
        columns (List[str], optional): the required columns. Defaults to [].

    Returns:
        pd.DataFrame: the resultant DataFrame
    """

    covid_frame = pd.read_csv(file_path)

    # Select required columns from the raw CSV
    if columns:
        covid_frame = covid_frame[columns]

    return covid_frame


def join_auxiliary_dataset(input: pd.DataFrame, file_path: PathLike, columns: List[str], aux_index_col: str, na_value: str = "", is_numerical:bool = True) -> pd.DataFrame:
    """Join the required columns in an auxiliary dataset
    to the original DataFrame with empty entries dropped

    Note:
        the auxiliary dataset needs to be indexed with ISO country code

    Args:
        input (pd.DataFrame): the input DataFrame, indexed with ISO country code
        file_path (PathLike): the file path to the auxiliary CSV dataset
        columns (List[str]): the required columns in the auxiliary dataset
        aux_index_col (str): the index column in the auxiliary DataFrame
        na_value (str): Default to "". The representation for NaN value in the auxiliary dataset
        is_numerical (bool): Default to True. Whether the data is numerical or not
    Raises:
        ValueError: raised if `columns` is empty

    Returns:
        pd.DataFrame: the resultant dataset
    """

    if not columns:
        raise ValueError(
            "Required column(s) in the auxiliary dataset is not specified")

    # Read the required columns in the auxiliary CSV with given index
    aux_frame = pd.read_csv(file_path).set_index(aux_index_col)[columns]
    
    # Replace the null values with NaN and drop the rows
    aux_frame = aux_frame.replace(na_value, pd.NA).dropna()
        
    # If data is numerical, convert to float 65
    if is_numerical:  
        aux_frame = aux_frame.astype(float)
        
    # Join the aux_frame on ISO and drop empty row entries
    result = input.join(aux_frame).dropna()

    return result


def average_over_date_range(input: pd.DataFrame, index_col: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Calculate the daily average of all data attributes in dataset
    for each country in a given period of time.

    Args:
        input (pd.DataFrame): the input DataFrame
        index_col (str): the index column in `input`
        start_date (date): start date inclusively
        end_date (date): end date inclusively

    Raises:
        ValueError: raised when `end_date` is earlier than the `start_date`
        KeyError: raised when `index_col` is not given

    Returns:
        pd.DataFrame: the resultant DataFrame, indexed with `index_col`
    """

    if end_date < start_date:
        raise ValueError(
            "The end date should not be earlier than the start date")

    if not index_col:
        raise KeyError("Index columns not given")

    # Select the date within the range
    input['date'] = pd.to_datetime(input['date'])
    mask = (input['date'] > start_date) & (input['date'] < end_date)
    result = input.loc[mask]

    # Take the daily average for each country and set index column
    result = result.groupby([index_col, 'continent'], as_index=False).mean()
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
        "life_expectancy"
    ]

    raw = load_covid_frame("./covid.csv", COVID_COLUMNS)

    START = datetime(2021, 10, 26)
    END = datetime(2021, 11, 26)

    country_raw = average_over_date_range(raw, 'iso_code', START, END)

    # Join GDP per Capita
    country_raw = join_auxiliary_dataset(
        country_raw, './data/gdp_per_capita_wb.csv', ["gdp_per_capita"], "Country Code", "..")

    # Join Population ages 65+
    country_raw = join_auxiliary_dataset(country_raw, './data/65_above_share_wb.csv', [
                                            "population_ages_65_and_above"], "Country Code", "..")

    # Join Population Density
    country_raw = join_auxiliary_dataset(
        country_raw, './data/pop_den_wb.csv', ["population_density"], "Country Code", "..")

    print(country_raw)
    print(country_raw.dtypes)
