from os import PathLike
import pandas as pd
from typing import List
from datetime import datetime


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


def join_auxiliary_dataset(input: pd.DataFrame, file_path: PathLike, columns: List[str], input_index_col: str, aux_index_col: str) -> pd.DataFrame:
    """Join the required columns in an auxiliary dataset
    to the original DataFrame with empty entries dropped

    Note:
        the auxiliary dataset needs to be indexed with ISO country code

    Args:
        input (pd.DataFrame): the input DataFrame, indexed with ISO country code
        file_path (PathLike): the file path to the auxiliary CSV dataset
        columns (List[str]): the required columns in the auxiliary dataset
        input_index_col (str): the index column in the input DataFrame
        aux_index_col (str): the index column in the auxiliary DataFrame

    Raises:
        ValueError: raised if `columns` is empty

    Returns:
        pd.DataFrame: the resultant dataset
    """

    if not columns:
        raise ValueError(
            "Required column(s) in the auxiliary dataset is not specified")

    aux_frame = pd.read_csv(file_path)[[columns]]

    aux_frame.set_index(aux_index_col)

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
    result = result.groupby([index_col, 'continent']).mean()
    result.set_index(index_col, inplace=True)

    return result


if __name__ == "__main__":
    with open("./columns.txt") as f:
        cols = f.readlines()

        for i in range(len(cols)):
            cols[i] = cols[i].replace("\n", "")

        raw = load_covid_frame("./covid.csv", cols)

        START = datetime(2021, 10, 26)
        END = datetime(2021, 11, 26)

        country_raw = average_over_date_range(raw, 'iso_code', START, END)
        print(country_raw)

        # print(country_raw.isna().sum())
