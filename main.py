import argparse
import os
from argparse import ArgumentParser

from covidcomp.data import RawRepresentation


def is_valid_file_path(parser: ArgumentParser, file_path: str) -> str:
    """Determine whether a file_path is valid or not.
    If not, an ArgumentParser error will be promopted.

    Args:
        parser (ArgumentParser): the ArgumentParser
        file_path (str): the given file path

    Returns:
        str: the valid file_path
    """
    if not os.path.exists(file_path):
        parser.error(f"The file {file_path} does not exist")
    else:
        return file_path


def main():
    parser = argparse.ArgumentParser(description="Train the COVID-19 comparison model.")
    parser.add_argument(
        "file_path",
        type=lambda x: is_valid_file_path(parser, x),
        help="the file path to the OWID COVID-19 dataset",
    )

    args = parser.parse_args()
    file_path = args.file_path

    raw = RawRepresentation(file_path)
    raw.export_csv()


if __name__ == "__main__":
    main()
