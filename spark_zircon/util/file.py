"""Module contains functions for working with files."""

import os


Path_to_file = str


def get_absolute_file_path(file_path: Path_to_file) -> Path_to_file:
    """To get absolute path for a given filename.

    :param file_path: Path to the file to be opened
    :type file_path: Path_to_file
    :return: Absolute path to the file to open
    :rtype: Path_to_file
    """
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, file_path)
