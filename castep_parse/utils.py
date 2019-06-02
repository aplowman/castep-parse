"""`castep_parse.utils.py`"""

import re
import os


def find_files_in_dir(dir_path, match_regex, recursive=False):
    """Return relative paths of file names which  match a regular expression.

    Parameters
    ----------
    dir_path : str
        Path of directory in which to search for files.
    match_regex : str
        Regular expression to match files against.
    recursive : bool, optional
        If True, search all subdirectories. By default, set to False.

    Returns
    -------
    matched_files : list of str
        List of file paths (relative to `dir_path`?) that match the given
        regular expression.

    """

    matched_files = []

    for idx, (root, _, files) in enumerate(os.walk(dir_path)):

        for f in files:

            if re.search(match_regex, f):

                if idx == 0:
                    matched_path = f

                else:
                    pth = os.path.relpath(root, dir_path).split(os.path.sep)
                    matched_path = os.path.join(*pth, f)

                matched_files.append(matched_path)

        if not recursive:
            break

    return matched_files
