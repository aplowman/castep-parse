"""`castep_parse.utils.py`"""

import re
import os
from io import TextIOWrapper
from pathlib import Path

import mendeleev
import numpy as np


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


def map_species_to_castep(species, species_idx):
    """Generate an index array that maps a species index array in the same way that CASTEP
    internally reorders atoms.

    Parameters
    ----------
    species : ndarray of shape (N, ) of str
        Unique chemical symbols indexed by `species_idx`.
    species_idx : ndarray of shape (M, ) of int
        Indices into `species` such that `species[species_idx]` produces an array of all
        atom species in the system.

    Returns
    -------
    map_idx : ndarray of shape (M, ) of int
        Index array that reorders `species_idx`

    Notes
    -----
    CASTEP orders atoms first by atomic number (proton number) and then by their original
    order. This function can be used to reorder atom coordinates in this way.

    """

    atom_z = [mendeleev.element(i).atomic_number for i in species]
    atom_z_srt_idx = np.argsort(atom_z)

    map_unsort = []
    for i in range(len(species)):
        map_unsort.append(np.where(species_idx == i)[0])

    map_sort = []
    for i in atom_z_srt_idx:
        map_sort.append(map_unsort[i])

    map_idx = np.concatenate(map_sort)

    return map_idx


def get_int_arr(arr):
    """Take a list or array and return an int array.

    Parameters
    ----------
    arr : list of ndarray

    Returns
    -------
    arr_int : ndarray of int

    """

    if isinstance(arr, list):
        arr = np.array(arr)

    if isinstance(arr, np.ndarray):

        if np.allclose(np.mod(arr, 1), 0):
            arr_int = arr.astype(int)

        else:
            raise ValueError('`arr` cannot be parsed as an int array.')

    else:
        raise ValueError('`arr` is not a list or array.')

    return arr_int


def format_args_check(**kwargs):
    """
    Check types of parameters used in `format_arr`.

    """

    if 'depth' in kwargs and not isinstance(kwargs['depth'], int):
        raise ValueError('`depth` must be an integer.')

    if 'indent' in kwargs and not isinstance(kwargs['indent'], str):
        raise ValueError('`indent` must be a string.')

    if 'col_delim' in kwargs and not isinstance(kwargs['col_delim'], str):
        raise ValueError('`col_delim` must be a string.')

    if 'row_delim' in kwargs and not isinstance(kwargs['row_delim'], str):
        raise ValueError('`row_delim` must be a string.')

    if 'dim_delim' in kwargs and not isinstance(kwargs['dim_delim'], str):
        raise ValueError('`dim_delim` must be a string.')

    if 'format_spec' in kwargs and not isinstance(kwargs['format_spec'],
                                                  (str, list)):
        raise ValueError('`format_spec` must be a string or list of strings.')

    if 'assign' in kwargs:

        if not isinstance(kwargs['assign'], str):
            raise ValueError('`assign` must be a string.')


def format_arr(arr, depth=0, indent='\t', col_delim='\t', row_delim='\n',
               dim_delim='\n', format_spec='{}'):
    """Get a string representation of a Numpy array, formatted with indents.

    Parameters
    ----------
    arr : ndarray or list of ndarray
        Array of any shape to format as a string, or list of arrays whose
        shapes match except for the final dimension, in which case the arrays
        will be formatted horizontally next to each other.
    depth : int, optional
        The indent depth at which to begin the formatting.
    indent : str, optional
        The string used as the indent. The string which indents each line of
        the array is equal to (`indent` * `depth`).
    col_delim : str, optional
        String to delimit columns (the innermost dimension of the array).
        Default is tab character, \t.
    row_delim : str, optional
        String to delimit rows (the second-innermost dimension of the array).
        Defautl is newline character, \n.
    dim_delim : str, optional
        String to delimit outer dimensions. Default is newline character, \n.
    format_spec : str or list of str, optional
        Format specifier for the array or a list of format specifiers, one for 
        each array listed in `arr`.

    Returns
    -------
    out : str
        The array formatted as a string.

    """

    # Validation:
    format_args_check(depth=depth, indent=indent, col_delim=col_delim,
                      row_delim=row_delim, dim_delim=dim_delim,
                      format_spec=format_spec)

    if isinstance(arr, np.ndarray):
        arr = [arr]

    out_shape = list(set([i.shape[:-1] for i in arr]))

    if len(out_shape) > 1:
        msg = ('Array shapes must be identical apart from the innermost '
               'dimension.')
        raise ValueError(msg)

    if not isinstance(arr, (list, np.ndarray)):
        msg = ('Cannot format as array, object is not an array or list of '
               'arrays: type is {}')
        raise ValueError(msg.format(type(arr)))

    if isinstance(format_spec, str):
        format_spec = [format_spec] * len(arr)

    elif isinstance(format_spec, list):
        fs_err_msg = ('`format_spec` must be a string or list of N strings '
                      'where N is the number of arrays specified in `arr`.')
        if not all([isinstance(i, str)
                    for i in format_spec]) or len(format_spec) != len(arr):
            raise ValueError(fs_err_msg)

    arr_list = arr
    out = ''
    dim_seps = ''
    d = arr_list[0].ndim

    if d == 1:
        out += (indent * depth)

        for sa_idx, sub_arr in enumerate(arr_list):
            for col_idx, col in enumerate(sub_arr):
                out += format_spec[sa_idx].format(col)
                if (col_idx < len(sub_arr) - 1):
                    out += col_delim

        out += row_delim

    else:

        if d > 2:
            dim_seps = dim_delim * (d - 2)

        sub_arr = []
        for i in range(out_shape[0][0]):

            sub_arr_lst = []
            for j in arr_list:
                sub_arr_lst.append(j[i])

            sub_arr.append(format_arr(sub_arr_lst, depth, indent, col_delim,
                                      row_delim, dim_delim, format_spec))

        out = dim_seps.join(sub_arr)

    return out


def get_castep_cell_constraints(lengths_equal, angles_equal, fix_lengths,
                                fix_angles):
    """Get CASTEP cell constraints encoded from a set of more user-friendly
    parameters.

    Parameters
    ----------
    lengths_equal : str
        Some combination of 'a', 'b' and 'c', or 'none'. Represents which
        supercell vectors are to remain equal to one another.
    angles_equal : str
        Some combination of 'a', 'b' and 'c', or 'none'. Represents which
        supercell angles are to remain equal to one another.
    fix_lengths : str
        Some combination of 'a', 'b' and 'c', or 'none'. Represents which
        supercell vectors are to remain fixed.
    fix_angles : str
        Some combination of 'a', 'b' and 'c', or 'none'. Represents which
        supercell angles are to remain fixed.

    Returns
    -------
    list of list of int

    TODO: Improve robustness; should return exception for some cases where it
    currently is not: E.g. angles_equal = bc; fix_angles = ab should not be
    allowed.

    """

    if lengths_equal is None:
        lengths_equal = ''

    if angles_equal is None:
        angles_equal = ''

    if fix_lengths is None:
        fix_lengths = ''

    if fix_angles is None:
        fix_angles = ''

    eqs_params = [lengths_equal, angles_equal]
    fxd_params = [fix_lengths, fix_angles]
    encoded = [[1, 2, 3], [1, 2, 3]]

    ambig_cell_msg = 'Ambiguous cell constraints specified.'

    # Loop through lengths, then angles:
    for idx, (fp, ep) in enumerate(zip(fxd_params, eqs_params)):

        if len(fp) == 3 and all([x in fp for x in ['a', 'b', 'c']]):
            encoded[idx] = [0, 0, 0]

        elif len(fp) == 2 and all([x in fp for x in ['a', 'b']]):
            encoded[idx] = [0, 0, 3]

        elif len(fp) == 2 and all([x in fp for x in ['a', 'c']]):
            encoded[idx] = [0, 2, 0]

        elif len(fp) == 2 and all([x in fp for x in ['b', 'c']]):
            encoded[idx] = [1, 0, 0]

        elif fp == 'a':

            if ep == '':
                encoded[idx] = [0, 2, 3]

            elif ep == 'bc':
                encoded[idx] = [0, 2, 2]

            else:
                raise ValueError(ambig_cell_msg)

        elif fp == 'b':

            if ep == '':
                encoded[idx] = [1, 0, 3]

            elif ep == 'ac':
                encoded[idx] = [1, 0, 1]

            else:
                raise ValueError(ambig_cell_msg)

        elif fp == 'c':

            if ep == '':
                encoded[idx] = [1, 2, 0]

            elif ep == 'ab':
                encoded[idx] = [1, 1, 0]

            else:
                raise ValueError(ambig_cell_msg)

        else:

            if len(ep) == 3 and all([x in ep for x in ['a', 'b', 'c']]):
                encoded[idx] = [1, 1, 1]

            elif any([x in ep for x in ['ab', 'ba']]):
                encoded[idx] = [1, 1, 3]

            elif any([x in ep for x in ['ac', 'ca']]):
                encoded[idx] = [1, 2, 1]

            elif any([x in ep for x in ['bc', 'cb']]):
                encoded[idx] = [1, 2, 2]

    for idx in range(len(encoded[1])):
        if encoded[1][idx] > 0:
            encoded[1][idx] += 3

    return encoded


def flexible_open(func):
    'Decorator for functions that may take a path or file handle.'
    def decorated(path_or_bytes, *args, **kwargs):

        if isinstance(path_or_bytes, (str, Path)):
            path_or_bytes = Path(path_or_bytes)
            with path_or_bytes.open('r') as handle:
                lines = handle.read().splitlines()

        elif isinstance(path_or_bytes, bytes):
            lines = path_or_bytes.decode('utf-8').split('\n')

        elif isinstance(path_or_bytes, TextIOWrapper):
            lines = path_or_bytes.read().splitlines()

        else:
            raise ValueError

        return func(lines, *args, **kwargs)

    return decorated


def array_nan_equal(a, b):
    'Check is two arrays are equal, where some elements may be NaNs.'
    nonan_a = ~np.isnan(a)
    nonan_b = ~np.isnan(b)
    if not np.array_equal(nonan_a, nonan_b):
        return False
    else:
        return np.allclose(a[nonan_a], b[nonan_a])


def merge_str_list(lst, merge_idx_start, merge_idx_end):
    return (
        lst[:merge_idx_start] +
        [''.join(lst[merge_idx_start: merge_idx_end])] +
        lst[merge_idx_end:]
    )
