"""`castep_parse.utils.py`"""

import re
import os

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
    """Generate an index array that maps a species index array in the same way 
    that CASTEP internally reorders atoms.

    Parameters
    ----------
    species : ndarray of shape (N, ) of str
        Unique chemical symbols indexed by `species_idx`.
    species_idx : ndarray of shape (M, ) of int
        Indices into `species` such that `species[species_idx]` produces an
        array of all atom species in the system.

    Returns
    -------
    map_idx : ndarray of shape (M, ) of int
        Index array which reorders `species_idx`

    Notes
    -----
    CASTEP orders atoms first by atomic number (proton number) and then by 
    their original order. This function can be used to reorder atom coordinates
    in this way.

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
