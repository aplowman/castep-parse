"""`castep_parse.writers.py`"""

from pathlib import Path
import itertools

import numpy as np

from castep_parse.utils import (get_int_arr, format_arr,
                                get_castep_cell_constraints)

__all__ = [
    'write_cell_file',
    'write_input_files',
]


def write_cell_file(supercell, atom_sites, species, species_idx, dir_path,
                    seedname, cell, sym_ops, cell_constraints,
                    atom_constraints, is_geom_opt):
    """Write a CASTEP cell file.

    Parameters
    ----------
    supercell : ndarray of shape (3, 3)
        Array of column vectors representing the edge vectors of the supercell.
    atom_sites : ndarray of shape (3, N)
        Array of column vectors representing the positions of each atom.
    species : ndarray of str of shape (M, )
        The distinct species of the atoms.
    species_idx : list or ndarray of shape (N, )
        Maps each atom site to a given species in `species`.
    dir_path : str or Path
        Directory in which to generate input files.
    seedname : str, optional
        The seedname of the CASTEP calculation. Default is `sim`.
    cell : dict, optional
        Key value pairs to add to the cell file.
    sym_ops: list of ndarray of shape (4, 3)
        Each array represents a symmetry operation, where the first three rows
        are the rotation matrix and the final row is the translation.
    cell_constraints : dict, optional
        A dict with the following keys:
            lengths_equal : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell vectors are to remain equal to one another.
            angles_equal : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell angles are to remain equal to one another.
            fix_lengths : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell vectors are to remain fixed.
            fix_angles : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell angles are to remain fixed.
    atom_constraints : dict, optional
        A dict with the following keys:
            fix_`mn`_idx : ndarray of dimension 1
                The atom indices whose `m` and `n` coordinates are to
                be fixed, where valid pairs of `mn` are (`xy`, `xz`, `yz`).
                By default, set to None.
            fix_xyz_idx : ndarray of dimension 1
                The atom indices whose `x`, `y` and `z` coordinates
                are to be fixed. By default, set to None.    
    is_geom_opt : bool
        Sets whether the simulation to be run is a geometry optimisation.

    Returns
    -------
    cell_path : Path
        Path to the newly generated cell file.

    """

    # Prepare atom constraints:
    if atom_constraints is not None:
        for k, v in atom_constraints.items():

            if isinstance(v, (np.ndarray, list)):

                atom_constraints[k] = get_int_arr(v)
                v = atom_constraints[k]

                if v.ndim != 1:
                    msg = ('`atom_constraints[{}]` must be a 1D list, 1D array'
                           ' or str.'.format(k))
                    raise ValueError(msg)

                if v.min() < 1 or v.max() > atom_sites.shape[1]:
                    msg = '`atom_constraints[{}]` must index `atom_sites`'
                    raise IndexError(msg.format(k))

            elif v is not None:
                msg = '`atom_constraints[{}]` must be a 1D list or 1D array.'
                raise ValueError(msg.format(k))

        f_xy = atom_constraints.get('fix_xy_idx')
        f_xz = atom_constraints.get('fix_xz_idx')
        f_yz = atom_constraints.get('fix_yz_idx')
        f_xyz = atom_constraints.get('fix_xyz_idx')

        if f_xy is None:
            f_xy = np.array([])
        if f_xz is None:
            f_xz = np.array([])
        if f_yz is None:
            f_yz = np.array([])
        if f_xyz is None:
            f_xyz = np.array([])

        atom_constr_opt = [f_xy, f_xz, f_yz, f_xyz]
        atom_constr_pairs = list(itertools.combinations(atom_constr_opt, 2))

        for pair in atom_constr_pairs:
            if len(pair[0]) > 0 and len(pair[1]) > 0:
                if len(np.intersect1d(pair[0], pair[1])) > 0:
                    msg = ('`{}_idx` and `{}_idx`  cannot contain the same '
                           'indices.'.format(pair[0], pair[1]))
                    raise ValueError(msg)

    # Write cell file:
    dir_path = Path(dir_path)
    cell_path = dir_path.joinpath(seedname + '.cell')

    with cell_path.open('w') as handle:  # TODO: universal newline mode here?

        # Supercell (need to transpose to array of row vectors):
        handle.write('%block lattice_cart\n')
        handle.write(format_arr(supercell.T,
                                format_spec='{:24.15f}',
                                col_delim=' '))
        handle.write('%endblock lattice_cart\n\n')

        # Atoms (need to transpose to array of row vectors):
        atom_species = species[species_idx][:, np.newaxis]

        handle.write('%block positions_abs\n')
        handle.write(format_arr([atom_species, atom_sites.T],
                                format_spec=['{:5}', '{:24.15f}'],
                                col_delim=' '))
        handle.write('%endblock positions_abs\n')

        # Cell constraints:
        if cell_constraints is not None:
            encoded_params = get_castep_cell_constraints(**cell_constraints)

            if (not (encoded_params[0] == [1, 2, 3] and
                     encoded_params[1] == [4, 5, 6])) and is_geom_opt:

                if (encoded_params[0] == [0, 0, 0] and
                        encoded_params[1] == [0, 0, 0]):
                    handle.write('\nfix_all_cell = True\n')

                else:
                    handle.write('\n%block cell_constraints\n')
                    handle.write('{}\t{}\t{}\n'.format(*encoded_params[0]))
                    handle.write('{}\t{}\t{}\n'.format(*encoded_params[1]))
                    handle.write('%endblock cell_constraints\n')

        # Atom constraints:
        if atom_constraints is not None:
            if any([len(x) for x in atom_constr_opt]) > 0:

                # For each atom, get the index within like-species atoms:
                # 1-based indexing instead of 0-based!
                sub_idx = np.zeros((atom_sites.shape[1]), dtype=int) - 1
                for sp_idx in range(len(species)):
                    w = np.where(species_idx == sp_idx)[0]
                    sub_idx[w] = np.arange(w.shape[0]) + 1

                cnst_fs = ['{:<5d}', '{:<5}', '{:<5d}', '{:24.15f}']
                handle.write('\n%block ionic_constraints\n')

                nc_xyz = f_xyz.shape[0]
                nc_xy = f_xy.shape[0]
                nc_xz = f_xz.shape[0]
                nc_yz = f_yz.shape[0]

                if nc_xyz > 0:
                    f_xyz -= 1
                    f_xyz_sp = np.tile(
                        atom_species[f_xyz], (1, 3)).reshape(nc_xyz * 3, 1)
                    f_xyz_sub_idx = np.repeat(sub_idx[f_xyz], 3)[:, np.newaxis]
                    f_xyz_cnst_idx = (np.arange(nc_xyz * 3) + 1)[:, np.newaxis]
                    f_xyz_cnst_coef = np.tile(np.eye(3), (nc_xyz, 1))

                    cnst_arrs_xyz = [f_xyz_cnst_idx, f_xyz_sp, f_xyz_sub_idx,
                                     f_xyz_cnst_coef]

                    handle.write(format_arr(cnst_arrs_xyz,
                                            format_spec=cnst_fs,
                                            col_delim=' '))

                if nc_xy > 0:
                    f_xy -= 1
                    f_xy_sp = np.tile(
                        atom_species[f_xy], (1, 2)).reshape(nc_xy * 2, 1)
                    f_xy_sub_idx = np.repeat(sub_idx[f_xy], 2)[:, np.newaxis]
                    f_xy_cnst_idx = (np.arange(nc_xy * 2) + 1 +
                                     (nc_xyz * 3))[:, np.newaxis]
                    f_xy_cnst_coef = np.tile(np.eye(3)[[0, 1]], (nc_xy, 1))

                    cnst_arrs_xy = [f_xy_cnst_idx, f_xy_sp, f_xy_sub_idx,
                                    f_xy_cnst_coef]

                    handle.write(format_arr(cnst_arrs_xy,
                                            format_spec=cnst_fs,
                                            col_delim=' '))

                if nc_xz > 0:
                    f_xz -= 1
                    f_xz_sp = np.tile(
                        atom_species[f_xz], (1, 2)).reshape(nc_xz * 2, 1)
                    f_xz_sub_idx = np.repeat(sub_idx[f_xz], 2)[:, np.newaxis]
                    f_xz_cnst_idx = (np.arange(nc_xz * 2) + 1 +
                                     (nc_xy * 2) + (nc_xyz * 3))[:, np.newaxis]
                    f_xz_cnst_coef = np.tile(np.eye(3)[[0, 2]], (nc_xz, 1))

                    cnst_arrs_xz = [f_xz_cnst_idx, f_xz_sp, f_xz_sub_idx,
                                    f_xz_cnst_coef]

                    handle.write(format_arr(cnst_arrs_xz,
                                            format_spec=cnst_fs,
                                            col_delim=' '))

                if nc_yz > 0:
                    f_yz -= 1
                    f_yz_sp = np.tile(
                        atom_species[f_yz], (1, 2)).reshape(nc_yz * 2, 1)
                    f_yz_sub_idx = np.repeat(sub_idx[f_yz], 2)[:, np.newaxis]
                    f_yz_cnst_idx = (np.arange(nc_yz * 2) + 1 + (nc_xz * 2) +
                                     (nc_xy * 2) + (nc_xyz * 3))[:, np.newaxis]

                    f_yz_cnst_coef = np.tile(np.eye(3)[[1, 2]], (nc_yz, 1))

                    cnst_arrs_yz = [f_yz_cnst_idx, f_yz_sp, f_yz_sub_idx,
                                    f_yz_cnst_coef]

                    handle.write(format_arr(cnst_arrs_yz,
                                            format_spec=cnst_fs,
                                            col_delim=' '))

                handle.write('%endblock ionic_constraints\n')

        # Symmetry ops
        if sym_ops is not None:

            sym_ops = np.vstack(sym_ops)
            handle.write('\n%block symmetry_ops\n')
            handle.write(format_arr(sym_ops,
                                    format_spec='{:24.15f}',
                                    col_delim=' '))
            handle.write('%endblock symmetry_ops\n')

        # Other cell file items:
        if cell is not None:
            handle.write('\n')
            for k, v in sorted(cell.items()):
                handle.write('{:25s}= {}\n'.format(k, v))

    return cell_path


def write_input_files(supercell, atom_sites, species, species_idx, dir_path,
                      seedname='sim', cell=None, param=None, sym_ops=None,
                      cell_constraints=None, atom_constraints=None):
    """Generate CASTEP input files.

    Parameters
    ----------
    supercell : ndarray of shape (3, 3)
        Array of column vectors representing the edge vectors of the supercell.
    atom_sites : ndarray of shape (3, N)
        Array of column vectors representing the positions of each atom.
    species : ndarray of str of shape (M, )
        The distinct species of the atoms.
    species_idx : list or ndarray of shape (N, )
        Maps each atom site to a given species in `species`.
    dir_path : str or Path
        Directory in which to generate input files. It will be generated if it
        does not exist.
    seedname : str, optional
        The seedname of the CASTEP calculation. Default is `sim`.
    cell : dict, optional
        Key value pairs to add to the cell file.
    param : dict, optional
        Key value pairs to add to the param file.
    sym_ops: list of ndarray of shape (4, 3)
        Each array represents a symmetry operation, where the first three rows
        are the rotation matrix and the final row is the translation.
    cell_constraints : dict, optional
        A dict with the following keys:
            lengths_equal : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell vectors are to remain equal to one another.
            angles_equal : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell angles are to remain equal to one another.
            fix_lengths : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell vectors are to remain fixed.
            fix_angles : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell angles are to remain fixed.
    atom_constraints : dict, optional
        A dict with the following keys:
            fix_`mn`_idx : ndarray of dimension 1
                The atom indices whose `m` and `n` coordinates are to
                be fixed, where valid pairs of `mn` are (`xy`, `xz`, `yz`).
                By default, set to None.
            fix_xyz_idx : ndarray of dimension 1
                The atom indices whose `x`, `y` and `z` coordinates
                are to be fixed. By default, set to None.

    Returns
    -------
    input_file_paths : list of Path
        List of the file paths to all newly created input files.

    TODO: Generalise atom constraints.

    """

    # Validation:

    species_idx = get_int_arr(species_idx)
    if species_idx.min() < 0 or species_idx.max() > (atom_sites.shape[1] - 1):
        raise IndexError('`species_idx` must index `atom_sites`')

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        dir_path.mkdir()

    input_file_paths = []
    task = 'SINGLEPOINT'

    # Write param file:
    if param is not None:
        task = param.get('task', task).upper()
        param_path = dir_path.joinpath(seedname + '.param')
        input_file_paths.append(param_path)
        with param_path.open('w') as handle:
            for k, v in sorted(param.items()):
                handle.write('{:25s}= {}\n'.format(k, v))

    geom_opt_str = ['GEOMETRYOPTIMISATION', 'GEOMETRYOPTIMIZATION']
    is_geom_opt = task in geom_opt_str

    # Write cell file:
    cell_path = write_cell_file(supercell, atom_sites, species, species_idx,
                                dir_path, seedname, cell, sym_ops,
                                cell_constraints, atom_constraints,
                                is_geom_opt)

    input_file_paths.append(cell_path)

    return input_file_paths
