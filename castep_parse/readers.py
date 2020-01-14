"""`castep_parse.readers.py`"""

import re
from pathlib import Path
from warnings import warn

import numpy as np

from castep_parse.utils import find_files_in_dir, flexible_open, map_species_to_castep

__all__ = [
    'read_castep_file',
    'read_geom_file',
    'read_cell_file',
    'read_output_files',
    'merge_geom_data',
    'merge_cell_data',
    'merge_output_data',
    'read_relaxation',
]


@flexible_open
def read_castep_file(path_or_file):
    """
    Parameters
    ----------
    path_or_file : str, Path, bytes or TextIOWrapper
        The main output file of a CASTEP simulation. If type is `str` or `Path`, assumed
        to be a file path.

    Notes
    -----
    The `flexible_open` decorators ensure the `path_or_file` object is converted to a list
    of strings.

    """

    # Each run header has this three times:
    pat_header = r'(\s\+-{49}\+)'

    # Want to have the whole string
    file_str = '\n'.join(path_or_file)

    header_split = re.split(pat_header, file_str)[1:]
    runs_list = [''.join(header_split[i:i + 6]) for i in range(0, len(header_split), 6)]

    runs = []
    total_time = 0
    geom_iters = []

    scf_idx = 0
    scf_all = {
        'cycles': [],
        'energies': {},
    }

    for run_idx, run_str in enumerate(runs_list):

        if 'Cell Contents' not in run_str:
            warn('Skipping run {}, looks incomplete.'.format(run_idx))
            continue

        run = parse_castep_run(run_str, run_idx)

        # Extract out SCF and SCF energies
        if 'scf' in run:

            scf = run.pop('scf')
            scf_energies = run.pop('scf_energies')

            scf_all['cycles'].extend(scf)
            for scf_ens in scf_energies:
                for en_name, en in scf_ens.items():
                    if not scf_all['energies'].get(en_name):
                        scf_all['energies'][en_name] = []
                    scf_all['energies'][en_name].append(en)

            run['SCF_idx'] = list(range(scf_idx, scf_idx + len(scf)))
            scf_idx += len(scf)

        if 'final_info' in run:
            t = run['final_info']['statistics']['total_time_s']
            total_time += t

        else:
            # Add on the last-recorded SCF time (of completed geom iterations):
            all_iters = run['geom']['iterations']
            if all_iters:
                final_step = all_iters[-1]['steps'][-1]
                if 'scf' in final_step:
                    t = final_step['scf'][-1, -1]
                    total_time += t

        if 'geom' in run:

            run_geom_iters = run['geom'].pop('iterations')

            # Extract out SCF cycles and energies from geom iteration steps
            for geom_idx, geom_iter in enumerate(run_geom_iters):

                for step in geom_iter['steps']:

                    if run_idx == 0 and geom_idx == 0:
                        step['SCF_idx'] = run['SCF_idx'][-1]

                    else:
                        scf = step.pop('scf')
                        scf_energies = step.pop('scf_energies')

                        scf_all['cycles'].append(scf)

                        for en_name, en in scf_energies.items():
                            if not scf_all['energies'].get(en_name):
                                scf_all['energies'][en_name] = []
                            scf_all['energies'][en_name].append(en)

                        step['SCF_idx'] = scf_idx
                        scf_idx += 1

            geom_iters.extend(run_geom_iters)

        runs.append(run)

    for en_name in scf_all['energies']:
        scf_all['energies'][en_name] = np.array(scf_all['energies'][en_name])

    out = {
        'runs': runs,
        'SCF': scf_all,
        'total_time_s': total_time,
    }

    if 'geom' in runs[0]:
        is_converged = bool(runs[-1]['geom']['final'])
        out.update({
            'geom': {
                'iterations': geom_iters,
                'is_converged': is_converged,
            }
        })

        if not is_converged:
            warn('Geometry not converged.')

    return out


@flexible_open
def read_geom_file(path_or_file):
    """Parse a .geom geometry trajectory file from a CASTEP geometry
    optimisation run.

    Parameters
    ----------
    path_or_file : str, Path, bytes or TextIOWrapper
        The .geom output file of a CASTEP geometry optimisation. If type is `str` or
        `Path`, assumed to be a file path.

    Returns
    -------
    geom_dat : dict
        Dictionary containing parsed data. Where Numpy arrays are returned,
        they are formed of *column* vectors.

    Notes
    -----
    The `flexible_open` decorator ensures the `path_or_file` object is converted to a list
    of strings.

    The .geom file includes the main results at the end of each BFGS step. All quantities
    in the .geom file are expressed in atomic units. Returned data is in eV (energies),
    Angstroms (ionic positions) and eV/Angstrom (ionic forces).

    Tested with CASTEP versions: 17.2, 16.11

    Structure of .geom file:

    -   HEADER, ends with "END header" line.

    -   Blocks representing the end of each BFGS step, separated by a blank line.

    -   Block line #1 (ends in "<-- c"):
        BFGS iteration number (and an indication of which quantities are
        converged in later CASTEP versions).

    -   Block line #2 (ends in "<-- E"):
        Final energy followed by final free energy.

    -   Block lines #3,4,5
        Unit cell row vectors, Cartesian coordinates, units of Bohr radius.

    -   Next three lines (if the cell is being relaxed; ends in "<-- S"):
        Stress vectors.

    -   Next n lines for n ions (starts with ion symbol and number; ends in "<-- R"):
        Ion positions, Cartesian coordinates, units of Bohr radius.

    -   Next n lines for n ions (starts with ion symbol and number; ends in "<-- F"):
        Ion forces, Cartesian coordinates, atomic units of force.

    Note: we assume that the ions are listed in the same order for ion positions and
    forces.

    I don't understand how the energy and free energy values are calculated as reported in
    the .geom file. For metallic systems, this energy seems to be closest to the "Final
    free energy (E-TS)" reported at the end of the last SCF cycle. (but not exactly the
    same (given rounding)).

    In the .castep file, the "BFGS: Final Enthalpy     =" seems to match the "Final free
    energy (E-TS)" reported at the end of the last SCF cycle (given rounding).

    """

    # As used in CASTEP v16.11 and v17.2, physical constants from:
    # [1] CODATA RECOMMENDED VALUES OF THE FUNDAMENTAL PHYSICAL CONSTANTS: 2010

    A_0_SI = 0.52917721092e-10      # m  [1]
    E_h_SI = 4.35974434e-18         # J  [1]
    E_h = 27.21138505               # eV [1]
    e_SI = 1.602176565e-19          # C  [1]
    F_SI = E_h_SI / A_0_SI          # N

    A_0 = A_0_SI * 1e10             # Angstrom
    F = F_SI * (1e-10 / e_SI)       # eV / Angstrom

    ITER_NUM = '<-- c'
    ENERGY = '<-- E'
    CELL = '<-- h'
    STRESS = '<-- S'
    POS = '<-- R'
    FORCE = '<-- F'

    bfgs_iter_idx = 0

    energies = []
    free_energies = []

    cell_ln_idx = 0
    all_cells = []
    current_cell = np.zeros((3, 3))

    stress_ln_idx = 0
    all_stresses = []
    current_stress = np.zeros((3, 3))

    all_ions = []
    all_forces = []
    current_bfgs_ions = []
    current_bfgs_forces = []

    species_set = False
    iter_num_line = False
    species = []
    species_idx = []
    iter_nums = []

    force_ln_idx = 0

    for ln in path_or_file:

        ln_s = ln.strip().split()

        # For CASTEP version 16.11, iteration number is a lone integer line
        #   - we assume there are no other lines which are a single integer
        # For CASTEP version 17.2, iteration number line has ITER_NUM in
        # line
        if len(ln_s) == 1:

            try:
                int(ln_s[0])
                iter_num_line = True

            except ValueError as e:
                pass

        if (ITER_NUM in ln) or iter_num_line:

            iter_nums.append(int(ln_s[0]))

            iter_num_line = False

            if bfgs_iter_idx > 0:

                # Save the ion positions for the previous BFGS step:
                all_ions.append(np.hstack(current_bfgs_ions))
                current_bfgs_ions = []

                all_forces.append(np.hstack(current_bfgs_forces))
                current_bfgs_forces = []

            # No need to record the species and ion indices after the first
            # BFGS block
            if bfgs_iter_idx == 1:
                species_set = True

            bfgs_iter_idx += 1

        elif ENERGY in ln:
            energies.append(float(ln_s[0]))
            free_energies.append(float(ln_s[1]))

        elif CELL in ln:

            current_cell[cell_ln_idx] = [float(ln_s[i]) for i in [0, 1, 2]]

            if cell_ln_idx == 2:
                cell_ln_idx = 0
                all_cells.append(current_cell.T)
                current_cell = np.zeros((3, 3))

            else:
                cell_ln_idx += 1

        elif STRESS in ln:

            current_stress[stress_ln_idx] = [
                float(ln_s[i]) for i in [0, 1, 2]]

            if stress_ln_idx == 2:
                stress_ln_idx = 0
                all_stresses.append(current_stress)
                current_stress = np.zeros((3, 3))

            else:
                stress_ln_idx += 1

        elif POS in ln:

            sp = ln_s[0]
            ion_idx = ln_s[1]

            if not species_set:

                if sp not in species:
                    species.append(sp)

                species_idx.append(species.index(sp))

            ion_xyz = [float(ln_s[i]) for i in [2, 3, 4]]
            current_bfgs_ions.append(np.array(ion_xyz)[:, np.newaxis])

        elif FORCE in ln:

            force_xyz = [float(ln_s[i]) for i in [2, 3, 4]]
            current_bfgs_forces.append(np.array(force_xyz)[:, np.newaxis])

    # Save the ion positions and forces for the final BFGS step:
    all_ions.append(np.hstack(current_bfgs_ions))
    all_forces.append(np.hstack(current_bfgs_forces))

    # Convert to Numpy arrays where sensible
    energies = np.array(energies)
    free_energies = np.array(free_energies)
    all_cells = np.array(all_cells)
    all_stresses = np.array(all_stresses)
    species = np.array(species)
    species_idx = np.array(species_idx)
    ions = np.array(all_ions)
    forces = np.array(all_forces)

    # Convert units to eV and Angstroms:
    energies *= E_h
    free_energies *= E_h
    ions *= A_0
    all_cells *= A_0
    forces *= F

    geom_dat = {
        'energies':         energies,
        'free_energies':    free_energies,
        'cells':            all_cells,
        'cell_stresses':    all_stresses,
        'species_idx':      species_idx,
        'species':          species,
        'atoms':            ions,
        'forces':           forces,
        'bfgs_num_iter':    len(energies),
        'iter_num':         iter_nums,
    }

    return geom_dat


@flexible_open
def read_cell_file(path_or_file, ret_frac=False):
    """Parse a CASTEP .cell file to get the atom positions, species and
    supercell.

    Parameters
    ----------
    path_or_file : str, Path, bytes or TextIOWrapper
        The input .cell file of the simulation. If type is `str` or `Path`, assumed to be
        a file path.    
    ret_frac : bool, optional
        If True, atoms are returned in fractional coordinates.

    Returns
    -------
    cell_dat : dict of (str : ndarray)
        supercell : ndarray of float of shape (3, 3)
            Supercell column vectors.
        atoms : ndarray of float of shape (3, N)
            Atom coordinates in either a Cartesian or fractional basis.
        species : ndarray of str of shape (M,)
            String array of the unique chemical symbols of the atoms.
        species_idx : ndarray of int of shape (N,)
            Array which maps each atom site to a chemical symbol in `species`.

    Notes
    -----
    The `flexible_open` decorator ensures the `path_or_file` object is converted to a list
    of strings.

    """

    atoms = []
    atom_sp = []
    atom_sp_idx = []
    supercell = np.empty((3, 3))
    sup_idx = 0

    parse_atoms_frac = False
    parse_atoms_abs = False
    parse_supercell = False
    is_frac = True

    for ln in path_or_file:

        ln = ln.strip()

        if parse_atoms_frac or parse_atoms_abs:

            if ln.upper() in [r'%ENDBLOCK POSITIONS_FRAC',
                              r'%ENDBLOCK POSITIONS_ABS']:
                parse_atoms_frac = False
                parse_atoms_abs = False
                continue

            ln_s = ln.split()
            ats = [float(i) for i in ln_s[1:]]
            atoms.append(ats)

            if ln_s[0] not in atom_sp:
                atom_sp.append(ln_s[0])

            atom_sp_idx.append(atom_sp.index(ln_s[0]))

        elif parse_supercell:

            if ln.upper() == r'%ENDBLOCK LATTICE_CART':
                parse_supercell = False
                continue

            ln_s = ln.split()
            try:
                sp = [float(i) for i in ln_s]
            except ValueError:
                continue

            supercell[:, sup_idx] = sp
            sup_idx += 1

        elif ln.upper() == r'%BLOCK POSITIONS_FRAC':
            parse_atoms_frac = True

        elif ln.upper() == r'%BLOCK POSITIONS_ABS':
            parse_atoms_abs = True
            is_frac = False

        elif ln.upper() == r'%BLOCK LATTICE_CART':
            parse_supercell = True

    atoms = np.array(atoms).T

    if is_frac and not ret_frac:
        atoms = np.dot(supercell, atoms)
    elif not is_frac and ret_frac:
        atoms = np.dot(np.linalg.inv(supercell), atoms)

    cell_dat = {
        'atom_sites': atoms,
        'supercell': supercell,
        'species': np.array(atom_sp),
        'species_idx': np.array(atom_sp_idx),
    }

    return cell_dat


def merge_geom_data(castep_dat, geom_dat):
    'Merge data from a .geom file into data from a .castep file.'

    iterations_idx = -1
    for idx, iter_num in enumerate(geom_dat['iter_num']):

        iterations_idx += 1
        while (iter_num != castep_dat['geom']['iterations'][iterations_idx]['iter_num']):
            iterations_idx += 1

        # print('updating geom for iteration idx: {}'.format(iterations_idx))
        castep_dat['geom']['iterations'][iterations_idx].update({
            'cell':         geom_dat['cells'][idx],
            'energy':       geom_dat['energies'][idx],
            'free_energy':  geom_dat['free_energies'][idx],
            'atoms':        geom_dat['atoms'][idx],
            'forces':       geom_dat['forces'][idx],
        })

        if geom_dat['cell_stresses'].size:
            castep_dat['geom']['iterations'][iter_num].update({
                'cell_stress':  geom_dat['cell_stresses'][idx],
            })

        castep_dat['geom']['species'] = geom_dat['species'][geom_dat['species_idx']]

    return castep_dat


def merge_cell_data(castep_dat, cell_dat):
    'Merge data from a .cell file into data from a .castep file.'

    # Reorder .cell atoms into order expected in .geom file or an output .cell file:

    species = cell_dat['species']
    species_idx_old = cell_dat['species_idx']

    species_map = map_species_to_castep(species, species_idx_old)
    species_idx = species_idx_old[species_map]
    species_all = species[species_idx]

    castep_dat.update({
        'structure': {
            'atoms': cell_dat['atom_sites'],
            'supercell': cell_dat['supercell'],
            'species': species_all,
        }
    })

    return castep_dat


def merge_output_data(castep_dat, geom_dat=None, cell_dat=None):
    'Merge .geom and/or .cell data with .castep file data'

    if geom_dat:
        castep_dat = merge_geom_data(castep_dat, geom_dat)

    if cell_dat:
        castep_dat = merge_cell_data(castep_dat, cell_dat)

    return castep_dat


def read_output_files(dir_path, seedname=None, ignore_missing_output=False):
    """Parse all output files from a CASTEP simulation.

    Parameters
    ----------
    dir_path : str or Path
        Path to the directory in which a CASTEP simulation has run.
    seedname : str, optional
        The CASTEP seedname of the simulation.
    ignore_missing_outputs : bool, optional
        If True, outputs in addition to the standard .castep file, such as the
        .geom file from a geometry optimisation simulation, that are missing
        from `dir_path` will not result in an error. If False, missing outputs
        will raise an error. By default, set to False.

    Returns
    -------
    outputs : dict

    Notes
    -----
    The main output file is`seedname`.castep.

    If `seedname` is specified, raise IOError if there is no `seedname`.castep
    file in `dir_path`.

    If `seedname` is not specified, raise IOError if there is not exactly one
    .castep file in `dir_path.

    Depending on the calculation task, additional output files are generated
    (e.g. `seedname`.geom for geometry optimisations). If these files are not
    found in `dir_path`, raise an IOError, unless `ignore_missing_output` is
    True.

    CASTEP versions tested: 17.2

    CASTEP tasks tested: SinglePoint, GeometryOptimisation

    """

    # Find the files ending in .castep in `dir_path`:
    all_cst_files = find_files_in_dir(dir_path, r'.castep$', recursive=False)

    # If no .castep files in `dir_path, raise IOError.
    if not all_cst_files:
        msg = 'No .castep files found in directory {}'.format(dir_path)
        raise FileNotFoundError(msg)

    if seedname is None:
        if len(all_cst_files) > 1:
            msg = 'Seedname not specified, but multiple .castep files found.'
            raise IOError(msg)
        else:
            seedname = all_cst_files[0].split('.castep')[0]

    cst_fn = '{}.castep'.format(seedname)
    dir_path = Path(dir_path)
    cst_path = dir_path.joinpath(cst_fn)

    if not cst_path.is_file():
        msg = 'File not found: {} in directory {}'.format(cst_fn, dir_path)
        raise FileNotFoundError(msg)

    # Parse the .castep file:
    cst_dat = read_castep_file(cst_path)

    calc_type = cst_dat['runs'][0]['parameters']['general_parameters']['type_of_calculation']
    is_geom = calc_type == 'geometry optimization'
    if is_geom:
        # TODO: check if need `write_geom` parameter from old version?

        # Parse the .geom file:
        geom_fn = '{}.geom'.format(seedname)
        geom_path = dir_path.joinpath(geom_fn)

        if not ignore_missing_output and not geom_path.is_file():
            msg = 'File not found: {} in directory {}'
            raise FileNotFoundError(msg.format(geom_fn, dir_path))

        geom_dat = read_geom_file(geom_path)

        cst_dat = merge_geom_data(cst_dat, geom_dat)

    return cst_dat


def read_relaxation(castep_path_or_file, geom_path_or_file):
    """Read CASTEP outputs into a dict format that can be used in the constructor of the
    `AtomisticRelaxation` class of the `atomistic` package.

    """

    cst_dat = read_castep_file(castep_path_or_file)
    geom_dat = read_geom_file(geom_path_or_file)
    merged_dat = merge_geom_data(cst_dat, geom_dat)

    data = {}
    atoms = []
    supercell = []

    for geom_iter in merged_dat['geom']['iterations']:

        final_step = geom_iter['steps'][-1]
        scf_idx = final_step['SCF_idx']
        for en_name, en in merged_dat['SCF']['energies'].items():
            if en_name not in data:
                data[en_name] = [en[scf_idx]]
            else:
                data[en_name].append(en[scf_idx])

        if geom_iter.get('atoms') is not None:
            atoms.append(geom_iter['atoms'])
            supercell.append(geom_iter['cell'])
        else:
            # Iteration is missing from the .geom file (due to structure reversion)
            atoms.append(atoms[-1])
            supercell.append(supercell[-1])

    for en_name in data:
        data[en_name] = np.array(data[en_name])

    atoms = np.array(atoms)
    supercell = np.array(supercell)

    out = {
        'initial_structure': {
            'atoms': atoms[0],
            'supercell': supercell[0],
            'species': merged_dat['geom']['species'],
        },
        'atom_displacements': atoms - atoms[0],
        'supercell_displacements': supercell - supercell[0],
        'data': data,
    }
    return out


def parse_castep_run(run_str, run_idx):

    pat_header = r'(\s\+-{49}\+)'
    pat_params = r'((?:\s\*{36}\sTitle\s\*{36})|(?:\s\*{79}\n))'
    pat_geom_iter_delim = r'(={80}\n\sStarting [L]?BFGS iteration\s+[0-9]+.+\n={80})'
    pat_finished_geom = r'[L]?BFGS: Geometry optimization completed successfully.'
    pat_final_geom_end = r'([L]?BFGS: Final (<frequency>|bulk modulus)\s+(?:=|unchanged)\s+.*)'

    header_split = re.split(pat_header, run_str)
    header_str = ''.join([header_split[i] for i in [1, 2, 3, 4, 5]])
    remainder_str = header_split[6]
    header = parse_castep_file_header(header_str)

    # TODO: parse pseudopotential section
    params_split = re.split(pat_params, remainder_str)
    pseudo_pot_str = params_split[0]
    params_str = ''.join([params_split[i] for i in [1, 2, 3]])
    remainder_str = params_split[4]
    parameters = parse_castep_file_parameters(params_str)

    run = {
        **header,
        'parameters': parameters,
    }

    end_str = None

    if parameters['general_parameters']['type_of_calculation'] == 'single point energy':
        print('single point')

        run_info_split = re.split(
            r'(\s\*{19} Unconstrained Forces \*{19})', remainder_str)
        run_info_str = run_info_split[0]
        end_str = run_info_split[1] + run_info_split[2]

    elif parameters['general_parameters']['type_of_calculation'] == 'geometry optimization':

        run_info_split = re.split(
            r'(\+-{16} MEMORY AND SCRATCH DISK ESTIMATES PER PROCESS -{14}\+)', remainder_str)

        if len(run_info_split) == 5:
            initial_SCF = True
            run_info_str = run_info_split[0] + run_info_split[1] + run_info_split[2]
            remainder_str = run_info_split[3] + run_info_split[4]
        elif len(run_info_split) == 3:
            initial_SCF = False
            run_info_str = run_info_split[0]
            remainder_str = run_info_split[1] + run_info_split[2]
        else:
            msg = 'Cannot parse run info for run idx {}'.format(run_idx)
            raise NotImplementedError(msg)

        # Extract out geom iterations:
        geom_iters_split = re.split(pat_geom_iter_delim, remainder_str)
        geom_iters_str_list = [i + j for i, j in
                               zip(geom_iters_split[1::2], geom_iters_split[2::2])]

        final_geom_str = None
        geom_initial_str = geom_iters_split[0]
        geom_initial = {
            'iter_num': 0,
            'resources': parse_castep_file_resource_estimates(geom_initial_str),
        }

        if initial_SCF:
            geom_initial.update({
                'forces': parse_castep_file_forces(geom_initial_str),
                **parse_castep_file_geom_iter_info(geom_initial_str),
            })

        geom_final_str = geom_iters_str_list[-1] if geom_iters_str_list else remainder_str

        geom_final = {}
        if re.search(pat_finished_geom, geom_final_str):
            # The geometry optimisation is finished in this run

            print('Geometry optimisation finished in run {}'.format(run_idx))

            final_iter_str, end_str = re.split(pat_finished_geom, geom_final_str)
            if geom_iters_str_list:
                geom_iters_str_list[-1] = final_iter_str

            final_geom_str, geom_freq_mod_str, freq_or_mod, end_str = re.split(
                pat_final_geom_end, end_str)
            final_geom_str = final_geom_str + geom_freq_mod_str

            if freq_or_mod == '<frequency>':
                final_cell_conts = parse_castep_file_cell_contents(
                    final_geom_str, is_initial=False)
                try:
                    final_freq = float(geom_freq_mod_str.strip().split()[-2])
                except ValueError:
                    final_freq = None
                geom_final.update({
                    'cell_contents': final_cell_conts,
                    'final_frequency': final_freq,
                })
            elif freq_or_mod == 'bulk modulus':
                final_cell = parse_castep_file_unit_cell(final_geom_str, is_initial=False)
                try:
                    final_bulk_mod = float(geom_freq_mod_str.strip().split()[-2])
                except ValueError:
                    final_freq = None
                geom_final.update({
                    'unit_cell': final_cell,
                    'bulk_modulus': final_bulk_mod,
                })

            final_enthalpy_str = re.search(
                r'[L]?BFGS: Final Enthalpy     =\s+(.*)', final_geom_str).groups()[0]
            final_enthalpy = float(final_enthalpy_str.strip().split()[0])

            geom_final.update({
                'final_enthalpy': final_enthalpy,
            })

        # Parse each geom iteration:
        geom_iter_initial = {k: v for k, v in geom_initial.items()
                             if k not in ['forces', 'resources']}

        geom_iters = []
        if run_idx == 0:
            # Only add the zeroth geom iteration if the first run of the file.
            geom_iters.append(
                {
                    **geom_iter_initial,
                    'steps': [
                        {
                            'forces': geom_initial['forces'],
                        }
                    ]
                }
            )

        if geom_iters_str_list:
            for geom_idx, geom_iter_str in enumerate(geom_iters_str_list, 1):

                if 'BFGS: finished iteration' not in geom_iter_str:
                    continue
                geom_iters.append(parse_castep_file_geom_iter(geom_iter_str, parameters))

        run.update({
            'geom': {
                'initial': geom_initial,
                'final': geom_final,
                'iterations': geom_iters,
            }
        })

    run_info = parse_castep_file_run_info(run_info_str, parameters)

    run.update({'system_info': {k: v for k, v in run_info.items()
                                if k not in ['scf', 'scf_energies']}})
    if 'scf' in run_info:
        # Not the case if continuation.
        run.update({
            'scf': run_info['scf'],
            'scf_energies': run_info['scf_energies'],
        })

    if end_str and ('Total time' in end_str):
        final_info = parse_castep_file_final_info(end_str)
        run.update({'final_info': final_info})

    return run


def parse_castep_file_header(header_str):

    for line in header_str.split('\n'):
        if 'CASTEP version' in line:
            version = line.strip().split()[-2]

    if not version:
        raise ValueError('Could not parse CASTEP version.')

    out = {
        'version': version
    }

    return out


def parse_castep_file_parameters(params_str):

    pat_param_groups = r'\s?\*+((?:\w|-|[^\S\n])+)\*+\n'
    param_group_split = re.split(pat_param_groups, params_str)[1:]

    out = {}
    for group_name, params in zip(param_group_split[::2], param_group_split[1::2]):
        group_name = group_name.lower().strip().replace(' ', '_')
        group_dict = {}
        for param_line in params.strip().split('\n'):
            if not param_line or ':' not in param_line:
                continue

            param_line = param_line.strip()
            name, val = param_line.rsplit(':', 1)
            name = '_'.join(name.strip().split())
            val = val.strip()
            group_dict.update({
                name: val
            })

        out.update({
            group_name: group_dict,
        })

    is_metallic = False
    metals_method = None
    method = out['electronic_minimization_parameters']['Method']
    if 'Treating system as metallic' in method:
        is_metallic = True

        if 'density mixing treatment' in method:
            metals_method = 'DM'
        elif 'ensemble DFT treatment':
            metals_method = 'EDFT'
        else:
            raise ValueError('Cannot determine metals method.')

    out.update({
        'is_metallic': is_metallic,
        'metals_method': metals_method,
    })

    return out


def parse_castep_file_run_info(run_info_str, parameters):

    pat_split = r'([^\S\n]+-{31}\n|\+-+ MEMORY AND SCRATCH DISK ESTIMATES PER PROCESS -+\+)'

    info_split = re.split(pat_split, run_info_str)[1:]
    info_list = [''.join(info_split[i:i + 4]) for i in range(0, len(info_split), 4)]

    unit_cell_str = info_list[0]
    cell_contents_str = info_list[1]
    species_info_str = info_list[2]
    kpoints_str = info_list[3]
    symm_str = info_list[4]

    out = {
        'cell_contents': parse_castep_file_cell_contents(cell_contents_str, is_initial=True),
        'unit_cell': parse_castep_file_unit_cell(unit_cell_str, is_initial=True),
        'kpoints': parse_castep_file_kpoint_info(kpoints_str),
    }

    if len(info_list) == 6:
        # If not continuation:
        remainder_str = info_list[5]

        if 'finite basis set correction' in remainder_str:
            # Expect multiple SCF cycles to estimate the finite basis set correction

            fbc_ln = re.search(r'(.*finite basis dEtot\/dlog\(Ecut\).*)',
                               remainder_str).groups()[0]
            fbc = fbc_ln.strip().split()[-1]

            pat_fbc = r'(Calculating total energy.*)'
            fbc_split = re.split(pat_fbc, remainder_str)
            resources = parse_castep_file_resource_estimates(fbc_split.pop(0))

            scf_str_list = [''.join(fbc_split[i:i + 2])
                            for i in range(0, len(fbc_split), 2)]
            scf = []
            scf_energies = []
            for i in scf_str_list:
                scf.append(parse_castep_file_scf(i, parameters['is_metallic']))
                scf_energies.append(parse_castep_file_scf_energies(
                    i, parameters['is_metallic']))

            out.update({
                'dEtot/dlog(Ecut)': fbc,
                'scf': scf,
                'scf_energies': scf_energies,
                'resource_estimates': resources,
            })

        else:
            resources = parse_castep_file_resource_estimates(remainder_str)
            scf = parse_castep_file_scf(remainder_str, parameters['is_metallic'])
            scf_energies = parse_castep_file_scf_energies(
                remainder_str, parameters['is_metallic'])
            out.update({
                'scf': [scf],
                'scf_energies': [scf_energies],
                'resource_estimates': resources,
            })

    return out


def parse_castep_file_kpoint_info(kpoint_info_str):

    body_str = re.split(r'[^\S\n]+-{31}\n', kpoint_info_str)[2]
    body_lns = body_str.strip().split('\n')[:3]

    lns_s = [ln.strip().split() for ln in body_lns]

    out = {
        'kpoint_MP_grid': [int(lns_s[0][i]) for i in [-3, -2, -1]],
        'kpoint_MP_offset': [float(lns_s[1][i]) for i in [-3, -2, -1]],
        'kpoint_num': int(lns_s[2][-1])
    }

    return out


def parse_castep_file_unit_cell(unit_cell_str, is_initial):

    unit_cell_body_str = re.split(r'[^\S\n]+-{31}\n', unit_cell_str)[2]

    ln_max = 13 if is_initial else 11
    unit_cell_lines = unit_cell_body_str.strip().split('\n')[:ln_max]

    out = {}
    lns = [ln.strip() for ln in unit_cell_lines]
    lns_s = [ln.split() for ln in lns]

    real_lattice = np.array([
        [lns_s[1][0], lns_s[1][1], lns_s[1][2]],
        [lns_s[2][0], lns_s[2][1], lns_s[2][2]],
        [lns_s[3][0], lns_s[3][1], lns_s[3][2]],
    ]).T.astype(float)

    recip_lattice = np.array([
        [lns_s[1][3], lns_s[1][4], lns_s[1][5]],
        [lns_s[2][3], lns_s[2][4], lns_s[2][5]],
        [lns_s[3][3], lns_s[3][4], lns_s[3][5]],
    ]).T.astype(float)

    lat_params = np.array([lns_s[6][2], lns_s[7][2], lns_s[8][2]]).astype(float)
    cell_angs = np.array([lns_s[6][5], lns_s[7][5], lns_s[8][5]]).astype(float)

    cell_vol = float(lns_s[10][4])

    out = {
        'real_lattice': real_lattice,
        'reciprocal_lattice': recip_lattice,
        'lattice_parameters': lat_params,
        'cell_angles': cell_angs,
        'cell_volume': cell_vol,
    }

    if is_initial:
        cell_dens_amu_ang = float(lns_s[11][2])
        cell_dens_g_cm = float(lns_s[12][1])
        out.update({
            'cell_density_AMU/Ang**3': cell_dens_amu_ang,
            'cell_density_g/cm**3': cell_dens_g_cm,
        })

    return out


def parse_castep_file_resource_estimates(estimates_str):

    pat_body = r'(?:\+-+ MEMORY AND SCRATCH DISK ESTIMATES PER PROCESS -+\+)|(?:\+-+\+)'
    estimate_lines = re.split(pat_body, estimates_str)[1].strip().split('\n')

    out = {}
    for ln in estimate_lines[:-2]:

        ln_s = ln.strip()

        key = ln_s[1:48].strip()
        if key:
            val_lst = ln_s[48:].strip().split()
            val = {
                'memory': val_lst[0] + ' ' + val_lst[1],
                'disk': val_lst[2] + ' ' + val_lst[3],
            }
            out.update({key: val})

    return out


def parse_castep_file_cell_contents(cell_contents_str, is_initial, parse_species=False):

    if is_initial:
        patt = r'x-{58}x|x{60}'
    else:
        patt = r'x-{63}x|x{65}'

    single_cell_split = re.split(patt, cell_contents_str)
    cell_lines = single_cell_split[2].strip().split('\n')
    atom_frac_coords = []
    for ln in cell_lines:
        ln_s = ln.strip().split()
        atom_frac_coords.append([float(ln_s[i]) for i in [3, 4, 5]])

    # TODO atom species

    return np.array(atom_frac_coords)


def parse_castep_file_scf(scf_str, is_metallic):

    # If metallic, extra column for Fermi energy:
    num_cols = 4 if is_metallic else 3

    single_scf_split = re.split(r'-{72} <-- SCF', scf_str)
    scf_lines = single_scf_split[2].strip().split('\n')
    scf_iter_data = np.ones((len(scf_lines), num_cols)) * np.nan

    for idx, line in enumerate(scf_lines):
        ln_s = line.strip().split()
        if idx == 0:
            if is_metallic:
                scf_iter_data[idx, 0] = float(ln_s[1])
                scf_iter_data[idx, 1] = float(ln_s[2])
                scf_iter_data[idx, 3] = float(ln_s[3])
            else:
                scf_iter_data[idx, 0] = float(ln_s[1])
                scf_iter_data[idx, 2] = float(ln_s[2])
        else:
            scf_iter_data[idx] = [float(ln_s[i]) for i in range(1, num_cols + 1)]

    return scf_iter_data


def parse_castep_file_scf_energies(scf_out_str, is_metallic):

    energy_key = 'Final energy ='
    if is_metallic:
        energy_key = 'Final energy, E             ='

    free_energy_key = 'Final free energy (E-TS)    ='
    zero_energy_key = 'NB est. 0K energy (E-0.5TS)      ='

    out = {}
    for ln in scf_out_str.split('\n'):
        ln_s = ln.strip()
        ln_ss = ln_s.split()
        if energy_key in ln_s:
            out.update({'final_energy': float(ln_ss[-2])})
        if free_energy_key in ln_s:
            out.update({'final_free_energy': float(ln_ss[-2])})
        if zero_energy_key in ln_s:
            out.update({'final_zero_energy': float(ln_ss[-2])})

    return out


def parse_castep_file_forces(forces_str):

    # Get forces type:
    force_type_dict = {
        'Constrained Forces':             'forces_constrained',
        'Unconstrained Forces':           'forces_unconstrained',
        'Constrained Symmetrised Forces': 'forces_constrained_sym',
        'Symmetrised Stress Tensor':      'stress_tensor_sym',

    }
    pat_force_type = r'\*{11,} (.*) \*{11,}'
    force_type_raw = re.search(pat_force_type, forces_str)

    try:
        force_type = force_type_dict[force_type_raw.groups()[0]]
    except KeyError:
        raise ValueError('Cannot parse forces block: \n"{!s}"'.format(forces_str))

    # Now split so we get the "body" of the block:
    pat_forces = r'(?:\* (?:-{45}|-{56}|-{80}) \*)|(?:\s(?:\*{84}|\*{60}|\*{49})[^\*]*)'
    forces_body = re.split(pat_forces, forces_str)[1]
    forces_lines = forces_body.strip().split('\n')[2:-1]
    lns_ss = [i.strip().split() for i in forces_lines]

    out = {}

    if force_type != 'stress_tensor_sym':
        species = []
        forces = []
        for ln in lns_ss:
            ln_i = [i.split("(cons'd)")[0] for i in ln]
            forces.append([float(ln_i[i]) for i in [3, 4, 5]])
            species.append(ln_i[1])

        out.update({
            force_type: np.array(forces),
            'species': np.array(species),
        })

    else:
        stress = np.array([
            [lns_ss[0][2], lns_ss[0][3], lns_ss[0][4]],
            [lns_ss[1][2], lns_ss[1][3], lns_ss[1][4]],
            [lns_ss[2][2], lns_ss[2][3], lns_ss[2][4]],
        ]).astype(float)
        pressure = float(lns_ss[4][2])

        out.update({
            force_type: stress,
            'pressure': pressure,
        })

    return out


def parse_castep_file_min_geom(min_geom_str):

    min_geom = {
        'step_type': [],
        'lambda': [],
        'F_delta_prime': [],
        'enthalpy': [],
    }

    min_geom_split = re.split(
        r'\+-{12}\+-{13}\+-{13}\+-{17}\+ <-- min [L]?BFGS', min_geom_str)
    min_geom_lines = min_geom_split[2].strip().split('\n')

    for line in min_geom_lines:
        ln_s = [i.strip() for i in line.split('|')]

        min_geom['step_type'].append(ln_s[1])
        min_geom['F_delta_prime'].append(float(ln_s[3]))
        min_geom['enthalpy'].append(float(ln_s[4]))

        try:
            min_geom['lambda'].append(float(ln_s[2]))
        except ValueError:
            min_geom['lambda'].append(np.nan)

    for key in min_geom:
        if key != 'step_type':
            min_geom[key] = np.array(min_geom[key])

    return min_geom


def parse_castep_file_geom_iter_step(geom_step_str, parameters):

    pat_cell = r'((?:[^\S\n]+-{31}\n(?:[^\S\n]+Unit Cell)\n[^\S\n]+-{31})|(?:[^\S\n]+Current cell volume.*)|(?:[^\S\n]+-{31}\n(?:[^\S\n]+Cell Contents)\n[^\S\n]+-{31}\s*x{65})|(?:[^\S\n]*x{65}))'

    cell_split = re.split(pat_cell, geom_step_str)

    if 'reverting to earlier configuration\n' in geom_step_str:
        cell_trial_str = cell_split[1] + cell_split[2] + cell_split[3]
        step_remainder = cell_split[4]
        cell_str = cell_split[5] + cell_split[6] + cell_split[7]
    else:
        cell_trial_str = None
        cell_str = cell_split[1] + cell_split[2] + cell_split[3]
        step_remainder = cell_split[4]

    scf_split = re.split(r'(-{72} <-- SCF)', step_remainder)

    scf_str = scf_split[1] + scf_split[2] + scf_split[3] + scf_split[4] + scf_split[5]
    step_remainder = scf_split[6]

    pat_forces = r'((?: \*+\sConstrained (?:Symmetrised )?Forces\s\*+)|(?: \*{84})|(?: \*+\sSymmetrised Stress Tensor\s\*+)|(?: \*{49}))'
    forces_split = re.split(pat_forces, step_remainder)

    scf_energies_str = forces_split[0].strip()

    forces_str = forces_split[1] + forces_split[2] + forces_split[3]

    min_geom_str = ' ' + forces_split[4].strip()

    scf = parse_castep_file_scf(scf_str, parameters['is_metallic'])
    scf_energies = parse_castep_file_scf_energies(
        scf_energies_str, parameters['is_metallic'])
    forces = parse_castep_file_forces(forces_str)
    min_geom = parse_castep_file_min_geom(min_geom_str)

    out = {
        'scf': scf,
        'scf_energies': scf_energies,
        'forces': forces,
        'min_geom': min_geom,
    }

    if 'Unit Cell' in cell_str:
        unit_cell = parse_castep_file_unit_cell(cell_str, is_initial=False)
        out.update({
            'unit_cell': unit_cell,
        })

    elif 'Cell Contents' in cell_str:
        cell_contents = parse_castep_file_cell_contents(cell_str, is_initial=False)
        out.update({
            'cell_contents': cell_contents,
        })

    return out


def parse_castep_file_geom_iter_info(geom_final_str):

    patt_finished_iter = r'([L]?BFGS: finished iteration.*)'
    match = re.search(patt_finished_iter, geom_final_str).groups()[0]
    enthalpy = float(match.strip().split()[-2])

    out = {
        'enthalpy': enthalpy,
    }

    patt_dE = '|  dE/ion   |'
    patt_S_max = '|   Smax    |'
    patt_F_max = '|  |F|max   |'
    patt_dR_max = '|  |dR|max  |'

    geom_final_lines = geom_final_str.split('\n')
    for ln in geom_final_lines:
        ln_s = ln.strip()
        ln_ss = ln_s.split()
        if patt_dE in ln_s:
            out.update({'dE_per_ion': float(ln_ss[3])})
        if patt_S_max in ln_s:
            out.update({'mag_S_max': float(ln_ss[3])})
        if patt_F_max in ln_s:
            out.update({'mag_F_max': float(ln_ss[3])})
        if patt_dR_max in ln_s:
            out.update({'mag_dR_max': float(ln_ss[3])})

    return out


def parse_castep_file_geom_iter(geom_iter_str, parameters):

    # Extract out geom iteration steps:
    patt_geom_iter_step = r'(-{80}\n\s[L]?BFGS: (?:starting|improving) iteration.*\n-{80})'
    patt_finished_iter = r'([L]?BFGS: finished iteration.*)'

    iter_steps_split = re.split(patt_geom_iter_step, geom_iter_str)
    iter_begin_str = iter_steps_split.pop(0)

    iter_steps_str_list = [i + j for i,
                           j in zip(iter_steps_split[::2], iter_steps_split[1::2])]

    iter_num_str = re.search(
        r'Starting [L]?BFGS iteration\s+([0-9]+)', iter_begin_str).groups()[0]
    iter_num = int(iter_num_str)

    # Remove iteration ending bit:
    final_step_str, fin_iter_str, iter_end_str = re.split(
        patt_finished_iter, iter_steps_str_list[-1])
    iter_steps_str_list[-1] = final_step_str
    fin_iter_str = fin_iter_str + iter_end_str
    final = parse_castep_file_geom_iter_info(fin_iter_str)

    out = {
        **final,
        'iter_num': iter_num,
        'steps': [],
    }
    for step_idx, i in enumerate(iter_steps_str_list):
        step_i = parse_castep_file_geom_iter_step(i, parameters)
        out['steps'].append(step_i)

    return out


def parse_castep_file_final_info(final_str):

    pat_forces = r'\s((?:\*{84}|\*{60}|\*{49})[^\*])'
    forces_split = re.split(pat_forces, final_str)

    forces_str_list = [''.join(forces_split[i:i+2])
                       for i in range(0, len(forces_split), 2)]
    remainder_str = forces_str_list.pop(-1)

    forces = {}
    for force_str in forces_str_list:
        forces.update(parse_castep_file_forces(force_str))

    stats = parse_castep_file_run_stats(remainder_str)

    out = {
        'forces': forces,
        'statistics': stats,
    }

    return out


def parse_castep_file_run_stats(stats_str):

    out = {}

    for ln in stats_str.strip().split('\n'):
        ln_s = ln.strip()
        ln_ss = ln_s.split()

        if 'Initialisation time' in ln_s:
            out.update({'init_time_s': float(ln_ss[-2])})

        elif 'Calculation time' in ln_s:
            out.update({'calculation_time_s': float(ln_ss[-2])})

        elif 'Finalisation time' in ln_s:
            out.update({'finalisation_time_s': float(ln_ss[-2])})

        elif 'Total time' in ln_s:
            out.update({'total_time_s': float(ln_ss[-2])})

        elif 'Peak Memory Use' in ln_s:
            out.update({'peak_memory_use_kB': float(ln_ss[-2])})

        elif 'Overall parallel efficiency' in ln_s:
            out.update({'overall_parallel_efficiency': ln_s.split(':')[1].strip()})

        elif 'k-point' in ln_s:
            out.update({
                'kpoint_distribution': ln_s.split('(')[1].split(')')[0],
                'kpoint_parallel_efficiency': ln_s.split(':')[1].strip(),
            })

    return out
