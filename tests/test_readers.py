"""`castep_parse.test.test_readers.py`"""

import os
import unittest
from pathlib import Path

import numpy as np

from castep_parse.readers import (
    read_castep_file,
    read_cell_file,
    read_geom_file,
    parse_castep_file_header,
    parse_castep_file_parameters,
    parse_castep_file_geom_iter_info,
    parse_castep_file_min_geom,
    parse_castep_file_forces,
    parse_castep_file_scf,
    parse_castep_file_scf_energies,
    parse_castep_file_cell_contents,
    parse_castep_file_unit_cell,
    parse_castep_file_resource_estimates,
    parse_castep_file_run_info,
    parse_castep_file_run_stats,
    parse_castep_file_kpoint_info,
    parse_castep_file_final_info
)
from castep_parse.utils import array_nan_equal

ROOT_FILES_PATH = os.path.join('tests', 'castep_files')


class ReadersTestCase(unittest.TestCase):
    'Just checking the test files can be parsed without incident.'

    def test_read_castep(self):
        cst_out_1 = read_castep_file(os.path.join(ROOT_FILES_PATH, 'GO_cell.castep'))
        cst_out_2 = read_castep_file(os.path.join(ROOT_FILES_PATH, 'GO_max_geom.castep'))
        cst_out_3 = read_castep_file(os.path.join(ROOT_FILES_PATH, 'SP.castep'))
        cst_out_4 = read_castep_file(os.path.join(ROOT_FILES_PATH, 'GO_single_iter.castep'))

    def test_read_geom(self):
        geom_1 = read_geom_file(os.path.join(ROOT_FILES_PATH, 'GO_cell.geom'))
        geom_2 = read_geom_file(os.path.join(ROOT_FILES_PATH, 'GO_max_geom.geom'))
        geom_3 = read_geom_file(os.path.join(ROOT_FILES_PATH, 'GO_single_iter.geom'))

    def test_read_cell(self):
        cell_1 = read_cell_file(os.path.join(ROOT_FILES_PATH, 'GO_cell.cell'))
        cell_2 = read_cell_file(os.path.join(ROOT_FILES_PATH, 'GO_max_geom.cell'))
        cell_3 = read_cell_file(os.path.join(ROOT_FILES_PATH, 'SP.cell'))
        cell_4 = read_cell_file(os.path.join(ROOT_FILES_PATH, 'GO_single_iter.cell'))


class ReadStructureTestCase(unittest.TestCase):
    'Checking for consistency between cell and geom file readers.'

    # TODO: add check that final geom structure matches -out.cell structure

    def test_initial_structure_consistency(self):
        'Check cell file structure is equivalent to first structure reported in geom file'

        def check_consistent(cell_dat, geom_dat):

            self.assertTrue(np.allclose(geom_dat['cells'][0], cell_dat['supercell']))

            atm_c = cell_dat['atom_sites']
            atm_c_srt_idx = np.lexsort(np.round(atm_c, decimals=7))
            atm_c_srt = atm_c[:, atm_c_srt_idx]
            species_c = cell_dat['species'][cell_dat['species_idx']]
            species_c_srt = species_c[atm_c_srt_idx]

            atm_g = geom_dat['atoms'][0]
            atm_g_srt_idx = np.lexsort(np.round(atm_g, decimals=7))
            atm_g_srt = atm_g[:, atm_g_srt_idx]
            species_g = geom_dat['species'][geom_dat['species_idx']]
            species_g_srt = species_g[atm_g_srt_idx]

            self.assertTrue(np.allclose(atm_c_srt, atm_g_srt))
            self.assertTrue(np.array_equal(species_c_srt, species_g_srt))

        cell_1 = read_cell_file(os.path.join(ROOT_FILES_PATH, 'GO_cell.cell'))
        geom_1 = read_geom_file(os.path.join(ROOT_FILES_PATH, 'GO_cell.geom'))
        check_consistent(cell_1, geom_1)

        cell_2 = read_cell_file(os.path.join(ROOT_FILES_PATH, 'GO_max_geom.cell'))
        geom_2 = read_geom_file(os.path.join(ROOT_FILES_PATH, 'GO_max_geom.geom'))
        check_consistent(cell_2, geom_2)


class FlexibleOpenTestCase(unittest.TestCase):
    'Test different inputs are equivalent in read_* functions due to `flexible_open`.'

    def test_flexible_read_cell(self):

        cell_filename = os.path.join(ROOT_FILES_PATH, 'GO_cell.cell')

        cell_out_1 = read_cell_file(cell_filename)
        cell_path = Path(cell_filename)
        cell_out_2 = read_cell_file(cell_path)

        with cell_path.open() as handle:
            cell_out_3 = read_cell_file(handle)

        with cell_path.open('rb') as handle:
            cell_bytes = handle.read()
            cell_out_4 = read_cell_file(cell_bytes)

        for k in ['atom_sites', 'supercell', 'species', 'species_idx']:
            self.assertTrue(
                np.array_equal(cell_out_1[k], cell_out_2[k]) and
                np.array_equal(cell_out_2[k], cell_out_3[k]) and
                np.array_equal(cell_out_3[k], cell_out_4[k])
            )

    def test_flexible_read_castep(self):

        cst_filename = os.path.join(ROOT_FILES_PATH, 'GO_cell.castep')

        cst_out_1 = read_castep_file(cst_filename)

        cst_path = Path(cst_filename)
        cst_out_2 = read_castep_file(cst_path)

        with cst_path.open() as handle:
            cst_out_3 = read_castep_file(handle)

        with cst_path.open('rb') as handle:
            cst_bytes = handle.read()
            cst_out_4 = read_castep_file(cst_bytes)

        self.assertTrue(
            array_nan_equal(cst_out_1['SCF']['cycles'][0], cst_out_2['SCF']['cycles'][0]) and
            array_nan_equal(cst_out_2['SCF']['cycles'][0], cst_out_3['SCF']['cycles'][0]) and
            array_nan_equal(cst_out_3['SCF']['cycles'][0], cst_out_4['SCF']['cycles'][0])
        )

    def test_flexible_read_geom(self):

        geom_filename = os.path.join(ROOT_FILES_PATH, 'GO_cell.geom')

        geom_out_1 = read_geom_file(geom_filename)
        geom_path = Path(geom_filename)
        geom_out_2 = read_geom_file(geom_path)

        with geom_path.open() as handle:
            geom_out_3 = read_geom_file(handle)

        with geom_path.open('rb') as handle:
            geom_bytes = handle.read()
            geom_out_4 = read_geom_file(geom_bytes)

        for k in ['energies', 'forces']:
            self.assertTrue(
                np.array_equal(geom_out_1[k], geom_out_2[k]) and
                np.array_equal(geom_out_2[k], geom_out_3[k]) and
                np.array_equal(geom_out_3[k], geom_out_4[k])
            )


class CastepFileParsersTestCase(unittest.TestCase):
    'Tests on the individual functions used to parse blocks of a .castep file.'

    def test_parse_castep_file_header(self):

        header_str = """
            +-------------------------------------------------+
            |                                                 |
            |      CCC   AA    SSS  TTTTT  EEEEE  PPPP        |
            |     C     A  A  S       T    E      P   P       |
            |     C     AAAA   SS     T    EEE    PPPP        |
            |     C     A  A     S    T    E      P           |
            |      CCC  A  A  SSS     T    EEEEE  P           |
            |                                                 |
            +-------------------------------------------------+
            |                                                 |
            | Welcome to Academic Release CASTEP version 17.2 |          
            | Ab Initio Total Energy Program                  |
            |                                                 |
            | Authors:                                        |
            | M. Segall, M. Probert, C. Pickard, P. Hasnip,   |
            | S. Clark, K. Refson, J. R. Yates, M. Payne      |
            |                                                 |
            | Contributors:                                   |
            | P. Lindan, P. Haynes, J. White, V. Milman,      |
            | N. Govind, M. Gibson, P. Tulip, V. Cocula,      |
            | B. Montanari, D. Quigley, M. Glover,            |
            | L. Bernasconi, A. Perlov, M. Plummer,           |
            | E. McNellis, J. Meyer, J. Gale, D. Jochym       |
            | J. Aarons, B. Walker, R. Gillen, D. Jones       |
            | T. Green, I. J. Bush, C. J. Armstrong,          |
            | E. J. Higgins, E. L. Brown, M. S. McFly,        |
            | J. Wilkins, B-C. Shih, P. J. P. Byrne           |
            |                                                 |
            | Copyright (c) 2000 - 2017                       |
            |                                                 |
            |     Distributed under the terms of an           |
            |     Agreement between the United Kingdom        |
            |     Car-Parrinello (UKCP) Consortium,           |
            |     Daresbury Laboratory and Accelrys, Inc.     |
            |                                                 |
            | Please cite                                     |
            |                                                 |
            |     "First principles methods using CASTEP"     |
            |                                                 |
            |         Zeitschrift fuer Kristallographie       |
            |           220(5-6) pp. 567-570 (2005)           |
            |                                                 |
            | S. J. Clark, M. D. Segall, C. J. Pickard,       |
            | P. J. Hasnip, M. J. Probert, K. Refson,         |
            | M. C. Payne                                     |
            |                                                 |
            |       in all publications arising from          |
            |              your use of CASTEP                 |
            |                                                 |
            +-------------------------------------------------+     
        """

        header = parse_castep_file_header(header_str)
        self.assertTrue(header['version'] == '17.2')

    def test_parse_castep_file_parameters(self):

        params_str = """
            ************************************ Title ************************************


            ***************************** General Parameters ******************************

            output verbosity                               : normal  (1)
            write checkpoint data to                       : sim.check
            type of calculation                            : single point energy
            stress calculation                             : off
            density difference calculation                 : off
            electron localisation func (ELF) calculation   : off
            Hirshfeld analysis                             : off
            unlimited duration calculation
            timing information                             : on
            memory usage estimate                          : on
            write final potential to formatted file        : off
            write final density to formatted file          : off
            write BibTeX reference list                    : on
            write OTFG pseudopotential files               : on
            write electrostatic potential file             : on
            write bands file                               : on
            checkpoint writing                             : both castep_bin and check files
            random number generator seed                   : randomised (154450526)

            *********************** Exchange-Correlation Parameters ***********************

            using functional                               : Local Density Approximation
            DFT+D: Semi-empirical dispersion correction    : off

            ************************* Pseudopotential Parameters **************************

            pseudopotential representation                 : reciprocal space
            <beta|phi> representation                      : reciprocal space
            spin-orbit coupling                            : off

            **************************** Basis Set Parameters *****************************

            basis set accuracy                             : FINE
            finite basis set correction                    : none

            **************************** Electronic Parameters ****************************

            number of  electrons                           :  192.0    
            net charge of system                           :  0.000    
            treating system as non-spin-polarized
            number of bands                                :        115

            ********************* Electronic Minimization Parameters **********************

            Method: Treating system as metallic with density mixing treatment of electrons,
                and number of  SD  steps               :          1
                and number of  CG  steps               :          4

            total energy / atom convergence tol.           : 0.1000E-04   eV
            eigen-energy convergence tolerance             : 0.1000E-05   eV
            max force / atom convergence tol.              : ignored
            periodic dipole correction                     : NONE

            ************************** Density Mixing Parameters **************************

            density-mixing scheme                          : Broyden
            max. length of mixing history                  :         20

            *********************** Population Analysis Parameters ************************

            Population analysis with cutoff                :  3.000       A

            *******************************************************************************

        """

        params = parse_castep_file_parameters(params_str)

        self.assertTrue(params['electronic_minimization_parameters']
                        ['total_energy_/_atom_convergence_tol.'] == '0.1000E-04   eV')

    def test_parse_castep_file_kpoint_info(self):

        kpoint_str = """
            -------------------------------
                k-Points For BZ Sampling
            -------------------------------
        MP grid size for SCF calculation is  2  2  1
                with an offset of   0.000  0.000  0.000
        Number of kpoints used =             2

        """

        kpoint_info = parse_castep_file_kpoint_info(kpoint_str)

        self.assertTrue(kpoint_info['kpoint_MP_grid'] == [2, 2, 1])
        self.assertTrue(kpoint_info['kpoint_MP_offset'] == [0.0, 0.0, 0.0])
        self.assertTrue(kpoint_info['kpoint_num'] == 2)

    def test_parse_castep_file_unit_cell(self):

        unit_cell_str = """
                                    -------------------------------
                                                Unit Cell
                                    -------------------------------
                    Real Lattice(A)                      Reciprocal Lattice(1/A)
            6.4632000   0.0000000   0.0000000        0.9721477   0.5612698  -0.0000000
            -3.2316000   5.5972954   0.0000000        0.0000000   1.1225395  -0.0000000
            0.0000000   0.0000000  10.2950000        0.0000000   0.0000000   0.6103143

                                Lattice parameters(A)       Cell Angles
                                a =    6.463200          alpha =   90.000000
                                b =    6.463200          beta  =   90.000000
                                c =   10.295000          gamma =  120.000000

                                Current cell volume =  372.436445       A**3
                                            density =    3.919015   AMU/A**3
                                                    =    6.507677     g/cm^3

        """

        unit_cell = parse_castep_file_unit_cell(unit_cell_str, is_initial=True)
        self.assertTrue(np.allclose(unit_cell['real_lattice'][0], [6.4632, -3.2316, 0]))
        self.assertTrue(np.allclose(unit_cell['reciprocal_lattice'][0],
                                    [0.9721477, 0, 0]))
        self.assertTrue(unit_cell['lattice_parameters'][1] == 6.4632)
        self.assertTrue(unit_cell['cell_density_AMU/Ang**3'] == 3.919015)

    def test_parse_castep_file_run_stats(self):

        stats_str = """

            Initialisation time =     10.51 s
            Calculation time    =     32.52 s
            Finalisation time   =      1.07 s
            Total time          =     44.10 s
            Peak Memory Use     = 693284 kB

            Overall parallel efficiency rating: Very good (88%)                             

            Data was distributed by:-
            k-point (2-way); efficiency rating: Very good (88%)                             

            Parallel notes:
            1) Calculation only took 43.1 s, so efficiency estimates may be inaccurate.     

        """

        stats = parse_castep_file_run_stats(stats_str)

        self.assertTrue(stats['init_time_s'] == 10.51)
        self.assertTrue(stats['calculation_time_s'] == 32.52)
        self.assertTrue(stats['finalisation_time_s'] == 1.07)
        self.assertTrue(stats['total_time_s'] == 44.10)
        self.assertTrue(stats['peak_memory_use_kB'] == 693284.0)

        self.assertTrue(stats['overall_parallel_efficiency'] == 'Very good (88%)')
        self.assertTrue(stats['kpoint_distribution'] == '2-way')
        self.assertTrue(stats['kpoint_parallel_efficiency'] == 'Very good (88%)')

    def test_parse_castep_file_run_info(self):

        run_info_str = """   
                                    -------------------------------
                                                Unit Cell
                                    -------------------------------
                    Real Lattice(A)                      Reciprocal Lattice(1/A)
            6.4632000   0.0000000   0.0000000        0.9721477   0.5612698  -0.0000000
            -3.2316000   5.5972954   0.0000000        0.0000000   1.1225395  -0.0000000
            0.0000000   0.0000000  10.2950000        0.0000000   0.0000000   0.6103143

                                Lattice parameters(A)       Cell Angles
                                a =    6.463200          alpha =   90.000000
                                b =    6.463200          beta  =   90.000000
                                c =   10.295000          gamma =  120.000000

                                Current cell volume =  372.436445       A**3
                                            density =    3.919015   AMU/A**3
                                                    =    6.507677     g/cm^3

                                    -------------------------------
                                                Cell Contents
                                    -------------------------------
                                    Total number of ions in cell =   16
                                Total number of species in cell =    1
                                    Max number of any one species =   16

                        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
                        x  Element    Atom        Fractional coordinates of atoms  x
                        x            Number           u          v          w      x
                        x----------------------------------------------------------x
                        x  Zr           1         0.333333   0.166667   0.375000   x 
                        x  Zr           2         0.166667   0.333333   0.125000   x 
                        x  Zr           3         0.333333   0.166667   0.875000   x 
                        x  Zr           4         0.166667   0.333333   0.625000   x 
                        x  Zr           5         0.833333   0.166667   0.375000   x 
                        x  Zr           6         0.666667   0.333333   0.125000   x 
                        x  Zr           7         0.833333   0.166667   0.875000   x 
                        x  Zr           8         0.666667   0.333333   0.625000   x 
                        x  Zr           9         0.333333   0.666667   0.375000   x 
                        x  Zr          10         0.166667   0.833333   0.125000   x 
                        x  Zr          11         0.333333   0.666667   0.875000   x 
                        x  Zr          12         0.166667   0.833333   0.625000   x 
                        x  Zr          13         0.833333   0.666667   0.375000   x 
                        x  Zr          14         0.666667   0.833333   0.125000   x 
                        x  Zr          15         0.833333   0.666667   0.875000   x 
                        x  Zr          16         0.666667   0.833333   0.625000   x 
                        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


                                    No user defined ionic velocities

                                    -------------------------------
                                            Details of Species
                                    -------------------------------

                                        Mass of species in AMU
                                                Zr   91.2240000

                                    Electric Quadrupole Moment (Barn)
                                                Zr   -0.1760000 Isotope 91

                                    Files used for pseudopotentials:
                                                Zr 3|2.1|7|8|9|40U:50:41:42

                                    -------------------------------
                                        k-Points For BZ Sampling
                                    -------------------------------
                                MP grid size for SCF calculation is  2  2  1
                                        with an offset of   0.000  0.000  0.000
                                Number of kpoints used =             2

                                    -------------------------------
                                        Symmetry and Constraints
                                    -------------------------------

                                Cell is a supercell containing 8 primitive cells
                                Maximum deviation from symmetry =  0.00000         ANG

                                Number of symmetry operations   =           1
                                Number of ionic constraints     =           3
                                Point group of crystal =     1: C1, 1, 1
                                Space group of crystal =   194: P6_3/mmc, -P 6c 2c

                        Set iprint > 1 for details on symmetry rotations/translations

                                    Centre of mass is constrained
                        Set iprint > 1 for details of linear ionic constraints

                                    Number of cell constraints= 0
                                    Cell constraints are:  1 2 3 4 5 6

                                    External pressure/stress (GPa)
                                    0.00000   0.00000   0.00000
                                                0.00000   0.00000
                                                        0.00000

            +---------------- MEMORY AND SCRATCH DISK ESTIMATES PER PROCESS --------------+
            |                                                     Memory          Disk    |
            | Baseline code, static data and system overhead      352.0 MB         0.0 MB |
            | BLAS internal memory storage                          0.0 MB         0.0 MB |
            | Model and support data                               50.1 MB        64.0 MB |
            | Electronic energy minimisation requirements          41.9 MB         0.0 MB |
            | Force calculation requirements                        6.3 MB         0.0 MB |
            |                                               ----------------------------- |
            | Approx. total storage required per process          444.0 MB        64.0 MB |
            |                                                                             |
            | Requirements will fluctuate during execution and may exceed these estimates |
            +-----------------------------------------------------------------------------+
            Calculating total energy with cut-off of  244.902 eV.
            ------------------------------------------------------------------------ <-- SCF
            SCF loop      Energy           Fermi           Energy gain       Timer   <-- SCF
                                        energy          per atom          (sec)   <-- SCF
            ------------------------------------------------------------------------ <-- SCF
            Initial  -1.72398321E+004  0.00000000E+000                        12.07  <-- SCF
                1  -2.01881895E+004  1.20817276E+001   1.84272338E+002      15.39  <-- SCF
                2  -2.06977223E+004  1.01841686E+001   3.18457989E+001      17.95  <-- SCF
                3  -2.07256217E+004  9.99025161E+000   1.74371675E+000      20.81  <-- SCF
                4  -2.07251982E+004  9.76895803E+000  -2.64691356E-002      23.49  <-- SCF
                5  -2.07252662E+004  9.76701620E+000   4.24855689E-003      27.20  <-- SCF
                6  -2.07252289E+004  9.76599078E+000  -2.33385155E-003      30.08  <-- SCF
                7  -2.07252275E+004  9.76323432E+000  -8.30161566E-005      32.50  <-- SCF
                8  -2.07252278E+004  9.76297539E+000   1.75286433E-005      34.90  <-- SCF
                9  -2.07252265E+004  9.76290887E+000  -8.27013368E-005      37.13  <-- SCF
                10  -2.07252265E+004  9.76287342E+000   9.76737366E-007      38.92  <-- SCF
                11  -2.07252265E+004  9.76286578E+000  -9.62048099E-007      40.61  <-- SCF
            ------------------------------------------------------------------------ <-- SCF

            Final energy, E             =  -20724.93931925     eV
            Final free energy (E-TS)    =  -20725.22649761     eV
            (energies not corrected for finite basis set)

            NB est. 0K energy (E-0.5TS)      =  -20725.08290843     eV


            Writing analysis data to sim.castep_bin

            Writing model to sim.check
        
        """

        parameters = {'is_metallic': True, }
        run_info = parse_castep_file_run_info(run_info_str, parameters)

        # TODO check something?!

    def test_parse_castep_file_resource_estimates(self):

        est_str = """
            +---------------- MEMORY AND SCRATCH DISK ESTIMATES PER PROCESS --------------+
            |                                                     Memory          Disk    |
            | Baseline code, static data and system overhead      236.0 MB         0.0 MB |
            | BLAS internal memory storage                          0.0 MB         0.0 MB |
            | Model and support data                             1099.9 MB         0.0 MB |
            | Electronic energy minimisation requirements        1050.6 MB         0.0 MB |
            | Force calculation requirements                       31.0 MB         0.0 MB |
            |                                               ----------------------------- |
            | Approx. total storage required per process         2386.5 MB         0.0 MB |
            |                                                                             |
            | Requirements will fluctuate during execution and may exceed these estimates |
            +-----------------------------------------------------------------------------+
        """

        est = parse_castep_file_resource_estimates(est_str)

        self.assertTrue(est['Approx. total storage required per process']
                        ['memory'] == '2386.5 MB')

        est_str = """
            +---------------- MEMORY AND SCRATCH DISK ESTIMATES PER PROCESS --------------+
            |                                                     Memory          Disk    |
            | Model and support data                             1098.1 MB         0.0 MB |
            | Electronic energy minimisation requirements        1050.6 MB         0.0 MB |
            | Geometry minimisation requirements                 1448.3 MB         0.0 MB |
            |                                               ----------------------------- |
            | Approx. total storage required per process         3596.9 MB         0.0 MB |
            |                                                                             |
            | Requirements will fluctuate during execution and may exceed these estimates |
            +-----------------------------------------------------------------------------+
        """
        est = parse_castep_file_resource_estimates(est_str)

        self.assertTrue(est['Approx. total storage required per process']
                        ['memory'] == '3596.9 MB')

    def test_parse_castep_file_cell_contents(self):

        cell_c_str = """
                            -------------------------------
                                        Cell Contents
                            -------------------------------
    
                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
                x  Element         Atom        Fractional coordinates of atoms  x
                x                 Number           u          v          w      x
                x---------------------------------------------------------------x
                x  Zr                1         0.750003   0.190838   0.507723   x
                x  Zr                2         0.249997   0.810043   0.491756   x
                x  I                 1         0.249995   0.818044   0.992372   x
                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx    
        """

        arr = parse_castep_file_cell_contents(cell_c_str, is_initial=False)

        self.assertTrue(np.allclose(arr[1], [0.249997, 0.810043, 0.491756]))

    def test_parse_castep_file_scf(self):
        scf_str = """    
            ------------------------------------------------------------------------ <-- SCF
            SCF loop      Energy           Fermi           Energy gain       Timer   <-- SCF
                                        energy          per atom          (sec)   <-- SCF
            ------------------------------------------------------------------------ <-- SCF
            Initial  -1.08131562E+005  0.00000000E+000                    335187.17  <-- SCF
                1  -1.08131967E+005  9.85009524E+000   4.82564918E-003  335690.22  <-- SCF
                2  -1.08131967E+005  9.85009522E+000   2.06025849E-007  336162.12  <-- SCF
                3  -1.08131967E+005  9.85010646E+000   3.32870285E-008  336535.82  <-- SCF
                4  -1.08131967E+005  9.85009630E+000  -1.42821580E-008  336905.45  <-- SCF
            ------------------------------------------------------------------------ <-- SCF
        """

        scf_arr = parse_castep_file_scf(scf_str, is_metallic=True)

    def test_parse_castep_scf_output(self):

        scf_out_str = """
            Integrated Spin Density     =    0.463639E-04 hbar/2                        
            Integrated |Spin Density|   =    0.663738E-03 hbar/2                        

            Final energy, E             =  -108131.5619032     eV
            Final free energy (E-TS)    =  -108131.9674947     eV
            (energies not corrected for finite basis set)

            NB est. 0K energy (E-0.5TS)      =  -108131.7646989     eV    
        """

        scf_out = parse_castep_file_scf_energies(scf_out_str, is_metallic=True)

        self.assertTrue(scf_out['final_energy'] == float(-108131.5619032))
        self.assertTrue(scf_out['final_free_energy'] == float(-108131.9674947))
        self.assertTrue(scf_out['final_zero_energy'] == float(-108131.7646989))

    def test_parse_castep_file_forces(self):

        forces_str = """
            ******************************** Constrained Forces ********************************
            *                                                                                  *
            *                           Cartesian components (eV/A)                            *
            * -------------------------------------------------------------------------------- *
            *                         x                    y                    z              *
            *                                                                                  *
            * Zr              1     -0.00002             -0.00132             -0.00259         *
            * I               1     -0.00001              0.05841              0.00006         *
            *                                                                                  *
            ************************************************************************************    
        """
        forces = parse_castep_file_forces(forces_str)

        self.assertTrue(
            np.allclose(
                forces['forces_constrained'],
                [
                    [-0.00002, -0.00132, -0.00259],
                    [-0.00001, 0.05841, 0.00006],
                ]
            )
        )
        self.assertTrue(np.all(forces['species'] == ['Zr', 'I']))

        stress_str = """
            *********** Symmetrised Stress Tensor ***********
            *                                               *
            *          Cartesian components (GPa)           *
            * --------------------------------------------- *
            *             x             y             z     *
            *                                               *
            *  x      5.484306      0.000000      0.000000  *
            *  y      0.000000      5.484306      0.000000  *
            *  z      0.000000      0.000000      2.859856  *
            *                                               *
            *  Pressure:   -4.6095                          *
            *                                               *
            *************************************************
        """

        stress = parse_castep_file_forces(stress_str)

        self.assertTrue(
            np.allclose(
                stress['stress_tensor_sym'],
                [
                    [5.484306, 0.0, 0.0],
                    [0.0, 5.484306, 0.0],
                    [0.0, 0.0, 2.859856],
                ]
            )
        )
        self.assertTrue(stress['pressure'] == -4.6095)

    def test_parse_castep_file_min_geom(self):

        min_geom_str = """
            +------------+-------------+-------------+-----------------+ <-- min LBFGS
            |    Step    |   lambda    |   F.delta'  |    enthalpy     | <-- min LBFGS
            +------------+-------------+-------------+-----------------+ <-- min LBFGS
            |  previous  |    0.000000 |    0.008972 |  -108131.967460 | <-- min LBFGS
            | trial step |    1.000000 |    0.003681 |  -108131.967495 | <-- min LBFGS
            |  line step |    1.695703 |    0.001329 |  -108131.967514 | <-- min LBFGS
            +------------+-------------+-------------+-----------------+ <-- min LBFGS    
        """

        min_geom = parse_castep_file_min_geom(min_geom_str)

        self.assertTrue(min_geom['step_type'] == ['previous', 'trial step', 'line step'])
        self.assertTrue(np.allclose(min_geom['lambda'], [0.0, 1.0, 1.695703]))
        self.assertTrue(
            np.allclose(
                min_geom['F_delta_prime'],
                [0.008972, 0.003681, 0.001329]
            )
        )
        self.assertTrue(
            np.allclose(
                min_geom['enthalpy'],
                [-108131.967460, -108131.967495, -108131.967514]
            )
        )

    def test_parse_castep_file_geom_iter_info(self):

        geom_iter_info_str = """
            LBFGS: finished iteration    46 with enthalpy= -1.08131968E+005 eV

            +-----------+-----------------+-----------------+------------+-----+ <-- LBFGS
            | Parameter |      value      |    tolerance    |    units   | OK? | <-- LBFGS
            +-----------+-----------------+-----------------+------------+-----+ <-- LBFGS
            |  dE/ion   |   6.479173E-007 |   1.000000E-006 |         eV | Yes | <-- LBFGS
            |  |F|max   |   2.296894E-002 |   1.000000E-002 |       eV/A | No  | <-- LBFGS
            |  |dR|max  |   1.033643E-003 |   1.000000E-003 |          A | No  | <-- LBFGS
            +-----------+-----------------+-----------------+------------+-----+ <-- LBFGS
        """

        # TODO could be |S| instead of |F| and |dR| ?

        geom_iter_info = parse_castep_file_geom_iter_info(geom_iter_info_str)

        self.assertTrue(np.isclose(geom_iter_info['dE_per_ion'], 6.479173e-7))
        self.assertTrue(np.isclose(geom_iter_info['mag_F_max'], 2.296894e-2))
        self.assertTrue(np.isclose(geom_iter_info['mag_dR_max'], 1.033643e-3))

    def test_parse_castep_file_final_info(self):
        'Check end of run info parsing.'

        end_str = """  
            **********************************************************
            *** There were at least     2 warnings during this run ***
            *** => please check the whole of this file carefully!  ***
            **********************************************************
            
            ******************* Unconstrained Forces *******************
            *                                                          *
            *               Cartesian components (eV/A)                *
            * -------------------------------------------------------- *
            *                         x            y            z      *
            *                                                          *
            * Zr              1      0.00073      0.00101     -0.00055 *
            * Zr            102      0.00018      0.00002     -0.00114 *
            * I               1     -0.00110      0.00831     -0.00440 *
            * Cs              1     -0.00057     -0.00270      0.00262 *
            *                                                          *
            ************************************************************
            
            ******************************** Constrained Forces ********************************
            *                                                                                  *
            *                           Cartesian components (eV/A)                            *
            * -------------------------------------------------------------------------------- *
            *                         x                    y                    z              *
            *                                                                                  *
            * Zr              1      0.00073              0.00101             -0.00055         *
            * Zr            102      0.00018              0.00002             -0.00114         *
            * I               1     -0.00110              0.00831             -0.00440         *
            * Cs              1     -0.00057             -0.00270              0.00262         *
            *                                                                                  *
            ************************************************************************************
            
            Pseudo atomic calculation performed for Zr 4s2 4p6 4d2 5s2
            
            Converged in 30 iterations to a total energy of -1285.7947 eV
            
            
            Pseudo atomic calculation performed for I 5s2 5p5
            
            Converged in 19 iterations to a total energy of -794.3080 eV
            
            
            Pseudo atomic calculation performed for Cs 5s2 5p6 6s1
            
            Converged in 28 iterations to a total energy of -774.1356 eV
            
            Charge spilling parameter for spin component 1 = 0.09%
            
                Atomic Populations (Mulliken)
                -----------------------------
            Species          Ion     s      p      d      f     Total  Charge (e)
            =====================================================================
            Zr              1     2.42   6.73   2.86   0.00  12.00    -0.00
            Zr            102     2.42   6.73   2.86   0.00  12.00    -0.00
            I               1     1.95   4.80   0.00   0.00   6.75     0.25
            Cs              1     1.38   4.61   0.00   0.00   5.99     3.01
            =====================================================================
            
                            Bond                   Population      Length (A)
            ======================================================================
                        Zr 78  -- Zr 81                  0.34        2.86246
                        Zr 50  -- Zr 82                  0.26        2.96191
                        Zr 47  -- Zr 50                  0.26        2.98562
                        Zr 21  -- I 1                   -0.82        2.99163
            ======================================================================
            

            Writing analysis data to sim.castep_bin

            Writing model to sim.check
            
            A BibTeX formatted list of references used in this run has been written to 
            sim.bib
            
            Initialisation time =     84.43 s
            Calculation time    =  28599.24 s
            Finalisation time   =     92.02 s
            Total time          =  28775.69 s
            Peak Memory Use     = 3783120 kB
            
            Overall parallel efficiency rating: Good (77%)                                  
            
            Data was distributed by:-
            G-vector (2-way); efficiency rating: Very good (88%)                            
            k-point (8-way); efficiency rating: Very good (86%)                             

            """

        parse_castep_file_final_info(end_str)
