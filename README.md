[![PyPI version](https://badge.fury.io/py/castep-parse.svg)](https://badge.fury.io/py/castep-parse)

# castep-parse
Input file writers and output file readers for the density functional theory code CASTEP.

## Installation

`pip install castep-parse`

## Notes

This code has generally been used for `task=singlepoint` and `task=geometryoptimisation` simulations. Consequently, input and output files associated with other simulation types are not yet supported.

## Functionality

### Readers:

- ✅ `read_castep_file`
- ️✅ `read_geom_file`
- ✅ `read_cell_file`
- ✅ `read_output_files`

### Writers:

- ✅ `write_cell_file`
- ✅ `write_input_files`

### Utilities:

- ✅ `map_species_to_castep`
- ✅ `get_castep_cell_constraints`

## Examples

### Do something useful
