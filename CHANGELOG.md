# Change Log

## [0.2.1] - 2019.xx.xx

## Fixed

- Fixed white space issue in parsing forces block in .castep file.
- Fixed bug in parsing .castep file when geometry <frequency> is unchanged from its initial value.
- Fixed bug in parsing .castep file when there is an initial SCF cycle on continuation of a geometry optimisation.
- Only add final info to run `dict` if the final info includes the total time string.
- Fixed issue when adding up total time if no previous SCF cycle to get last time from.
- Skip runs if they don't include the string "Cell Contents"

## [0.2.0] - 2019.11.22

## Changed

- `read_castep_file` is now more logical and testable (but less-performant). Instead of iterating over lines in the .castep file, it splits it into sections and parsing individual blocks. This means it handles much better the situation a CASTEP run has been continued and the output is appended to the same .castep file as the original run.
- Use of `flexible_open` decorator is now more limited but better defined. In particular, the functions `read_castep_file`, `read_geom_file` and `read_cell_file` accept as their "file" input argument one of these types: `str`, `pathlib.Path`, `bytes` or `TextIOWrapper`. If a string or `Path` object, it is assumed to be the file path.

## Added

- `read_relaxation` function
- `merge_geom_data` function to merge data from .geom file with that from .castep file.

## [0.1.4] - 2019.08.24

## Changed

- `read_castep_file`, `read_geom_file` and `read_cell_file` now additionally accept input as bytes.

## [0.1.3] - 2019.08.24

## Changed

- `read_castep_file`, `read_geom_file` and `read_cell_file` now accept any of a file handle, string or Path object.

## Fixed

- Fix bug with `write_input_files` when `param` argument is missing.

## [0.1.2] - 2019.06.02

### Fixed

- Fix encoding in function to get `long_description`.

## [0.1.1] - 2019.06.02

### Changed

- Added `long_description` parameter to `setuptools.setup`.
