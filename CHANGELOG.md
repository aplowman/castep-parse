# Change Log

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
