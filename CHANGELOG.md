# Changelog

All notable changes to vesin are documented here, following the [keep a
changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/luthaf/vesin/)

<!-- Possible sections for each package:

### Added

### Fixed

### Changed

### Removed
-->

### Added

- vesin can now compute neighbor lists on GPUs using CUDA. This is automatically
  integrated in vesin-torch, and can be used with CuPy in the `vesin` python
  package.
- vesin now offers a Fortran API, you can enable it by giving the `-DVESIN_FORTRAN=ON` option to cmake (#50)
- the single file build now contains a comment mentionning which version of
  vesin the file corresponds to.

### Changed

- The `periodic` argument to `vesin_neighbors()` in C and C++,
  `NeighborList.compute()` in all other languages can now be set separatly for
  each box dimension, allowing mixed periodic and non-periodic boundary
  conditions.
- `VesinDevice` in C and C++ is now a struct containing both a device kind (i.e.
  CPU, CUDA, etc.) and a device index.
