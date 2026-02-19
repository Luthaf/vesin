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

## [Version 0.5.0](https://github.com/Luthaf/vesin/releases/tag/v0.5.0) - 2026-02-19

### Added

- vesin can now compute neighbor lists on GPUs using CUDA. This is automatically
  integrated in vesin-torch, and can be used with CuPy in the `vesin` python
  package.
- `vesin.NeighborList` now accepts either numpy arrays, cupy arrays or torch
  tensor for the `points`, `box`, and `periodic` parameters.

## [Version 0.4.2](https://github.com/Luthaf/vesin/releases/tag/v0.4.2) - 2025-11-06

### Changed

- arbitrary box orientations are now supported with mixed PBC (#88)


## [Version 0.4.1](https://github.com/Luthaf/vesin/releases/tag/v0.4.1) - 2025-11-03

### Added

- `vesin-torch` wheels on PyPI now support PyTorch v2.9

## [Version 0.4.0](https://github.com/Luthaf/vesin/releases/tag/v0.4.0) - 2025-10-27

### Added

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
