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

## [Version 0.5.6](https://github.com/Luthaf/vesin/releases/tag/v0.5.6) - 2026-05-05

### Added

- The CUDA implementation now support the `sorted` option

### Changed

- When passing `sorted=true`, pairs are now only sorted by the index of the
  first point (`i`). This is faster and enough in most cases.

## [Version 0.5.5](https://github.com/Luthaf/vesin/releases/tag/v0.5.5) - 2026-04-20

### Fixed

- Fixed a bug when using `copy=False` in vesin.metatomic

### Changed

- Tweaked parameters for the cuda implementation to make the code faster
- Change the default number of pairs we allocate for in the CUDA implementation,
  fewer people should have to manually override the default value

## [Version 0.5.4](https://github.com/Luthaf/vesin/releases/tag/v0.5.4) - 2026-04-02

### Fixed

- The cell list algorithm are now properly O(N) even in the case of non-periodic
  dimensions. The code previously fell back to a O(N^2) implementation in this
  case.

### Added

- `vesin.torch.NeighborList` can now be saved with `torch.script.save`,
  including when used as attribute of a model
- The `vesin-torch` wheels on PyPI are now compatible with PyTorch v2.11

### Changed

- The main API for compatibility with metatomic models is now
  `vesin.metatomic.neighbor_lists_for_model`, which allows re-using the same
  calculator across multiple simulation steps. The previous API
  (`compute_requested_neighbors` and `compute_requested_neighbors_from_options`)
  is now deprecated.

## [Version 0.5.3](https://github.com/Luthaf/vesin/releases/tag/v0.5.3) - 2026-03-10

### Fixed

- We now use the same cudart library from Python and C++ when using torch
  tensors with `vesin.NeighborList`.

## [Version 0.5.2](https://github.com/Luthaf/vesin/releases/tag/v0.5.2) - 2026-02-26

### Fixed

- The results are now returned with the same dtype as the input even when
  we return an empty neighbor list

## [Version 0.5.1](https://github.com/Luthaf/vesin/releases/tag/v0.5.1) - 2026-02-24

### Added

- Added `vesin.metatomic.compute_requested_neighbors_from_options` to allow
  computing all neighbor lists requested by a metatomic model in a
  TorchScript-compatible way.

### Changed

- Users can control how much memory is allocated for the CUDA neighbor list with
  the `VESIN_CUDA_MAX_PAIRS_PER_POINT` environment variable.
- We now also try to load `libcudart.so.{11,12,13}` if the code can not find
  `libcudart.so` on Linux.


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
