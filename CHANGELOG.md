# Changelog

## [Unreleased]

### Added

- UDIM support is added.  `Pact to` options `Active UDIM` and `Closest UDIM`
  should work same as `py.ops.uv.pack_islands()`.
- You can interrupt calculation via `ESC` key.  Note that this does not cancel
  the operation.  If a solution with better coverage is already found, UVs
  will be updated.  (You can undo the operator)

### Changed

- Moved from `Unwrap` menu to `UV`
  menu. [b659c64](https://github.com/muhuk/grid-uv-packer/commit/37f5e510f557d3d6eb7c1956eb2515575b659c64)
- When `Max Iterations` is set to `1` packer is run single threaded.  Note
  that you need a lot more than one iteration to get decent results, this is
  intended for
  debugging. [8f36daf](https://github.com/muhuk/grid-uv-packer/commit/06f16e6ce46babb7420e6b2053cf9bc038f36daf)
- Previously any result with 20% coverage or better was considered successful
  and written to UVs.  Now only the results with better coverage than input
  UVs are written. [fc949cb](https://github.com/muhuk/grid-uv-packer/commit/df5e0f9bc89937fb120cd71926dedbea7fc949cb)

### Fixed

- Fixed an issue where rotated islands were overlapping with other
  islands. [6ae83fe](https://github.com/muhuk/grid-uv-packer/commit/f17f337423dbe9ffad4e4641b6be9a5ab6ae83fe)
- Earlier packing tended to produce results that cover a rectangular area
  (mostly x axis, y axis underutilized) instead of the whole UV square.  This
  is fixed with
  [e107900](https://github.com/muhuk/grid-uv-packer/commit/e85f809bc52e5fc591e191cc990ccaf6ee107900)
  and
  [3863a48](https://github.com/muhuk/grid-uv-packer/commit/4ce97666e49fcaabe248a31e00beb54073863a48).

### Removed

- Removed baseline fitness calculation.  This was not something users were
  seeing but before UDIM support, if all islands were not in UV unit square
  the operator gave an error.  Otherwise a baseline fitness was being
  calculated and the UVs were only updated if the grid packer's result was
  better than the original UV map.  When the UDIM support added calculating
  the baseline fitness became difficult so it was removed.  Now the UVs are
  updated provided that all islands are placed on the grid (a solution is
  found).  If the result is not good, you can undo.

## [0.3] - 2022-12-29

### Added

- Added setting for maximum number of iterations.
- Added setting for maximum run time.  Set this to `0` to disable run time
  limit.
- Implemented island rotation.  This setting is on by default, you can
  distable it by unchecking `rotate`.

### Changed

- Minimum required version of Blender is now `3.4.0`.

## [0.2] - 2022-12-15

### Added
- Made random seed configurable.  Set seed to `0` to get a random
  seed. ([57d1536](https://github.com/muhuk/grid-uv-packer/commit/e17443220fb5f74daaffddead07c389f657d1536))

### Changed
- When UV islands are out of bounds `grid_pack` command will be cancelled
  instead of throwing an
  exception. ([0d2dc3a](https://github.com/muhuk/grid-uv-packer/commit/ecb3ea902b6cd78ffea7b746060d5dd230d2dc3a))
- Improved coverage (fitness)
  ([68adb5c](https://github.com/muhuk/grid-uv-packer/commit/49f6bd25592c167362d19a2952509038a68adb5c))
  calculation & margin calculation
  ([4dc76e8](https://github.com/muhuk/grid-uv-packer/commit/d69f0015395c33149028cadb80e448eca4dc76e8)).

### Fixed
- Fixed the non-breaking exception when parsing
  `bl_info`. ([7e1fcd9](https://github.com/muhuk/grid-uv-packer/commit/214566e9cac4113241374e20ba3631dc37e1fcd9))

## [0.1] - 2022-12-13

### Added
- Initial proof of concept release.  Not ready for production use.

[unreleased]: https://github.com/muhuk/grid-uv-packer/compare/v0.3...HEAD
[0.3]: https://github.com/muhuk/grid-uv-packer/compare/v0.2...v0.3
[0.2]: https://github.com/muhuk/grid-uv-packer/compare/v0.1...v0.2
[0.1]: https://github.com/muhuk/grid-uv-packer/releases/tag/v0.1
