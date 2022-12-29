# Changelog

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

[0.2]: https://github.com/muhuk/grid-uv-packer/compare/v0.2...v0.3
[0.2]: https://github.com/muhuk/grid-uv-packer/compare/v0.1...v0.2
[0.1]: https://github.com/muhuk/grid-uv-packer/releases/tag/v0.1
