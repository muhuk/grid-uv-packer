# Grid UV Packer

![grid-uv-packer logo](./img/grid-uv-packer-logo.png)

**This addon is experimental.  Feel free to download and test it.  Your
feedback is welcome.**

A pure-Python UV packer that packs irregularly shaped UV islands efficiently.

grid-uv-packer is released with [GPL license](./COPYING.txt), same as
[Blender](https://www.blender.org/about/license/)

## Features

**TBD**

## Installation

- Download the source zip from [releases
  page](https://github.com/muhuk/grid-uv-packer/releases).  You need to click
  to expand `Assets` under the correct version and download the file named
  `guvp-#.#.zip` (where #.# is the version number)
- In Blender, open `Preferences` editor (`Edit -> Preferences`
  from the menu).
- Click `Install...` button and choose the zip file you have downloaded.
- Once the addon is installed, you **must** still enable it manually in
  `Preferences` view.  It is not automatically enabled.
- After installation, you can delete the zip file you have downloaded.

## Usage

Once you enabled the addon, in UV editor a `Grid UV Pack` command will be
available in `UV -> Unwrap` menu (`U` key by default).  This command will
create a modal dialog where you can adjust settings like grid size. Your UV's
will be modified once you click `OK` on this modal dialog.
