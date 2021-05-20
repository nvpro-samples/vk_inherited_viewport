# VK_NV_inherited_viewport_scissor Sample

2020's take on the classic "fountain of exploding objects" demo;
objects are spawned at the origin with a random velocity and fall down
to their doom.

This sample provides a simple example of how a subpass secondary
command buffer can be re-used across frames. If the sample detects the
optional `VK_NV_inherited_viewport_scissor` extension, it uses it to
re-use the secondary command buffer even after a window
resize. Indirect draw commands allow the secondary command buffer to
be re-used despite the dynamically-changing object counts and
positions. For a simple renderer like this, the CPU time saved by not
re-recording the command buffer is likely negligible, but the
techniques can be extended to more complicated renderers, where the
savings can be substantial.

[Tutorial Link](./docs/inherited.md.html)

## Build and Run

Clone https://github.com/nvpro-samples/nvpro_core.git
next to this repository (or pull latest `master` if you already have it)
TODO may become `main` branch.

If needed, run `git submodule update --recursive` in `nvpro_core`.

`mkdir build && cd build && cmake .. # Or use CMake GUI`

Then start the generated `.sln` in VS or run `make -j`.

Run `vk_inherited_viewport` or `../../bin_x64/Release/vk_inherited_viewport.exe`

I advise you to not run the Debug build unless you have my patched Vulkan
validation layers TODO remove notice once patches are accepted & shipped.

## LICENSE

Copyright 2021 NVIDIA CORPORATION. Released under Apache License,
Version 2.0. See "LICENSE" file for details.
