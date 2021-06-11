# VK_NV_inherited_viewport_scissor Sample

2020's take on the classic "fountain of exploding objects" demo;
objects are spawned at the origin with a random velocity and fall down
to their doom.

The primary purpose of this sample is to demonstrate how to integrate
[`VK_NV_inherited_viewport_scissor`](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#VK_NV_inherited_viewport_scissor)
as an <i>optional</i> extension. This extension enables subpass
secondary command buffers to inherit viewport/scissor state from the
calling primary command buffer (or another secondary command buffer).

The sample draws the same scene to four viewports on screen, with the
actual drawing done by secondary command buffers re-used across
frames. This is made possible by indirect draw commands, which allow
the secondary command buffers to adapt to dynamically-changing object
counts and positions. With the extension enabled, the sample is able
to use the same secondary command buffer for all 4 viewports, and
skip re-recording them even when the window or viewports are resized.

For a simple renderer like this, the CPU time saved by not
re-recording the command buffer is likely negligible, but the
techniques can be extended to more complicated renderers, where the
savings can be substantial.

[TUTORIAL LINK](https://nvpro-samples.github.io/vk_inherited_viewport/docs/inherited.md.html)

## Dependencies

The optional
[`VK_NV_inherited_viewport_scissor`](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#VK_NV_inherited_viewport_scissor)
extension is available in new [NVIDIA Vulkan
Drivers](https://developer.nvidia.com/vulkan-driver) for Windows 10 or
Linux. The header file for the extension is included in this
repository, so you do not need to update your Vulkan SDK.

However, validation layer support for the extension requires at least
[v1.2.178](https://github.com/KhronosGroup/Vulkan-ValidationLayers/releases/tag/v1.2.178).
As of 2021-07-05, this is included in the latest [official Vulkan SDK
Package](https://vulkan.lunarg.com/sdk/home).
<!-- Tested on Ubuntu 18.04 with Vulkan SDK 1.2.182.0 -->

## Build and Run

Clone https://github.com/nvpro-samples/nvpro_core.git
next to this repository (or pull latest `master` if you already have it)

`mkdir build && cd build && cmake .. # Or use CMake GUI`

If there are missing dependencies (e.g. glfw), run `git submodule
update --init --recursive --checkout --force` in the `nvpro_core`
repository.

Then start the generated `.sln` in VS or run `make -j`.

Run `vk_inherited_viewport` or `../../bin_x64/Release/vk_inherited_viewport.exe`

You are advised not to run the debug build unless you have the
required validation layers.

## LICENSE

Copyright 2021 NVIDIA CORPORATION. Released under Apache License,
Version 2.0. See "LICENSE" file for details.
