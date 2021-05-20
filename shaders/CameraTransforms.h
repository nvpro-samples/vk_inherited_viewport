// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
// Polyglot include file (GLSL and C++) for the camera transforms struct.

#ifndef VK_NV_INHERITED_SCISSOR_VIEWPORT_CAMERA_TRANSFORMS_H_
#define VK_NV_INHERITED_SCISSOR_VIEWPORT_CAMERA_TRANSFORMS_H_

#ifdef __cplusplus
#include <nvmath/nvmath_glsltypes.h> // emulate glsl types in C++
#include <stdint.h>
#endif

struct CameraTransforms
{
  #ifdef __cplusplus
    using mat4 = nvmath::mat4;
    using uint = uint32_t;
  #endif

  // View and projection matrices, along with their inverses.
  mat4 view;
  mat4 proj;
  mat4 viewInverse;
  mat4 projInverse;

  // If nonzero, color fragments based on their depth value.
  uint colorByDepth;
};

#endif /* !VK_NV_INHERITED_SCISSOR_VIEWPORT_CAMERA_TRANSFORMS_H_ */
