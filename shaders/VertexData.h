// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
// Polyglot include file (GLSL and C++) for vertex data.

#ifndef VK_NV_INHERITED_SCISSOR_VIEWPORT_VERTEX_DATA_H_
#define VK_NV_INHERITED_SCISSOR_VIEWPORT_VERTEX_DATA_H_

#ifdef __cplusplus
#include <array>
#include <vulkan/vulkan.h>
#include "nvmath/nvmath_glsltypes.h" // emulate glsl types in C++
#endif

#define POSITION_ATTRIB_LOCATION 0
#define NORMAL_ATTRIB_LOCATION   1
#define COLOR_ATTRIB_LOCATION    2

struct VertexData
{
  #ifdef __cplusplus
    using vec4 = nvmath::vec4;
  #endif

  vec4 position;
  vec4 normal;   // w ignored.
  vec4 color;    // a is specular reflection brightness.

#ifdef __cplusplus
  static constexpr auto vec4format = VK_FORMAT_R32G32B32A32_SFLOAT;
  static std::array<VkVertexInputAttributeDescription, 3> getAttributes();
#endif
};

#ifdef __cplusplus
std::array<VkVertexInputAttributeDescription, 3> VertexData::getAttributes()
{
  using A = VkVertexInputAttributeDescription;
  return {
    A{POSITION_ATTRIB_LOCATION, 0, vec4format, offsetof(VertexData, position)},
    A{NORMAL_ATTRIB_LOCATION, 0, vec4format, offsetof(VertexData, normal)},
    A{COLOR_ATTRIB_LOCATION, 0, vec4format, offsetof(VertexData, color)}};
};
#endif

#endif /* !VK_NV_INHERITED_SCISSOR_VIEWPORT_VERTEX_DATA_H_ */

