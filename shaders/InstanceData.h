// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
// Polyglot include file (GLSL and C++) for instanced draw data.

#ifndef VK_NV_INHERITED_SCISSOR_VIEWPORT_INSTANCE_DATA_H_
#define VK_NV_INHERITED_SCISSOR_VIEWPORT_INSTANCE_DATA_H_

struct InstanceData
{
  #ifdef __cplusplus
    using mat4 = glm::mat4;
  #endif

  mat4 modelMatrix;
};

#endif /* !VK_NV_INHERITED_SCISSOR_VIEWPORT_INSTANCE_DATA_H_ */

