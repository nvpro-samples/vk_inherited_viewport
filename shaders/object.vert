// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#version 460
#extension GL_GOOGLE_include_directive : require

#include "BindingNumbers.h"
#include "CameraTransforms.h"
#include "InstanceData.h"
#include "VertexData.h"

layout(location=POSITION_ATTRIB_LOCATION) in vec4 i_position;
layout(location=NORMAL_ATTRIB_LOCATION)   in vec4 i_normal;
layout(location=COLOR_ATTRIB_LOCATION)    in vec4 i_color;

layout(location=0) out vec3 v_worldNormal;
layout(location=1) out vec3 v_worldPosition;
layout(location=2) out vec4 v_color;

layout(set=0, binding=CAMERA_TRANSFORMS_BINDING) uniform CameraTransformsBuffer
{
  CameraTransforms cameraTransforms;
};

layout(set=0, binding=INSTANCES_BINDING) readonly restrict buffer InstancesBuffer
{
  InstanceData instances[];
};

void main()
{
    mat4 M = instances[gl_InstanceIndex].modelMatrix;
    mat4 V = cameraTransforms.view;
    mat4 P = cameraTransforms.proj;

    v_worldPosition = (M * i_position).xyz;
    gl_Position     = P * V * vec4(v_worldPosition, 1);
    v_worldNormal   = mat3(M) * i_normal.xyz;
    v_color         = i_color;
}
