// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#version 460
#extension GL_GOOGLE_include_directive : enable

#include "skybox.glsl"

#include "CameraTransforms.h"

layout(location=0) in vec3 v_worldNormal;
layout(location=1) in vec3 v_worldPosition;
layout(location=2) in vec4 v_color;

layout(set=0, binding=0) uniform UniformBuffer
{
  CameraTransforms cameraTransforms;
};

layout(location=0) out vec4 out_color;

vec3 resistor_colors[10] = vec3[] (
    vec3(0.2, 0.2, 0.2),
    vec3(0.5, 0.2, 0.0),
    vec3(1.0, 0.0, 0.0),
    vec3(1.0, 0.5, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(0.0, 0.5, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(0.5, 0.0, 0.8),
    vec3(0.5, 0.5, 0.5),
    vec3(1.0, 1.0, 1.0));

void main()
{
  // Calculate the reflection vector for the given camera position.
  vec3 normal    = normalize(v_worldNormal);
  vec3 origin    = (cameraTransforms.viewInverse * vec4(0, 0, 0, 1)).xyz;
  vec3 direction = v_worldPosition - origin;
  vec3 reflected = reflect(direction, normal);

  // Sample reflection color darkened by specular brightness.
  vec3 reflectColor = sampleSkybox(reflected) * v_color.a;

  // Add base color with reflection color.
  out_color = vec4(v_color.xyz + reflectColor, 1.0);

  // Debug depth by resister color codes.
  if (cameraTransforms.colorByDepth != 0)
  {
    int i = clamp(int(gl_FragCoord.z * 10), 0, 9);
    out_color = vec4(resistor_colors[i], 1.0);
  }
}
