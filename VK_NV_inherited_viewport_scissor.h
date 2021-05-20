// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
//
// Declare stuff for VK_NV_inherited_viewport_scissor in the case where the real
// Vulkan header doesn't yet have it.
#ifndef VK_NV_inherited_viewport_scissor

#define VK_NV_inherited_viewport_scissor 1
#define VK_NV_INHERITED_VIEWPORT_SCISSOR_SPEC_VERSION                1
#define VK_NV_INHERITED_VIEWPORT_SCISSOR_EXTENSION_NAME              "VK_NV_inherited_viewport_scissor"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INHERITED_VIEWPORT_SCISSOR_FEATURES_NV ((VkStructureType)1000278000)
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_VIEWPORT_SCISSOR_INFO_NV ((VkStructureType)1000278001)

typedef struct VkPhysicalDeviceInheritedViewportScissorFeaturesNV {
    VkStructureType                       sType;
    void*                                 pNext;
    VkBool32                              inheritedViewportScissor2D;
} VkPhysicalDeviceInheritedViewportScissorFeaturesNV;

typedef struct VkCommandBufferInheritanceViewportScissorInfoNV {
    VkStructureType                       sType;
    const void*                           pNext;
    VkBool32                              viewportScissor2D;
    uint32_t                              viewportDepthCount;
    const VkViewport*                     pViewportDepths;
} VkCommandBufferInheritanceViewportScissorInfoNV;

#endif
