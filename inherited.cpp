// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
//
// Host code for vk_inherited_viewport sample.
// See README.md or docs/inherited.md.html for commentary.
#include <algorithm>
#include <array>
#include <cassert>
#include <math.h>
#include <random>
#include <string.h>
#include <vector>
#include <vulkan/vulkan.h>
#include "VK_NV_inherited_viewport_scissor.h"
#include "GLFW/glfw3.h"

#include "nvh/fileoperations.hpp"  // For nvh::findFile
#include "nvmath/nvmath.h"         // For vec4 operator overloads
#include "nvmath/nvmath_glsltypes.h"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvpsystem.hpp"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "imgui_helper.h"

#include "FrameManager.hpp"

// GLSL polyglots
#include "shaders/BindingNumbers.h"
#include "shaders/CameraTransforms.h"
#include "shaders/InstanceData.h"
#include "shaders/VertexData.h"

// Whether the device has the VK_NV_inherited_viewport_scissor
// extension, and whether the user wants it in-use.
static bool has_VK_NV_inherited_viewport_scissor = false;
static bool use_VK_NV_inherited_viewport_scissor = false;

static bool forceRerecordSecondaryCmdBuf = false;

static bool vsync = true;

// Camera parameters (global)
static nvmath::vec3f cameraPosition;
static float         cameraTheta = 0.0f, cameraPhi = 1.5f;   // Radians
static float         cameraFOV   = 60.0f;                    // Degrees
constexpr float      cameraNear  = 0.8f, cameraFar = 100.f;
static bool          colorByDepth = false;
static float         minDepth = 0.0f, maxDepth = 1.0f;

// Updated from the above.
static nvmath::vec3f cameraForward;

// Whether the Dear Imgui window is open.
static bool guiVisible = true;

// Whether we should continuously vary the viewport size to test
// viewport resize handling.
static bool wobbleViewport = false;

// Wether we should be printing framebuffer / secondary command buffer
// recreations to stdout.
static bool printEvents = true;

// Limits
#define MAX_INSTANCES (1 << 15)
#define MAX_MODELS    (1 <<  4)
#define MAX_INDICES   (1 << 20)
#define MAX_VERTICES  (1 << 18)

// Parameters for simple animation (objects spawned, fall with gravity, and get
// destroyed below death height).
static const nvmath::vec3f spawnOrigin     = nvmath::vec3f(12.0f, -3.0f, 0.0f);
static constexpr float spawnSpeed          = 8.0f;
static constexpr float gravityAcceleration = 2.0f;
static constexpr int   ticksPerSecond      = 256;
static constexpr float secondsPerTick      = 1.0f / ticksPerSecond;
static constexpr int   ticksPerNewInstance = 8;
static constexpr float deathHeight         = -100.0f;
static constexpr int   maxTicksPerFrame    = 16;
static constexpr float maxTurnsPerSecond   = 1.4f;
static bool            paused              = false;

// Some magic that interacts with our CMake setup, basically this
// figures out the location of the original source file directory
// (and shader directory) relative to the executable file's location.
//
// This is used by nvh::findFile to search for shader files.
static const std::vector<std::string> searchPaths = {
  NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY),
  NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY "shaders/"),
  PROJECT_RELDIRECTORY,
  PROJECT_RELDIRECTORY "shaders/"};


// Get the framebuffer size for the given glfw window; suspend until the glfw
// window has nonzero size (i.e. not minimized).
void waitNonzeroFramebufferSize(GLFWwindow* pWindow,
                                uint32_t*   pWidth,
                                uint32_t*   pHeight)
{
  int width, height;
  glfwGetFramebufferSize(pWindow, &width, &height);
  while (width == 0 || height == 0)
  {
    glfwWaitEvents();
    glfwGetFramebufferSize(pWindow, &width, &height);
  }
  *pWidth  = uint32_t(width);
  *pHeight = uint32_t(height);
}


// Data that doesn't change after upload to device.
struct StaticBufferData
{
  uint32_t   indices[MAX_INDICES];
  VertexData vertices[MAX_VERTICES];

  static constexpr VkIndexType indexType = VK_INDEX_TYPE_UINT32;
};

// Data that can change per-frame, stored in one buffer.
struct DynamicBufferData
{
  int unused;  // guard for offsetof mistakes

  // indirectCmds[i] holds info for drawing all instances of model i. Fill
  // unused entries with 0s. vertexOffset defines the location in
  // StaticBufferData::vertices that the model's verts are stored, and
  // firstIndex and indexCount defines the section in
  // StaticBufferData::indices.
  alignas(1024) VkDrawIndexedIndirectCommand indirectCmds[MAX_MODELS];

  // Array for storing instance data (model matrix, etc.)
  // Subsections are used for different models, defined by
  // indirectCmds[i].firstInstance and instanceCount.
  alignas(1024) InstanceData instances[MAX_INSTANCES];

  // Camera transformation matrices, bound as UBO.
  alignas(1024) CameraTransforms cameraTransforms;
};

// Instance data, plus extra data needed for animation.
struct AnimatedInstanceData : InstanceData
{
  nvmath::vec3f position;
  nvmath::vec3f velocity;
  nvmath::vec3f rotationAxis;
  float         turnsPerSecond;
  float         rotationAngle;
};


// Simple model, list of vertices and index buffer.
struct Model
{
  std::vector<uint32_t>   indices;
  std::vector<VertexData> vertices;
};

// Information needed to locate the vertices etc. of a model, plus its list of
// instances.
struct ModelDrawInfo
{
  // Model indices stored in this section of StaticBufferData::indices
  uint32_t firstIndex;
  uint32_t indexCount;

  // Vertex indices stored starting at this index in StaticBufferData::vertices
  int32_t vertexOffset;

  // Section of DynamicBufferData::instances reserved for this model's instances
  uint32_t firstInstance;
  uint32_t maxInstances;

  // List of instances. size() must be no more than maxInstances.
  std::vector<AnimatedInstanceData> instances;
};

static std::vector<Model> makeModels();


// Object that manages the lifetimes of the buffers used for the sample.
// Includes utility code for copying from staging buffers to device,
// and for synchronizing access to staging with VkFence.
class ScopedBuffers
{
  VkDevice                         m_device;
  nvvk::ResourceAllocatorDedicated m_allocator;

  // Device buffer holding StaticBufferData, its staging buffer and
  // mapped staging buffer pointer.
  nvvk::Buffer    m_staticBuffer;
  nvvk::Buffer    m_staticStaging;
  StaticBufferData*        m_staticStagingPtr;

  // Usage of the static buffer.
  static constexpr VkBufferUsageFlags staticUsage =
    VK_BUFFER_USAGE_TRANSFER_DST_BIT
    | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  // Same, but for DynamicBufferData.
  nvvk::Buffer    m_dynamicBuffer;
  nvvk::Buffer    m_dynamicStaging;
  DynamicBufferData*       m_dynamicStagingPtr;

  static constexpr VkBufferUsageFlags dynamicUsage =
    VK_BUFFER_USAGE_TRANSFER_DST_BIT
    | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

  // Fence for synchronizing access to dynamic staging buffer.
  // Signalled when the staging->device transfer done, wait for
  // signal before modifying staging buffer.
  VkFence m_dynamicStagingFence;

public:
  ScopedBuffers(nvvk::Context& ctx)
  {
    m_device = ctx;
    m_allocator.init(ctx, ctx.m_physicalDevice);

    m_staticBuffer = m_allocator.createBuffer(
      sizeof(StaticBufferData),
      staticUsage,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_staticStaging = m_allocator.createBuffer(
      sizeof(StaticBufferData),
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
      | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    m_staticStagingPtr = static_cast<StaticBufferData*>(
      m_allocator.map(m_staticStaging));

    m_dynamicBuffer = m_allocator.createBuffer(
      sizeof(DynamicBufferData),
      dynamicUsage,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_dynamicStaging = m_allocator.createBuffer(
      sizeof(DynamicBufferData),
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
      | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    m_dynamicStagingPtr = static_cast<DynamicBufferData*>(
      m_allocator.map(m_dynamicStaging));

    VkFenceCreateInfo fenceInfo = {
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr,
        VK_FENCE_CREATE_SIGNALED_BIT };
    NVVK_CHECK(
        vkCreateFence(m_device, &fenceInfo, nullptr, &m_dynamicStagingFence));
  }

  ~ScopedBuffers()
  {
    vkDestroyFence(m_device, m_dynamicStagingFence, nullptr);
    m_allocator.destroy(m_staticBuffer);
    m_allocator.destroy(m_staticStaging);
    m_allocator.destroy(m_dynamicBuffer);
    m_allocator.destroy(m_dynamicStaging);
    m_allocator.deinit();
  }

  VkBuffer getStaticBuffer() const
  {
    return m_staticBuffer.buffer;
  }

  StaticBufferData* getStaticStagingPtr() const
  {
    return m_staticStagingPtr;
  }

  // Record a command for transferring static buffer data from staging
  // to device.  No synchronization done.
  void cmdTransferStaticStaging(VkCommandBuffer cmdBuf) const
  {
    VkBufferCopy bufferCopy { 0, 0, sizeof(StaticBufferData) };
    vkCmdCopyBuffer(
        cmdBuf, m_staticStaging.buffer, m_staticBuffer.buffer, 1, &bufferCopy);
  }

  VkBuffer getDynamicBuffer() const
  {
    return m_dynamicBuffer.buffer;
  }

  // Return the staging buffer pointer. Use the fence below to
  // synchronize access.
  DynamicBufferData* getDynamicStagingPtr() const
  {
    return m_dynamicStagingPtr;
  }

  // Get the said fence. Initially in a signalled state.
  VkFence getFence() const
  {
    return m_dynamicStagingFence;
  }

  // Record a command for transferring dynamic buffer data from
  // staging to device. Includes barriers before and
  // after for the given stage and access flags.
  void cmdTransferDynamicStaging(VkCommandBuffer      cmdBuf,
                                 VkPipelineStageFlags stageFlags,
                                 VkAccessFlags        accessFlags) const
  {
    VkBufferMemoryBarrier beforeBarrier {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        nullptr,
        accessFlags,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        0, 0,
        m_dynamicBuffer.buffer,
        0, VK_WHOLE_SIZE };
    vkCmdPipelineBarrier(
        cmdBuf,
        stageFlags,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 1, &beforeBarrier, 0, nullptr);

    VkBufferCopy bufferCopy { 0, 0, sizeof(DynamicBufferData) };
    vkCmdCopyBuffer(cmdBuf, m_dynamicStaging.buffer, m_dynamicBuffer.buffer,
                    1, &bufferCopy);

    VkBufferMemoryBarrier afterBarrier {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        accessFlags,
        0, 0,
        m_dynamicBuffer.buffer,
        0, VK_WHOLE_SIZE };
    vkCmdPipelineBarrier(
        cmdBuf,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        stageFlags,
        0, 0, nullptr, 1, &afterBarrier, 0, nullptr);
  }
};


// Manager for framebuffers + shared depth buffer, one per swap chain
// image.  Adapted from https://github.com/Overv/VulkanTutorial under
// CC0 1.0 Universal license (public domain).
//
// Also adapted from Myricube (GPL licensed) BUT I (dakeley@nvidia.com)
// wrote it so it's okay (I'm exempt from my own license).
class Framebuffers
{
  // From nvvk::SwapChain::getChangeID().  Basically, if this
  // doesn't match that of nvvk::SwapChain, the swap chain has been
  // re-created, and we need to re-create the framebuffers here to
  // match.
  uint32_t m_lastChangeId;

  // Borrowed device pointer and render pass.
  VkDevice     m_device;
  VkRenderPass m_renderPass;

  // Shared depth buffer and its format, memory, and ImageView.
  VkFormat             m_depthFormat;
  nvvk::Image m_depthImage;
  VkImageView          m_depthView;

  // Extent used to create framebuffer.
  VkExtent2D m_extent{};

  // Allocator for creating the above.
  nvvk::ResourceAllocatorDedicated m_allocator;

  // framebuffer[i] is the framebuffer for swap image i, as you'd
  // expect.  This is cleared to indicate when this class is in an
  // unitinialized state.
  std::vector<VkFramebuffer> m_framebuffers;

public:
  bool initialized() const { return !m_framebuffers.empty(); }

  Framebuffers(VkPhysicalDevice physicalDevice,
               VkDevice         device,
               VkRenderPass     renderPass,
               VkFormat         depthFormat)
      : m_device(device)
      , m_renderPass(renderPass)
      , m_depthFormat(depthFormat)
  {
    m_allocator.init(device, physicalDevice);
  }

  ~Framebuffers()
  {
    destroyFramebuffers();
    m_allocator.deinit();
  }

  // Check the swap chain and recreate now if needed (now = no
  // synchronization done; note however that we can rely on
  // FrameManager to wait on the main thread queue to idle before
  // re-creating a swap chain).
  void recreateNowIfNeeded(nvvk::SwapChain& swapChain) noexcept
  {
    if (initialized() && swapChain.getChangeID() == m_lastChangeId)
    {
        return;
    }

    // Destroy old resources.
    destroyFramebuffers();

    auto width  = swapChain.getWidth();
    auto height = swapChain.getHeight();
    m_extent = { width, height };

    // Make depth buffer.
    VkImageCreateInfo depthCreateInfo {
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0,
        VK_IMAGE_TYPE_2D, m_depthFormat,
        { width, height, 1 },
        1, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, nullptr,
        VK_IMAGE_LAYOUT_UNDEFINED };

    m_depthImage = m_allocator.createImage(
        depthCreateInfo,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Make depth view.
    VkImageViewCreateInfo viewInfo{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0,
        m_depthImage.image, VK_IMAGE_VIEW_TYPE_2D, m_depthFormat,
        {}, { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}};
    NVVK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_depthView));

    // Make a framebuffer for every swap chain image.
    uint32_t imageCount = swapChain.getImageCount();
    m_framebuffers.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; ++i)
    {
      std::array<VkImageView, 2> attachments = {
          swapChain.getImageView(i), m_depthView };

      VkFramebufferCreateInfo framebufferInfo{};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferInfo.renderPass = m_renderPass;
      framebufferInfo.attachmentCount =
        static_cast<uint32_t>(attachments.size());
      framebufferInfo.pAttachments = attachments.data();
      framebufferInfo.width = width;
      framebufferInfo.height = height;
      framebufferInfo.layers = 1;

      NVVK_CHECK(vkCreateFramebuffer(
        m_device, &framebufferInfo, nullptr, &m_framebuffers.at(i)));
    }

    m_lastChangeId = swapChain.getChangeID();
    if (printEvents)
    {
      printf(
          "\x1b[33mRecreated framebuffer:\x1b[0m "
          "m_lastChangeId=%i width=%i height=%i\n",
          int(m_lastChangeId), int(width), int(height));
    }
  }

  void destroyFramebuffers()
  {
    if (initialized())
    {
      vkDestroyImageView(m_device, m_depthView, nullptr);
      m_allocator.destroy(m_depthImage);
      for (VkFramebuffer fb : m_framebuffers)
      {
        vkDestroyFramebuffer(m_device, fb, nullptr);
      }
      m_framebuffers.clear();
    }
    assert(!initialized());
  }

  VkFramebuffer operator[] (size_t i) const
  {
    return m_framebuffers.at(i);
  }

  VkExtent2D getExtent() const
  {
    return m_extent;
  }
};


// Class for managing the simple one-subpass, depth buffer VkRenderPass.
class ScopedRenderPass
{
  // Managed by us.
  VkRenderPass m_renderPass;

  // Borrowed device pointer.
  VkDevice m_device;

public:
  ScopedRenderPass(VkDevice device, VkFormat colorFormat, VkFormat depthFormat)
  {
    m_device = device;

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format         = colorFormat;
    colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format         = depthFormat;
    depthAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass   = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass   = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                              | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                               | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                               | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                               | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                              | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                              | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = dependency.srcAccessMask;

    std::array<VkAttachmentDescription, 2> attachments = {colorAttachment,
                                                          depthAttachment};
    VkRenderPassCreateInfo                 renderPassInfo{};
    renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments    = attachments.data();
    renderPassInfo.subpassCount    = 1;
    renderPassInfo.pSubpasses      = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies   = &dependency;

    NVVK_CHECK(
      vkCreateRenderPass(device, &renderPassInfo, nullptr, &m_renderPass));
  }

  ScopedRenderPass(ScopedRenderPass&&) = delete;

  ~ScopedRenderPass()
  {
    vkDestroyRenderPass(m_device, m_renderPass, nullptr);
  }

  operator VkRenderPass() const
  {
    return m_renderPass;
  }
};


// Descriptor Set + Layout used to deliver the buffer data.
class BufferDescriptors
{
  // Container for buffer descriptors.
  nvvk::DescriptorSetContainer m_descriptors;

public:
  BufferDescriptors(VkDevice device,
                    VkBuffer staticBuffer,
                    VkBuffer dynamicBuffer)
      : m_descriptors(device)
  {
    // Set up descriptor set layout, one descriptor each for camera transforms
    // (UBO) ,instances array, and vertex array (both as shader storage
    // buffers).
    m_descriptors.addBinding(CAMERA_TRANSFORMS_BINDING,
                             VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                             VK_SHADER_STAGE_ALL_GRAPHICS);
    m_descriptors.addBinding(INSTANCES_BINDING,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                             VK_SHADER_STAGE_ALL_GRAPHICS);
    m_descriptors.initLayout();

    // Only need 1 descriptor set (never changes). Initialize it.
    m_descriptors.initPool(1);
    VkDescriptorBufferInfo uboInfo{
        dynamicBuffer,
        offsetof(DynamicBufferData, cameraTransforms),
        sizeof(DynamicBufferData::cameraTransforms)};
    VkDescriptorBufferInfo instanceBufferInfo{
        dynamicBuffer,
        offsetof(DynamicBufferData, instances),
        sizeof(DynamicBufferData::instances)};
    VkDescriptorBufferInfo vertexBufferInfo{
        staticBuffer,
        offsetof(StaticBufferData, vertices),
        sizeof(StaticBufferData::vertices)};
    std::array<VkWriteDescriptorSet, 2> writes = {
        m_descriptors.makeWrite(0, CAMERA_TRANSFORMS_BINDING, &uboInfo),
        m_descriptors.makeWrite(0, INSTANCES_BINDING, &instanceBufferInfo)};
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
  }
  // Destructor not needed -- m_descriptors cleans itself up.

  BufferDescriptors(BufferDescriptors&&) = delete;

  VkDescriptorSet getSet() const
  {
    return m_descriptors.getSet(0);
  }

  VkDescriptorSetLayout getLayout() const
  {
    return m_descriptors.getLayout();
  }
};


// Dynamic viewport/scissor pipeline for drawing the background.
// No depth write or test, and the vertex shader hard-codes
// drawing a full-screen triangle.
class BackgroundPipeline
{
  // Borrowed device pointer.
  VkDevice m_device;

  // We manage these.
  VkPipeline       m_pipeline = VK_NULL_HANDLE;
  VkPipelineLayout m_layout   = VK_NULL_HANDLE;

public:
  BackgroundPipeline(VkDevice                 device,
                     VkRenderPass             renderPass,
                     const BufferDescriptors& bufferDescriptors)
      : m_device(device)
  {
    // Set up pipeline layout.
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount         = 1;
    VkDescriptorSetLayout setLayout           = bufferDescriptors.getLayout();
    pipelineLayoutInfo.pSetLayouts            = &setLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges    = nullptr;
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr,
                                      &m_layout));

    // Hides all the graphics pipeline boilerplate (in particular
    // enabling dynamic viewport and scissor). We just have to
    // disable the depth test and write.
    nvvk::GraphicsPipelineState pipelineState;
    pipelineState.depthStencilState.depthTestEnable  = false;
    pipelineState.depthStencilState.depthWriteEnable = false;

    // Compile shaders and state into graphics pipeline.
    auto vertSpv =
        nvh::loadFile("background.vert.spv", true, searchPaths, true);
    auto fragSpv =
        nvh::loadFile("background.frag.spv", true, searchPaths, true);
    nvvk::GraphicsPipelineGenerator generator(m_device, m_layout, renderPass,
                                              pipelineState);
    generator.addShader(vertSpv, VK_SHADER_STAGE_VERTEX_BIT);
    generator.addShader(fragSpv, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_pipeline = generator.createPipeline();
  }

  BackgroundPipeline(BackgroundPipeline&&) = delete;

  ~BackgroundPipeline()
  {
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_layout, nullptr);
  }

  operator VkPipeline() const
  {
    return m_pipeline;
  }

  VkPipelineLayout getLayout() const
  {
    return m_layout;
  }
};


class ObjectPipeline
{
  // Borrowed device pointer.
  VkDevice   m_device;

  // Managed pipeline and pipeline layout.
  VkPipeline       m_pipeline = VK_NULL_HANDLE;
  VkPipelineLayout m_layout   = VK_NULL_HANDLE;

public:
  ObjectPipeline(VkDevice                 device,
                 VkRenderPass             renderPass,
                 const BufferDescriptors& bufferDescriptors)
      : m_device(device)
  {
    // Setup pipeline layout with given descriptor set layout.
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount         = 1;
    VkDescriptorSetLayout descriptorSetLayout = bufferDescriptors.getLayout();
    pipelineLayoutInfo.pSetLayouts            = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges    = nullptr;
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr,
                                      &m_layout));

    // Hides all the graphics pipeline boilerplate (in particular
    // enabling dynamic viewport and scissor).
    nvvk::GraphicsPipelineState pipelineState;

    // Set up vertex bindings (note that instanced data is instead
    // passed as storage buffer).
    for (VkVertexInputAttributeDescription attr : VertexData::getAttributes())
    {
      pipelineState.addAttributeDescription(attr);
    }
    VkVertexInputBindingDescription bindingDescription{
        0, sizeof(VertexData), VK_VERTEX_INPUT_RATE_VERTEX};
    pipelineState.addBindingDescription(bindingDescription);

    // Compile shader modules plus above state into graphics pipeline.
    auto vertSpv = nvh::loadFile("object.vert.spv", true, searchPaths, true);
    auto fragSpv = nvh::loadFile("object.frag.spv", true, searchPaths, true);
    nvvk::GraphicsPipelineGenerator generator(m_device, m_layout, renderPass,
                                              pipelineState);
    generator.addShader(vertSpv, VK_SHADER_STAGE_VERTEX_BIT);
    generator.addShader(fragSpv, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_pipeline = generator.createPipeline();
  }

  ~ObjectPipeline()
  {
    vkDestroyPipelineLayout(m_device, m_layout, nullptr);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
  }

  operator VkPipeline() const
  {
    return m_pipeline;
  }

  VkPipelineLayout getLayout() const
  {
    return m_layout;
  }
};


// Fill the given transforms, calculated from global camera value and the given
// framebuffer size.
void cameraTransformsFromGlobalCamera(CameraTransforms* out,
                                      float             width,
                                      float             height)
{
  float aspectRatio = width / height;
  float cosTheta    = cos(cameraTheta);
  float sinTheta    = sin(cameraTheta);
  float cosPhi      = cos(cameraPhi);
  float sinPhi      = sin(cameraPhi);

  cameraForward = nvmath::vec3f(sinPhi * cosTheta, cosPhi, sinPhi * sinTheta);
  nvmath::vec3f up(0, 1, 0);
  nvmath::vec3f cameraCenter = cameraPosition + cameraForward;
  nvmath::mat4  view = nvmath::look_at(cameraPosition, cameraCenter, up);
  nvmath::mat4  proj =
      nvmath::perspectiveVK(cameraFOV, aspectRatio, cameraNear, cameraFar);

  out->view         = view;
  out->proj         = proj;
  out->viewInverse  = nvmath::invert(view);  // Use view, not out->view, slow.
  out->projInverse  = nvmath::invert(proj);
  out->colorByDepth = colorByDepth;
}


// Add GLFW callbacks to add mouse controls for the camera.
void addCallbacks(GLFWwindow* pWindow)
{
  static float mouseX, mouseY;
  static bool  lmb_down, mmb_down, rmb_down;

  glfwSetMouseButtonCallback(
    pWindow,
    [] (GLFWwindow* window, int button, int action, int mods)
    {
      ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

      bool leftover;
      bool& mouseFlag =
        button == GLFW_MOUSE_BUTTON_LEFT   ? lmb_down :
        button == GLFW_MOUSE_BUTTON_MIDDLE ? mmb_down :
        button == GLFW_MOUSE_BUTTON_RIGHT  ? rmb_down : leftover;
      mouseFlag = (action != GLFW_RELEASE) && !ImGui::GetIO().WantCaptureMouse;
    });
  glfwSetCursorPosCallback(
    pWindow,
    [] (GLFWwindow* window, double x, double y)
    {
      float dx = float(x - mouseX);
      mouseX   = float(x);
      float dy = float(y - mouseY);
      mouseY   = float(y);

      // Left mouse button rotates camera.
      if (lmb_down)
      {
        cameraTheta += dx * 0.01f;
        cameraPhi   += dy * 0.01f;
        if (cameraPhi > 3.10f) cameraPhi = 3.10f;
        if (cameraPhi < 0.04f) cameraPhi = 0.04f;
      }

      // Middle mouse button alters FOV.
      if (mmb_down)
      {
        cameraFOV += dy * 0.2f;
        if (cameraFOV < 1.000f) cameraFOV = 1.000f;
        if (cameraFOV > 170.0f) cameraFOV = 170.0f;
      }

      // Right mouse button pans around.
      if (rmb_down)
      {
        nvmath::vec3f cameraRightwards = nvmath::cross(
          cameraForward, nvmath::vec3f(0, 1, 0));
        cameraPosition.y += dy * -0.01f;
        cameraPosition   += cameraRightwards * dx * 0.01f;
      }
    });

  // Scroll wheel moves forwards/backwards.
  glfwSetScrollCallback(
    pWindow,
    [] (GLFWwindow* window, double x, double y)
    {
      ImGui_ImplGlfw_ScrollCallback(window, x, y);
      if (!ImGui::GetIO().WantCaptureMouse)
      {
        cameraPosition += cameraForward * float(y * 0.5f);
      }
    });

  // Space toggles animation
  // Numbers set restrictive depth bounds.
  // 'r' resets depth bounds to default.
  // 'd' toggles depth debug coloring.
  // 'i' toggles inheriting viewport/scissors.
  // 'f' force re-record secondary command buffer.
  // 'u' toggles UI visibility.
  // 'v' toggles vsync.
  glfwSetCharCallback(
    pWindow,
    [] (GLFWwindow*, unsigned int codepoint)
    {
      switch (codepoint)
      {
        case 'd':
          colorByDepth = !colorByDepth;
          break;
        case 'f':
          forceRerecordSecondaryCmdBuf = true;
          break;
        case 'i':
          if (has_VK_NV_inherited_viewport_scissor)
          {
            use_VK_NV_inherited_viewport_scissor ^= 1;
          }
          else
          {
            printf("VK_NV_inherited_viewport_scissor not supported\n");
          }
          break;
        case 'p':
          printEvents ^= 1;
          break;
        case 'u':
          guiVisible ^= 1;
          break;
        case 'v':
          vsync ^= 1;
          break;
        case 'w':
          wobbleViewport ^= 1;
          break;
        case ' ':
          paused = !paused;
          break;
        case 'r':
          minDepth = 0.0f;
          maxDepth = 1.0f;
          break;
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
          minDepth = std::max(0.0f, (codepoint - '0') * 0.1f - 0.1f);
          maxDepth = std::min(1.0f, (codepoint - '0') * 0.1f + 0.2f);
          break;
      }
    });
}



// Simple container for ImGui stuff, useful only for my basic needs.
// Unfortunately I couldn't initialize everything in a constructor for
// this class, you have to call cmdInit to initialize the state.
// Note that most UI controls modify the global state that glfw callbacks
// also modify; I'm too lazy to pass around user data pointers everywhere.
class Gui
{
  VkDevice            m_device{};
  VkDescriptorPool    m_pool{};
  ImGuiContext*       m_guiContext{};

  // For fps counter, updated once per second.
  float   m_displayedFPS         = 0;
  float   m_displayedFrameTime   = 0;
  float   m_frameCountThisSecond = 1;
  float   m_frameTimeThisSecond  = 0;
  int64_t m_thisSecond           = 0;
  double  m_lastUpdateTime       = 0;

public:
  // Must be called once after FrameManager initialized, so that the
  // correct queue is chosen. Some initialization is done directly,
  // some by recording commands to the given command buffer.
  void cmdInit(VkCommandBuffer      cmdBuf,
               GLFWwindow*          pWindow,
               const nvvk::Context& ctx,
               const FrameManager&  frameManager,
               VkRenderPass         renderPass,
               uint32_t             subpass)
  {
    m_device = ctx;

    m_guiContext = ImGui::CreateContext(nullptr);
    assert(m_guiContext != nullptr);
    ImGui::SetCurrentContext(m_guiContext);

    ImGuiH::Init(800, 800, nullptr, ImGuiH::FONT_PROPORTIONAL_SCALED);
    ImGuiH::setFonts(ImGuiH::FONT_PROPORTIONAL_SCALED);
    ImGuiH::setStyle(true);

    std::array<VkDescriptorPoolSize, 3> poolSizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1} };
    VkDescriptorPoolCreateInfo poolInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        poolSizes.size(), poolSizes.size(), poolSizes.data() };
    assert(m_pool == VK_NULL_HANDLE);
    NVVK_CHECK(vkCreateDescriptorPool(ctx, &poolInfo, nullptr, &m_pool));

    ImGui_ImplVulkan_InitInfo info {
        ctx.m_instance,
        ctx.m_physicalDevice,
        ctx.m_device,
        frameManager.getQueueFamilyIndex(),
        frameManager.getQueue(),
        VK_NULL_HANDLE,      // No need for pipeline cache for simple sample.
        m_pool,
        subpass,
        frameManager.getSwapChain().getImageCount(),
        frameManager.getSwapChain().getImageCount(),
        VK_SAMPLE_COUNT_1_BIT,
        nullptr,
        [] (VkResult err) { NVVK_CHECK(err); } };

    ImGui_ImplVulkan_Init(&info, renderPass);
    ImGui_ImplVulkan_CreateFontsTexture(cmdBuf);

    ImGui_ImplGlfw_InitForVulkan(pWindow, false);
  }

  Gui() = default;
  Gui(Gui&&) = delete;
  ~Gui()
  {
    vkDestroyDescriptorPool(m_device, m_pool, nullptr);
    ImGui_ImplVulkan_DestroyFontUploadObjects();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
  }

  // Per-frame ImGui code.
  void doFrame()
  {
    updateFpsSample();

    ImGui::NewFrame();
    ImGui_ImplGlfw_NewFrame();

    if (guiVisible)
    {
      ImGui::Begin("Toggle UI [u]");
      ImGui::PushItemWidth(150 * ImGuiH::getDPIScale());

      ImGui::Checkbox("paused [space]", &paused);

      if (has_VK_NV_inherited_viewport_scissor)
      {
        ImGui::Checkbox("Inherit viewport/scissor [i]",
                        &use_VK_NV_inherited_viewport_scissor);
      }
      else
      {
        ImGui::TextUnformatted("VK_NV_inherited_viewport_scissor");
        ImGui::TextUnformatted("not available.");
      }

      ImGui::Separator();

      ImGui::Text("FPS: %.0f", m_displayedFPS);
      ImGui::Text("Frame Time: %7.4f ms", m_displayedFrameTime * 1000.);
      ImGui::Checkbox("vsync [v]", &vsync);
      ImGui::Checkbox("Print Resize Events [p]", &printEvents);

      ImGui::Separator();

      ImGui::SliderFloat("FOV", &cameraFOV, 1.0, 170.0);
      ImGui::SliderFloat("minDepth", &minDepth, 0.0, 1.0);
      ImGui::SliderFloat("maxDepth", &maxDepth, 0.0, 1.0);
      ImGui::Checkbox("Visualize Depth [d]", &colorByDepth);
      ImGui::Checkbox("Wobble Viewport Size [w]", &wobbleViewport);
      ImGui::End();
    }
    ImGui::Render();
  }

private:
  void updateFpsSample()
  {
    double now = glfwGetTime();
    if (m_lastUpdateTime == 0)
    {
      m_lastUpdateTime = now;
      return;
    }

    if (int64_t(now) != m_thisSecond)
    {
      m_displayedFPS       = m_frameCountThisSecond;
      m_displayedFrameTime = m_frameTimeThisSecond;

      m_thisSecond           = int64_t(now);
      m_frameCountThisSecond = 1;
      m_frameTimeThisSecond  = 0;
    }
    else
    {
      float frameTime = float(now - m_lastUpdateTime);
      m_frameCountThisSecond++;
      m_frameTimeThisSecond = std::max(m_frameTimeThisSecond, frameTime);
    }
    m_lastUpdateTime = now;
  }
};



// Class defining main loop of the sample.
class App
{
public:
  static void
  mainLoop(nvvk::Context& context, GLFWwindow* window, VkSurfaceKHR surface)
  {
    // Run until user clicks X button or equivalent
    App app(context, window, surface);
    while(!glfwWindowShouldClose(window))
    {
      app.doFrame();
    }
  }

private:
  // Provided from outside.
  nvvk::Context& m_context;
  GLFWwindow*    m_window;
  VkSurfaceKHR   m_surface;

  // Secondary command buffer (contains actual draw commands), the
  // viewport used when it was recorded, and whether the secondary
  // command buffer enabled inheriting 2D viewport and scissors from
  // the primary command buffer.
  VkCommandBuffer m_secondaryCmdBuf = VK_NULL_HANDLE;
  VkViewport      m_secondaryCmdBufViewport{};
  bool            m_secondaryCmdBufInheritViewport = false;

  // Buffers and info on subsections allocated for each model.
  ScopedBuffers              m_buffers;
  std::vector<ModelDrawInfo> m_modelDraws;

  // Command buffer and submit info for uploading DynamicBufferData
  // from the staging buffer to the device.
  VkCommandBuffer m_dynamicBufferUploadCmdBuffer{};
  VkSubmitInfo    m_dynamicBufferUploadSubmitInfo{};

  // Depth and color formats to use.
  VkFormat m_depthFormat, m_colorFormat;

  // Render pass, framebuffers + size, and descriptor pool/layout.
  ScopedRenderPass  m_renderPass;
  Framebuffers      m_framebuffers;
  BufferDescriptors m_bufferDescriptors;

  // Graphics pipelines, and current viewport to use.
  BackgroundPipeline m_backgroundPipeline;
  ObjectPipeline     m_objectPipeline;
  VkViewport         m_viewport{};

  // GUI stuff.
  Gui m_gui;

  // Updated each frame.
  double m_prevFrameTime;

  // Used for ticking the animation.
  uint64_t     m_tickNumber             = 0;
  bool         m_warnedTooManyInstances = false;
  std::mt19937 m_rng; // I don't seed it so the animation's the same each time.

  // Swapchain management object. Declare LAST so that its destructor
  // (which calls vkQueueWaitIdle) runs before the above objects are destroyed.
  FrameManager m_frameManager;

  // Initialize everything (above object order is very important!)
  App(nvvk::Context& ctx, GLFWwindow* window, VkSurfaceKHR surface)
      : m_context(ctx)
      , m_window(window)
      , m_surface(surface)
      , m_buffers(ctx)
      , m_depthFormat(nvvk::findDepthFormat(ctx.m_physicalDevice))
      , m_colorFormat(VK_FORMAT_B8G8R8A8_SRGB)
      , m_renderPass(ctx, m_colorFormat, m_depthFormat)
      , m_framebuffers(ctx.m_physicalDevice, ctx, m_renderPass, m_depthFormat)
      , m_bufferDescriptors(ctx,
                            m_buffers.getStaticBuffer(),
                            m_buffers.getDynamicBuffer())
      , m_backgroundPipeline(ctx, m_renderPass, m_bufferDescriptors)
      , m_objectPipeline(ctx, m_renderPass, m_bufferDescriptors)
      , m_prevFrameTime(glfwGetTime())
      , m_frameManager(ctx, surface, 1, 1, true, m_colorFormat,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
  {
    VkCommandBuffer cmdBuf = m_frameManager.recordOneTimeCommandBuffer();

    // Create models.
    std::vector<Model> models = makeModels();

    // Initialize static buffer with models, and fill m_modelDraws
    // (basically allocate buffer subsections for each model).
    fillStaticStagingInitDraws(models);
    m_buffers.cmdTransferStaticStaging(cmdBuf);

    // Initialize Dear ImGui stuff.
    m_gui.cmdInit(cmdBuf, window, ctx, m_frameManager, m_renderPass, 0);

    // End and submit command buffer.
    NVVK_CHECK(vkEndCommandBuffer(cmdBuf));
    VkSubmitInfo submitInfo {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0, nullptr, nullptr,
        1, &cmdBuf,
        0, nullptr };
    NVVK_CHECK(vkQueueSubmit(
        m_frameManager.getQueue(), 1, &submitInfo, VK_NULL_HANDLE));

    // Record the DynamicBufferData staging->device copy command.
    m_dynamicBufferUploadCmdBuffer = m_frameManager.recordPrimaryCommandBuffer();
    m_buffers.cmdTransferDynamicStaging(
        m_dynamicBufferUploadCmdBuffer,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_INDIRECT_COMMAND_READ_BIT
        | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
        | VK_ACCESS_UNIFORM_READ_BIT
        | VK_ACCESS_SHADER_READ_BIT);
    NVVK_CHECK(vkEndCommandBuffer(m_dynamicBufferUploadCmdBuffer));
    m_dynamicBufferUploadSubmitInfo = {
        VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
        1, &m_dynamicBufferUploadCmdBuffer, 0, nullptr };

    // Wait for initialization tasks to finish and clean up
    // StaticBufferData upload command buffer.
    vkQueueWaitIdle(m_frameManager.getQueue());
    vkFreeCommandBuffers(ctx, m_frameManager.getCommandPool(), 1, &cmdBuf);
  }

  // Body of main loop.
  void doFrame()
  {
    // Tick animation at fps-independent rate.
    double newTime = glfwGetTime();
    double dt      = newTime - m_prevFrameTime;
    double steps   = floor(newTime * ticksPerSecond)
                   - floor(m_prevFrameTime * ticksPerSecond);
    int ticks      = std::min(int(steps), maxTicksPerFrame);
    if (!paused)
    {
      for (int i = 0; i < ticks; ++i) tickAnimation();
    }
    m_prevFrameTime = newTime;

    // Get events and window size from GLFW (modifies camera).
    uint32_t width, height;
    glfwPollEvents();
    waitNonzeroFramebufferSize(m_window, &width, &height);

    // Write new camera and instance data to staging buffer, then
    // copy to device. Note that a VkFence is used to synchronize host
    // and device access to staging buffer (see ScopedBuffers).
    VkFence dynamicStagingFence = m_buffers.getFence();
    vkWaitForFences(m_context, 1, &dynamicStagingFence, VK_TRUE, UINT64_MAX);
    DynamicBufferData* mapped = m_buffers.getDynamicStagingPtr();
    uncheckedUpdateDynamicStaging(mapped);

    // Start the upload of the data from the DynamicBufferData staging buffer.
    // Need the fence to ensure the transfer finishes before updating the
    // staging buffer in the next frame.
    vkResetFences(m_context, 1, &dynamicStagingFence);
    vkQueueSubmit(m_frameManager.getQueue(),
                  1, &m_dynamicBufferUploadSubmitInfo, m_buffers.getFence());

    // Begin the frame, starting primary command buffer recording.
    // beginFrame converts the intended width/height to actual
    // swap chain width/height, which could differ from requested.
    // Update (local copy of) viewport with this width/height.
    VkCommandBuffer primaryCmdBuf;
    nvvk::SwapChainAcquireState acquired;
    m_frameManager.wantVsync(vsync);
    m_frameManager.beginFrame(&primaryCmdBuf, &acquired, &width, &height);
    m_viewport.width  = float(width) * getViewportHorizontalScale();
    m_viewport.height = float(height);

    // Do ImGui stuff.
    m_gui.doFrame();

    // Udate viewport depth bounds from global variables (ImGui and
    // glfw key callbacks modifiy them).
    m_viewport.minDepth = minDepth;
    m_viewport.maxDepth = maxDepth;

    // Select framebuffer for this frame.
    m_framebuffers.recreateNowIfNeeded(m_frameManager.getSwapChain());
    VkFramebuffer framebuffer = m_framebuffers[acquired.index];

    // Replace the secondary command buffer if it is no longer
    // suitable for use.
    updateSecondaryCmdBuf();

    // Record and submit primary command buffer, and present frame.
    // The primary command buffer is magically cleaned up.
    cmdDrawFrame(primaryCmdBuf, framebuffer);
    m_frameManager.endFrame(primaryCmdBuf);
  }

  float getViewportHorizontalScale()
  {
    return !wobbleViewport ? 1.0f :
                             0.875f + 0.125f * float(sin(glfwGetTime() * 4.0));
  }

  // Check if the secondary command buffer needs to be re-recorded.
  bool needNewSecondaryCmdBuf()
  {
    // Debug backdoor.
    if (forceRerecordSecondaryCmdBuf)
    {
      forceRerecordSecondaryCmdBuf = false;
      return true;
    }
    // Changed inheritance setting.
    if (use_VK_NV_inherited_viewport_scissor != m_secondaryCmdBufInheritViewport)
    {
      return true;
    }

    // Missing?
    if (m_secondaryCmdBuf == VK_NULL_HANDLE) return true;

    // Depth bounds changed?
    if (m_viewport.minDepth != m_secondaryCmdBufViewport.minDepth) return true;
    if (m_viewport.maxDepth != m_secondaryCmdBufViewport.maxDepth) return true;

    // Viewport 2D params changed, and not inheriting this state?
    if (!m_secondaryCmdBufInheritViewport)
    {
      if (m_viewport.x != m_secondaryCmdBufViewport.x) return true;
      if (m_viewport.y != m_secondaryCmdBufViewport.y) return true;
      if (m_viewport.width != m_secondaryCmdBufViewport.width) return true;
      if (m_viewport.height != m_secondaryCmdBufViewport.height) return true;
    }

    // It's okay at this point.
    return false;
  }

  // If the secondary command buffer isn't suitable for use anymore,
  // schedule it for deletion at the end of this frame, and record
  // a new one. Must be called within a beginFrame/endFrame pair.
  void updateSecondaryCmdBuf()
  {
    if (!needNewSecondaryCmdBuf()) return;

    // Get a new cmd buffer and throw away the old one.
    m_frameManager.freeFrameCommandBuffer(m_secondaryCmdBuf);
    m_secondaryCmdBuf = m_frameManager.allocateSecondaryCommandBuffer();

    // Begin subpass.
    // The extension struct needed to enable inheriting 2D viewport+scisors.
    VkCommandBufferInheritanceViewportScissorInfoNV inheritViewportInfo {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_VIEWPORT_SCISSOR_INFO_NV,
        nullptr,
        VK_TRUE,
        1, &m_viewport };

    // Typical inheritance info, add the extra extension if the user requests
    // it.
    uint32_t subpass = 0;
    VkCommandBufferInheritanceInfo inheritance {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
        use_VK_NV_inherited_viewport_scissor ? &inheritViewportInfo : nullptr,
        m_renderPass, subpass };

    auto flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT
               | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    VkCommandBufferBeginInfo beginInfo {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VkCommandBufferUsageFlags(flags),
        &inheritance };
    NVVK_CHECK(vkBeginCommandBuffer(m_secondaryCmdBuf, &beginInfo));

    // Set viewport and scissors (if not inherited).
    auto width  = uint32_t(m_viewport.width);
    auto height = uint32_t(m_viewport.height);
    if (!use_VK_NV_inherited_viewport_scissor)
    {
      vkCmdSetViewport(m_secondaryCmdBuf, 0, 1, &m_viewport);
      VkRect2D scissor { { 0, 0 }, { width, height } };
      vkCmdSetScissor(m_secondaryCmdBuf, 0, 1, &scissor);
    }

    VkDescriptorSet descriptorSet = m_bufferDescriptors.getSet();

    // Draw background as full-screen triangle.
    vkCmdBindPipeline(m_secondaryCmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_backgroundPipeline);
    vkCmdBindDescriptorSets(
        m_secondaryCmdBuf,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_backgroundPipeline.getLayout(),
        0, 1, &descriptorSet, 0, nullptr);
    vkCmdDraw(m_secondaryCmdBuf, 3, 1, 0, 0);

    // Draw objects. Bind vertex and index buffer.  Draw indirect
    // allows us to vary the number of drawn objects without
    // re-recording this secondary command buffer.
    VkBuffer     vertexBuffer       = m_buffers.getStaticBuffer();
    VkDeviceSize vertexBufferOffset = offsetof(StaticBufferData, vertices);
    vkCmdBindVertexBuffers(
        m_secondaryCmdBuf,
        0, 1,
        &vertexBuffer, &vertexBufferOffset);
    vkCmdBindIndexBuffer(
        m_secondaryCmdBuf,
        m_buffers.getStaticBuffer(),
        offsetof(StaticBufferData, indices),
        StaticBufferData::indexType);
    vkCmdBindPipeline(
        m_secondaryCmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_objectPipeline);
    vkCmdBindDescriptorSets(
        m_secondaryCmdBuf,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_objectPipeline.getLayout(),
        0, 1, &descriptorSet, 0, nullptr);
    vkCmdDrawIndexedIndirect(
        m_secondaryCmdBuf,
        m_buffers.getDynamicBuffer(),
        offsetof(DynamicBufferData, indirectCmds),
        MAX_MODELS,
        sizeof(VkDrawIndexedIndirectCommand));

    // End command buffer and record parameters used.
    NVVK_CHECK(vkEndCommandBuffer(m_secondaryCmdBuf));
    m_secondaryCmdBufViewport = m_viewport;
    m_secondaryCmdBufInheritViewport = use_VK_NV_inherited_viewport_scissor;
    if (printEvents)
    {
      printf(
          "\x1b[36mRecorded secondary command buffer:\x1b[0m\n"
          "  width=%u height=%u minDepth=%.2f maxDepth=%.2f inherit=%s\n",
          width, height, minDepth, maxDepth,
          m_secondaryCmdBufInheritViewport ? "\x1b[32mtrue\x1b[0m" :
                                             "\x1b[31mfalse\x1b[0m");
    }
  }

  // Write new values to dynamic staging buffer. No synchronization done.
  void uncheckedUpdateDynamicStaging(DynamicBufferData* mapped)
  {
    // Convert models into draw indexed indirect commands.
    for (size_t i = 0; i < m_modelDraws.size(); ++i)
    {
      mapped->indirectCmds[i].indexCount    = m_modelDraws[i].indexCount;
      mapped->indirectCmds[i].instanceCount =
          uint32_t(m_modelDraws[i].instances.size());
      mapped->indirectCmds[i].firstIndex    = m_modelDraws[i].firstIndex;
      mapped->indirectCmds[i].vertexOffset  = m_modelDraws[i].vertexOffset;
      mapped->indirectCmds[i].firstInstance = m_modelDraws[i].firstInstance;
    }
    // Fill excess commands with 0s to avoid trouble.
    for (size_t i = m_modelDraws.size(); i < MAX_MODELS; ++i)
    {
      mapped->indirectCmds[i].indexCount    = 0;
      mapped->indirectCmds[i].instanceCount = 0;
      mapped->indirectCmds[i].firstIndex    = 0;
      mapped->indirectCmds[i].vertexOffset  = 0;
      mapped->indirectCmds[i].firstInstance = 0;
    }

    // Update camera transformations.
    cameraTransformsFromGlobalCamera(
        &mapped->cameraTransforms, m_viewport.width, m_viewport.height);

    // Upload instances.
    for (size_t m = 0; m < m_modelDraws.size(); ++m)
    {
      const auto& instances  = m_modelDraws[m].instances;
      uint32_t instanceCount = uint32_t(instances.size());
      uint32_t firstInstance = m_modelDraws[m].firstInstance;
      assert(instanceCount <= m_modelDraws[m].maxInstances);

      for (uint32_t i = 0; i < instanceCount; ++i)
      {
        mapped->instances[i + firstInstance] = instances[i];
      }
    }
  }

  // Record commands to primary command buffer. Just starts render
  // pass, set viewport/scissors, and call secondary command buffer.
  void cmdDrawFrame(VkCommandBuffer cmdBuf, VkFramebuffer framebuffer)
  {
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color.float32[0] = 1.0f;
    clearValues[0].color.float32[1] = 1.0f;
    clearValues[0].color.float32[2] = 1.0f;
    clearValues[0].color.float32[3] = 1.0f;
    clearValues[1].depthStencil     = {1.0f, 0};

    auto width  = uint32_t(m_viewport.width);
    auto height = uint32_t(m_viewport.height);
    VkRect2D scissor { {0, 0}, {width, height} };

    VkRenderPassBeginInfo beginInfo {
        VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        nullptr,
        m_renderPass,
        framebuffer,
        { { 0, 0 }, m_framebuffers.getExtent() },
        uint32_t(clearValues.size()),
        clearValues.data() };

    // If the secondary command buffer doesn't inherit the viewport/scissor,
    // then it sets it itself and the primary cmd buffer need not set that
    // state.
    if (m_secondaryCmdBufInheritViewport)
    {
      vkCmdSetViewport(cmdBuf, 0, 1, &m_viewport);
      vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
    }
    vkCmdBeginRenderPass(cmdBuf, &beginInfo,
                         VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    vkCmdExecuteCommands(cmdBuf, 1, &m_secondaryCmdBuf);

    // Need to give Dear ImGui a chance to draw stuff too.  I don't
    // try to reuse these commands, but it still needs to be in a
    // temporary secondary command buffer to match the above.
    // Somewhat distracting source of complexity...
    VkCommandBufferInheritanceInfo inheritance {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, nullptr,
        m_renderPass, 0 };
    VkCommandBuffer guiCmdBuf = m_frameManager.recordSecondaryCommandBuffer(
        VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, &inheritance);
    m_frameManager.freeFrameCommandBuffer(guiCmdBuf);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), guiCmdBuf);
    vkEndCommandBuffer(guiCmdBuf);
    vkCmdExecuteCommands(cmdBuf, 1, &guiCmdBuf);

    vkCmdEndRenderPass(cmdBuf);
  }

  void fillStaticStagingInitDraws(const std::vector<Model>& models)
  {
    std::vector<ModelDrawInfo> modelDraws;

    assert(models.size() < MAX_MODELS);

    // For now, equally allocate sections of the instances array for each
    // model.
    uint32_t maxInstancesPerModel = MAX_INSTANCES / MAX_MODELS;

    uint32_t firstIndex  = 0;
    int32_t vertexOffset = 0;

    // First, figure out how we're going to allocate buffer memory for each
    // model.
    for (size_t m = 0; m < models.size(); ++m)
    {
      ModelDrawInfo drawInfo;
      drawInfo.firstIndex    = firstIndex;
      drawInfo.indexCount    = uint32_t(models[m].indices.size());
      drawInfo.vertexOffset  = vertexOffset;
      drawInfo.firstInstance = uint32_t(m * maxInstancesPerModel);
      drawInfo.maxInstances  = maxInstancesPerModel;

      firstIndex += drawInfo.indexCount;
      vertexOffset += int32_t(models[m].vertices.size());

      modelDraws.push_back(drawInfo);
    }

    assert(firstIndex <= MAX_INDICES);
    assert(vertexOffset <= MAX_VERTICES);

    // Then fill in vertex and index staging buffers as allocated.
    StaticBufferData* ptr = m_buffers.getStaticStagingPtr();
    for (size_t m = 0; m < models.size(); ++m)
    {
      uint32_t indexCount  = uint32_t(models[m].indices.size());
      uint32_t vertexCount = uint32_t(models[m].vertices.size());
      size_t indexSize     = sizeof(decltype(models[m].indices[0]));
      size_t vertexSize    = sizeof(decltype(models[m].vertices[0]));

      // Copy indices
      memcpy(
          &ptr->indices[modelDraws[m].firstIndex],
          models[m].indices.data(),
          indexSize * indexCount);

      // Copy vertices
      memcpy(
          &ptr->vertices[modelDraws[m].vertexOffset],
          models[m].vertices.data(),
          vertexSize * vertexCount);
    }

    m_modelDraws = std::move(modelDraws);
  }

  // Perform one discrete step of the gravity simulation (possibly spawning new
  // objects).
  void tickAnimation()
  {
    // Add a new instance, if scheduled.
    if (m_tickNumber++ % ticksPerNewInstance == 0 && !m_modelDraws.empty())
    {
      AnimatedInstanceData instance{};
      instance.position  = spawnOrigin;
      {
        float theta       = m_rng() * 1.4629180792671596e-09f;
        float phi         = m_rng() * 3.657295198167899e-10f;
        float cosPhi      = cosf(phi);
        instance.velocity = spawnSpeed * nvmath::vec3f(
          sinf(theta) * cosPhi, sinf(phi), cosf(theta) * cosPhi);
      }
      {
        float theta           = m_rng() * 1.4629180792671596e-09f;
        float phi             = m_rng() * 3.657295198167899e-10f;
        float cosPhi          = cosf(phi);
        instance.rotationAxis = nvmath::vec3f(
          sinf(theta) * cosPhi, sinf(phi), cosf(theta) * cosPhi);
      }
      instance.turnsPerSecond = m_rng() * (maxTurnsPerSecond / 4294967296.0f);
      instance.rotationAngle  = m_rng() * (-3.141592f / 4294967296.0f);

      size_t modelNumber = m_rng() % m_modelDraws.size();
      auto&  modelData   = m_modelDraws[modelNumber];
      if (modelData.instances.size() >= modelData.maxInstances)
      {
        if (!m_warnedTooManyInstances)
        {
          fprintf(stderr, "tickAnimation: reached maxInstances limit\n");
          m_warnedTooManyInstances = true;
        }
      }
      else
      {
        modelData.instances.push_back(instance);
      }
    }

    // Update existing instances.
    for (auto& modelData : m_modelDraws)
    {
      auto& instances = modelData.instances;
      for (size_t i = 0; i < instances.size(); ++i)
      {
        AnimatedInstanceData& instance = instances[i];

        // Update velocity.
        instance.velocity.y -= gravityAcceleration * secondsPerTick;

        // Update position.
        instance.position +=
          nvmath::vec4(instance.velocity * secondsPerTick, 1.0f);

        // Update rotation.
        instance.rotationAngle += instance.turnsPerSecond * secondsPerTick
                                * 6.283185307179586f;

        // If below height limit, remove this instance. Replace with
        // instance at the back, to avoid linear time removal.
        if (instance.position.y < deathHeight)
        {
          instances[i] = instances.back();
          instances.pop_back();
          continue;
        }

        // Compute new model matrix.
        nvmath::mat4 M;
        M.identity();
        M.translate(instance.position);
        M.rotate(instance.rotationAngle, instance.rotationAxis);
        instance.modelMatrix = M;
      }
    }
  }
};



int main()
{
  // Create Vulkan glfw window.
  glfwInit();
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  const char* pTitle = "Inherited 2D Viewport Scissor";
  GLFWwindow* pWindow = glfwCreateWindow(800, 800, pTitle, nullptr, nullptr);
  assert(pWindow != nullptr);
  addCallbacks(pWindow);

  // Init Vulkan 1.1 device with swap chain extensions glfw needs.
  nvvk::ContextCreateInfo deviceInfo;
  deviceInfo.apiMajor = 1;
  deviceInfo.apiMinor = 1;
  uint32_t count;
  const char** extensions = glfwGetRequiredInstanceExtensions(&count);
  assert(extensions != nullptr);
  for (uint32_t i = 0; i < count; ++i)
  {
    deviceInfo.addInstanceExtension(extensions[i]);
  }
  deviceInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  // Enable VK_NV_inherited_viewport_scissor if supported.
  VkPhysicalDeviceInheritedViewportScissorFeaturesNV
    inheritedViewportScissorFeatures {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INHERITED_VIEWPORT_SCISSOR_FEATURES_NV,
      nullptr };
  deviceInfo.addDeviceExtension(
    VK_NV_INHERITED_VIEWPORT_SCISSOR_EXTENSION_NAME,
    true,
    &inheritedViewportScissorFeatures);

  nvvk::Context ctx;
  ctx.init(deviceInfo);

  // Ignore validation error caused by ImGui Vulkan backend's
  // use of vkFlushMappedMemoryRanges; TODO investigate.
  ctx.ignoreDebugMessage(249857837);

  // Check for VK_NV_inherited_viewport_scissor
  if (ctx.hasDeviceExtension(VK_NV_INHERITED_VIEWPORT_SCISSOR_EXTENSION_NAME))
  {
    if (inheritedViewportScissorFeatures.inheritedViewportScissor2D)
    {
      has_VK_NV_inherited_viewport_scissor = true;
      use_VK_NV_inherited_viewport_scissor = true;
    }
    else
    {
      printf("\x1b[35mNOTE:\x1b[0m "
        "VK_NV_inherited_viewport_scissor not supported :/\n");
    }
  }

  // Get the surface to draw to.
  VkSurfaceKHR surface;
  NVVK_CHECK(glfwCreateWindowSurface(
      ctx.m_instance, pWindow, nullptr, &surface));

  // Run the program.
  App::mainLoop(ctx, pWindow, surface);

  // At this point, FrameManager's destructor in mainLoop ensures all
  // pending commands are complete. So, we can clean up the surface,
  // Vulkan device, and glfw.
  vkDestroySurfaceKHR(ctx.m_instance, surface, nullptr);
  ctx.deinit(); // Note: nvvk::Context does not have a useful destructor.
  glfwTerminate();
}


// Make a bunch of models, hidden at the bottom of the file as it's mostly
// unrelated to the actual graphics techniques demonstrated.
static std::vector<Model> makeModels()
{
  std::vector<Model> models;
  Model              model;

  // *** SPHERE ***
  // Create a spherical model by recursively tesselating an octahedron.
  struct VertexIndex
  {
    nvmath::vec3f vertex;
    uint32_t      index;  // Keep track of this vert's _eventual_ index in vertices.
  };
  struct Triangle
  {
    VertexIndex vert0, vert1, vert2;
  };

  VertexIndex posX{{ 1, 0, 0}, 0};
  VertexIndex negX{{-1, 0, 0}, 1};
  VertexIndex posY{{0,  1, 0}, 2};
  VertexIndex negY{{0, -1, 0}, 3};
  VertexIndex posZ{{0, 0,  1}, 4};
  VertexIndex negZ{{0, 0, -1}, 5};
  uint32_t    vertexCount = 6;

  // Initial triangle list is octahedron.
  std::vector<Triangle> triangles{
    {posX, posY, posZ}, {posX, negZ, posY}, {posX, posZ, negY},
    {posX, negY, negZ}, {negX, posZ, posY}, {negX, posY, negZ},
    {negX, negY, posZ}, {negX, negZ, negY}};

  // Recursion: every iteration, convert the current model to a new
  // model by breaking each triangle into 4 triangles.
  for(int recursions = 0; recursions < 4; ++recursions)
  {
    std::vector<Triangle> new_triangles;
    for(Triangle t : triangles)
    {
      // Split each of three edges in half, then fixup the
      // length of the midpoint to match m_lanternModelRadius.
      // Record the index the new vertices will eventually have in vertices.
      VertexIndex v01{
        nvmath::normalize(t.vert0.vertex + t.vert1.vertex),
        vertexCount++};
      VertexIndex v12{
        nvmath::normalize(t.vert1.vertex + t.vert2.vertex),
        vertexCount++};
      VertexIndex v02{
        nvmath::normalize(t.vert0.vertex + t.vert2.vertex),
        vertexCount++};

      // Old triangle becomes 4 new triangles.
      new_triangles.push_back({t.vert0, v01, v02});
      new_triangles.push_back({t.vert1, v12, v01});
      new_triangles.push_back({t.vert2, v02, v12});
      new_triangles.push_back({v01, v12, v02});
    }
    triangles = std::move(new_triangles);
  }

  model.vertices.resize(vertexCount);
  model.indices.reserve(triangles.size() * 3);

  // Write out the vertices to the vertices vector, and
  // connect the tesselated triangles with indices in the indices vector.
  for(Triangle t : triangles)
  {
    auto getVertex = [] (VertexIndex vi) -> VertexData {
      VertexData result;
      result.position = nvmath::vec4(vi.vertex, 1.0f);
      result.normal   = nvmath::vec4(vi.vertex, 1.0f);
      float foo       = vi.vertex.y * 0.7f - 0.35f;
      float bar       = vi.vertex.y == 0 ? 0.2f : 0.0f;
      result.color    = nvmath::vec4(bar, 0.0, 0.0, 1.0 - foo);
      return result;
    };
    model.vertices[t.vert0.index] = getVertex(t.vert0);
    model.vertices[t.vert1.index] = getVertex(t.vert1);
    model.vertices[t.vert2.index] = getVertex(t.vert2);
    model.indices.push_back(t.vert0.index);
    model.indices.push_back(t.vert1.index);
    model.indices.push_back(t.vert2.index);
  }
  models.push_back(std::move(model));
  model.vertices.clear();
  model.indices.clear();

  // *** CUBE ***
  nvmath::vec4 normalNegX(-1, 0, 0, 0);
  nvmath::vec4  colorNegX(0, 0.2, 0.2, 0);
  nvmath::vec4 normalNegY(0, -1, 0, 0);
  nvmath::vec4  colorNegY(0.2, 0, 0.2, 0);
  nvmath::vec4 normalNegZ(0, 0, -1, 0);
  nvmath::vec4  colorNegZ(0.2, 0.2, 0, 0);
  nvmath::vec4 normalPosX(1, 0, 0, 0);
  nvmath::vec4  colorPosX(0.3, 0, 0, 0);
  nvmath::vec4 normalPosY(0, 1, 0, 0);
  nvmath::vec4  colorPosY(0, 0.3, 0, 0);
  nvmath::vec4 normalPosZ(0, 0, 1, 0);
  nvmath::vec4  colorPosZ(0, 0, 0.3, 0);
  model.vertices = {
    { { -.5, +.5, +.5, 1.0 }, normalNegX, colorNegX },
    { { -.5, +.5, -.5, 1.0 }, normalNegX, colorNegX },
    { { -.5, -.5, +.5, 1.0 }, normalNegX, colorNegX },
    { { -.5, +.5, -.5, 1.0 }, normalNegX, colorNegX },
    { { -.5, -.5, -.5, 1.0 }, normalNegX, colorNegX },
    { { -.5, -.5, +.5, 1.0 }, normalNegX, colorNegX },

    { { +.5, -.5, -.5, 1.0 }, normalPosX, colorPosX },
    { { +.5, +.5, -.5, 1.0 }, normalPosX, colorPosX },
    { { +.5, -.5, +.5, 1.0 }, normalPosX, colorPosX },
    { { +.5, +.5, -.5, 1.0 }, normalPosX, colorPosX },
    { { +.5, +.5, +.5, 1.0 }, normalPosX, colorPosX },
    { { +.5, -.5, +.5, 1.0 }, normalPosX, colorPosX },

    { { -.5, -.5, -.5, 1.0 }, normalNegY, colorNegY },
    { { +.5, -.5, +.5, 1.0 }, normalNegY, colorNegY },
    { { -.5, -.5, +.5, 1.0 }, normalNegY, colorNegY },
    { { -.5, -.5, -.5, 1.0 }, normalNegY, colorNegY },
    { { +.5, -.5, -.5, 1.0 }, normalNegY, colorNegY },
    { { +.5, -.5, +.5, 1.0 }, normalNegY, colorNegY },

    { { +.5, +.5, +.5, 1.0 }, normalPosY, colorPosY },
    { { +.5, +.5, -.5, 1.0 }, normalPosY, colorPosY },
    { { -.5, +.5, -.5, 1.0 }, normalPosY, colorPosY },
    { { -.5, +.5, +.5, 1.0 }, normalPosY, colorPosY },
    { { +.5, +.5, +.5, 1.0 }, normalPosY, colorPosY },
    { { -.5, +.5, -.5, 1.0 }, normalPosY, colorPosY },

    { { -.5, -.5, -.5, 1.0 }, normalNegZ, colorNegZ },
    { { -.5, +.5, -.5, 1.0 }, normalNegZ, colorNegZ },
    { { +.5, -.5, -.5, 1.0 }, normalNegZ, colorNegZ },
    { { +.5, +.5, -.5, 1.0 }, normalNegZ, colorNegZ },
    { { +.5, -.5, -.5, 1.0 }, normalNegZ, colorNegZ },
    { { -.5, +.5, -.5, 1.0 }, normalNegZ, colorNegZ },

    { { -.5, +.5, +.5, 1.0 }, normalPosZ, colorPosZ },
    { { +.5, -.5, +.5, 1.0 }, normalPosZ, colorPosZ },
    { { +.5, +.5, +.5, 1.0 }, normalPosZ, colorPosZ },
    { { +.5, -.5, +.5, 1.0 }, normalPosZ, colorPosZ },
    { { -.5, +.5, +.5, 1.0 }, normalPosZ, colorPosZ },
    { { -.5, -.5, +.5, 1.0 }, normalPosZ, colorPosZ }};

  model.indices = {
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35 };

  models.push_back(std::move(model));
  model.vertices.clear();
  model.indices.clear();

  // Map subdivided unit grid [0, 1] x [0, 1] to mesh.
  auto mapGrid = [] (
    uint32_t gridX, uint32_t gridY, // Defines # of verts, not quads.
    bool     stitchX, bool   stitchY,
    auto xyToVertex, // (x', y') -> VertexData
    auto preprocessX, // x -> x'
    auto preprocessY) // y -> y'
  {
    float xScale = 1.0f / (stitchX ? gridX : gridX - 1);
    float yScale = 1.0f / (stitchY ? gridY : gridY - 1);

    auto gridToIndex = [gridY] (uint32_t x, uint32_t y) {
      return uint32_t(gridY * x + y);
    };

    Model model;
    model.vertices.resize(gridX * gridY);
    model.indices.reserve(6 * gridX * gridY); // overestimate.

    for (uint32_t x = 0; x < gridX-1; ++x)
    {
      for (uint32_t y = 0; y < gridY-1; ++y)
      {
        model.indices.push_back(gridToIndex(x, y));
        model.indices.push_back(gridToIndex(x+1, y));
        model.indices.push_back(gridToIndex(x, y+1));
        model.indices.push_back(gridToIndex(x+1, y));
        model.indices.push_back(gridToIndex(x+1, y+1));
        model.indices.push_back(gridToIndex(x, y+1));
      }
    }

    if (stitchX)
    {
      for (uint32_t y = 0; y < gridY-1; ++y)
      {
        model.indices.push_back(gridToIndex(gridX-1, y));
        model.indices.push_back(gridToIndex(0, y));
        model.indices.push_back(gridToIndex(gridX-1, y+1));
        model.indices.push_back(gridToIndex(0, y));
        model.indices.push_back(gridToIndex(0, y+1));
        model.indices.push_back(gridToIndex(gridX-1, y+1));
      }
    }

    if (stitchY)
    {
      for (uint32_t x = 0; x < gridX-1; ++x)
      {
        model.indices.push_back(gridToIndex(x, gridY-1));
        model.indices.push_back(gridToIndex(x+1, gridY-1));
        model.indices.push_back(gridToIndex(x, 0));
        model.indices.push_back(gridToIndex(x+1, gridY-1));
        model.indices.push_back(gridToIndex(x+1, 0));
        model.indices.push_back(gridToIndex(x, 0));
      }
    }

    if (stitchX & stitchY)
    {
      model.indices.push_back(gridToIndex(gridX-1, gridY-1));
      model.indices.push_back(gridToIndex(0, gridY-1));
      model.indices.push_back(gridToIndex(gridX-1, 0));
      model.indices.push_back(gridToIndex(0, gridY-1));
      model.indices.push_back(gridToIndex(0, 0));
      model.indices.push_back(gridToIndex(gridX-1, 0));
    }

    std::vector<decltype(preprocessY(0.0f))> ys(gridY);
    for (uint32_t y = 0; y < gridY; ++y)
    {
      ys[y] = preprocessY(y * yScale);
    }

    for (uint32_t x = 0; x < gridX; ++x)
    {
      auto xs = preprocessX(x * xScale);
      for (uint32_t y = 0; y < gridY; ++y)
      {
        auto idx = gridToIndex(x, y);
        model.vertices[idx] = xyToVertex(xs, ys[y]);
      }
    }

    return model;
  };

  auto identity = [] (float f) { return f; };

  struct SinCos { float s; float c; };
  auto sinCos = [] (float rads) { return SinCos { sinf(rads), cosf(rads) }; };
  auto sinCosTurns = [sinCos] (float t) { return sinCos(t*6.283185307179586f); };

  // *** TORUS (doughnut) ***
  auto mapToTorus = [] (SinCos bigAngle, SinCos smallAngle) {
    float bigRadius = 0.75f, smallRadius = 0.45f;
    auto radialUnitVector = nvmath::vec3f(bigAngle.s, 0, bigAngle.c);
    auto ringUnitVector   = radialUnitVector * smallAngle.c
                          + nvmath::vec3f(0, smallAngle.s, 0);
    VertexData result;
    float icing = std::max(0.0f, 4.0f * (smallAngle.s - 0.75f));
    result.position = nvmath::vec4f(
      bigRadius * radialUnitVector + smallRadius * ringUnitVector, 1.0f);
    result.normal = ringUnitVector;
    result.color  = nvmath::vec4(
      icing*0.6, icing*0.3, icing*0.5, 0.6 * (1-icing));
    return result;
  };
  models.push_back(
    mapGrid(64, 32, true, true, mapToTorus, sinCosTurns, sinCosTurns));

  // *** COIL ***
  auto mapToCoil = [sinCosTurns] (float t, SinCos smallAngle) {
    float bigRadius = 0.5f, smallRadius = 0.15f, turns = 5.0f, height = 2.0f;
    SinCos bigAngle = sinCosTurns(t * turns);

    auto radialUnitVector = nvmath::vec3f(bigAngle.s, 0, bigAngle.c);
    auto ringUnitVector   = radialUnitVector * smallAngle.c
                          + nvmath::vec3f(0, smallAngle.s, 0);
    float  y              = height * (t - 0.5f);
    VertexData result;
    result.position = nvmath::vec4f(
      bigRadius * radialUnitVector
    + smallRadius * ringUnitVector
    + nvmath::vec3f(0, y, 0), 1.0f);
    result.normal = ringUnitVector; // Close enough.
    result.color  = nvmath::vec4(0, 0, 0, 0.8);
    return result;
  };
  models.push_back(
    mapGrid(256, 12, false, true, mapToCoil, identity, sinCosTurns));

  return models;
}
