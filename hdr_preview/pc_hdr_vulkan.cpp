#include "pc_hdr_vulkan.h"

#include <vector>
#include <stdexcept>
#include <cstring>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <string>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <vulkan/vulkan_win32.h>

struct pc_hdr_context {
    HWND hwnd{};
    int videoWidth{};
    int videoHeight{};

    bool validationEnabled{false};
    bool debugUtilsEnabled{false};
    bool debugUtilsDeviceEnabled{false};
    bool uploadTracingEnabled{false};

    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    uint32_t graphicsQueueFamily{0};
    VkQueue graphicsQueue{VK_NULL_HANDLE};

    VkDebugUtilsMessengerEXT debugMessenger{VK_NULL_HANDLE};
    PFN_vkDestroyDebugUtilsMessengerEXT pfnDestroyDebugUtilsMessengerEXT{nullptr};
    PFN_vkSetDebugUtilsObjectNameEXT pfnSetDebugUtilsObjectNameEXT{nullptr};
    PFN_vkCmdBeginDebugUtilsLabelEXT pfnCmdBeginDebugUtilsLabelEXT{nullptr};
    PFN_vkCmdEndDebugUtilsLabelEXT pfnCmdEndDebugUtilsLabelEXT{nullptr};

    VkSurfaceKHR surface{VK_NULL_HANDLE};
    VkSwapchainKHR swapchain{VK_NULL_HANDLE};
    VkFormat swapchainFormat{VK_FORMAT_UNDEFINED};
    VkColorSpaceKHR swapchainColorSpace{VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    VkExtent2D swapchainExtent{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkFramebuffer> framebuffers;

    VkRenderPass renderPass{VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    VkPipeline pipeline{VK_NULL_HANDLE};
    VkPipelineCache pipelineCache{VK_NULL_HANDLE};

    VkCommandPool commandPool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> commandBuffers;

    VkSemaphore imageAvailableSemaphore{VK_NULL_HANDLE};
    VkSemaphore renderFinishedSemaphore{VK_NULL_HANDLE};
    std::vector<VkFence> swapchainImageFences;
    std::vector<bool> imageFenceInUse;

    // Y / UV data buffers (contiguous layout, repacked by CPU)
    VkBuffer yBuffer{VK_NULL_HANDLE};
    VkDeviceMemory yBufferMemory{VK_NULL_HANDLE};
    void* yBufferMapped{nullptr};

    VkBuffer uvBuffer{VK_NULL_HANDLE};
    VkDeviceMemory uvBufferMemory{VK_NULL_HANDLE};
    void* uvBufferMapped{nullptr};

    uint32_t yBufferElements{};  // = width * height
    uint32_t uvBufferElements{}; // = (width/2)*(height/2)

    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool descriptorPool{VK_NULL_HANDLE};
    VkDescriptorSet descriptorSet{VK_NULL_HANDLE};

    // Push constants: video dimensions
    struct PushConstants {
        int width;
        int height;
    } pushConstants{};

    // HDR metadata
    PFN_vkSetHdrMetadataEXT pfnSetHdrMetadataEXT{nullptr};
    bool hdrSurfaceRequested{false};
    bool hdrMetadataEnabled{false};
};

static bool env_truthy(const char* value) {
    if (!value) {
        return false;
    }
    std::string lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
}

static constexpr const wchar_t* kPipelineCacheFilename = L"pc_hdr_pipelines.cache";

static std::vector<char> readFileIfExists(const wchar_t* filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

static void writeBinaryFile(const wchar_t* filename, const std::vector<char>& data) {
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return;
    }
    file.write(data.data(), static_cast<std::streamsize>(data.size()));
    file.close();
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* /*userData*/) {
    const char* severity = "INFO";
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        severity = "ERROR";
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        severity = "WARN";
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        severity = "VERBOSE";
    }

    char buffer[2048];
    std::snprintf(buffer,
                  sizeof(buffer),
                  "[pc_hdr_vulkan][%s] (%u) %s\n",
                  severity,
                  messageType,
                  callbackData && callbackData->pMessage ? callbackData->pMessage : "(no message)");
    OutputDebugStringA(buffer);
    return VK_FALSE;
}

static void setObjectName(pc_hdr_context* ctx,
                          uint64_t handle,
                          VkObjectType type,
                          const std::string& name) {
    if (!ctx || !ctx->pfnSetDebugUtilsObjectNameEXT || handle == 0) {
        return;
    }
    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name.c_str();
    ctx->pfnSetDebugUtilsObjectNameEXT(ctx->device, &info);
}

static void traceUpload(pc_hdr_context* ctx, const char* fmt, ...) {
    if (!ctx || !ctx->uploadTracingEnabled) {
        return;
    }
    char buffer[256];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    OutputDebugStringA(buffer);
}

static void beginLabel(pc_hdr_context* ctx,
                       VkCommandBuffer cmd,
                       const char* name,
                       const float color[4]) {
    if (!ctx || !ctx->pfnCmdBeginDebugUtilsLabelEXT) {
        return;
    }
    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    if (color) {
        std::copy(color, color + 4, label.color);
    }
    ctx->pfnCmdBeginDebugUtilsLabelEXT(cmd, &label);
}

static void endLabel(pc_hdr_context* ctx, VkCommandBuffer cmd) {
    if (!ctx || !ctx->pfnCmdEndDebugUtilsLabelEXT) {
        return;
    }
    ctx->pfnCmdEndDebugUtilsLabelEXT(cmd);
}

static void vk_check(VkResult res, const char* where) {
    if (res != VK_SUCCESS) {
        throw std::runtime_error(std::string("Vulkan error at ") + where);
    }
}

static std::vector<char> readFile(const wchar_t* filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    vk_check(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule), "vkCreateShaderModule");
    return shaderModule;
}

// Choose HDR surface format if available, else fallback.
static void chooseSurfaceFormat(pc_hdr_context* ctx) {
    uint32_t count = 0;
    vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(ctx->physicalDevice, ctx->surface, &count, nullptr),
             "vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
    if (count == 0) {
        throw std::runtime_error("No surface formats");
    }
    std::vector<VkSurfaceFormatKHR> formats(count);
    vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(ctx->physicalDevice, ctx->surface, &count, formats.data()),
             "vkGetPhysicalDeviceSurfaceFormatsKHR(data)");

    const bool wantHdrSurface = ctx->hdrSurfaceRequested;

    if (wantHdrSurface) {
        for (const auto& f : formats) {
            if (f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
                ctx->swapchainFormat = f.format;
                ctx->swapchainColorSpace = f.colorSpace;
                return;
            }
        }
        // Dump available formats/colorspaces for debugging.
        OutputDebugStringA("[pc_hdr] ERROR: HDR10 colorspace not available on this surface.\n");
        for (size_t i = 0; i < formats.size(); ++i) {
            char buf[256];
            std::snprintf(
                buf, sizeof(buf),
                "[pc_hdr] surface format[%zu]: format=%d colorspace=%d\n",
                i,
                (int)formats[i].format,
                (int)formats[i].colorSpace
            );
            OutputDebugStringA(buf);
        }
        throw std::runtime_error("HDR10 swapchain colorspace not available");
    }

    for (const auto& f : formats) {
        if (f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            ctx->swapchainFormat = f.format;
            ctx->swapchainColorSpace = f.colorSpace;
            return;
        }
    }

    ctx->swapchainFormat = formats[0].format;
    ctx->swapchainColorSpace = formats[0].colorSpace;
}

// Choose queue family with graphics+present.
static void chooseQueueFamily(pc_hdr_context* ctx) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physicalDevice, &count, nullptr);
    if (count == 0) throw std::runtime_error("No queue families");

    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physicalDevice, &count, props.data());

    for (uint32_t i = 0; i < count; ++i) {
        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(ctx->physicalDevice, i, ctx->surface, &presentSupport);
        if ((props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && presentSupport) {
            ctx->graphicsQueueFamily = i;
            return;
        }
    }
    throw std::runtime_error("No graphics+present queue family");
}

static void createSwapchain(pc_hdr_context* ctx, int widgetWidth, int widgetHeight) {
    // Query surface capabilities
    VkSurfaceCapabilitiesKHR caps{};
    vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx->physicalDevice, ctx->surface, &caps),
             "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    VkExtent2D extent{};
    if (caps.currentExtent.width != UINT32_MAX) {
        extent = caps.currentExtent;
    } else {
        extent.width  = (uint32_t)std::max((int)caps.minImageExtent.width,
                                           std::min(widgetWidth,  (int)caps.maxImageExtent.width));
        extent.height = (uint32_t)std::max((int)caps.minImageExtent.height,
                                           std::min(widgetHeight, (int)caps.maxImageExtent.height));
    }

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = ctx->surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = ctx->swapchainFormat;
    createInfo.imageColorSpace = ctx->swapchainColorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t queueFamilyIndices[] = { ctx->graphicsQueueFamily };
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 1;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;

    createInfo.preTransform = caps.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = ctx->swapchain;

    VkSwapchainKHR old = ctx->swapchain;
    vk_check(vkCreateSwapchainKHR(ctx->device, &createInfo, nullptr, &ctx->swapchain),
             "vkCreateSwapchainKHR");
    setObjectName(ctx,
                  reinterpret_cast<uint64_t>(ctx->swapchain),
                  VK_OBJECT_TYPE_SWAPCHAIN_KHR,
                  "pc_hdr_swapchain");
    if (old != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(ctx->device, old, nullptr);
    }

    ctx->swapchainExtent = extent;

    ctx->swapchainImages.clear();
    uint32_t count = 0;
    vk_check(vkGetSwapchainImagesKHR(ctx->device, ctx->swapchain, &count, nullptr),
             "vkGetSwapchainImagesKHR(count)");
    ctx->swapchainImages.resize(count);
    vk_check(vkGetSwapchainImagesKHR(ctx->device, ctx->swapchain, &count, ctx->swapchainImages.data()),
             "vkGetSwapchainImagesKHR(data)");
    for (size_t i = 0; i < ctx->swapchainImages.size(); ++i) {
        setObjectName(ctx,
                      reinterpret_cast<uint64_t>(ctx->swapchainImages[i]),
                      VK_OBJECT_TYPE_IMAGE,
                      std::string("pc_hdr_swapchain_image_") + std::to_string(i));
    }

    // Image views
    for (auto view : ctx->swapchainImageViews) {
        vkDestroyImageView(ctx->device, view, nullptr);
    }
    ctx->swapchainImageViews.clear();
    ctx->swapchainImageViews.reserve(count);

    for (auto image : ctx->swapchainImages) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = ctx->swapchainFormat;
        viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        vk_check(vkCreateImageView(ctx->device, &viewInfo, nullptr, &imageView),
                 "vkCreateImageView");
        setObjectName(ctx,
                      reinterpret_cast<uint64_t>(imageView),
                      VK_OBJECT_TYPE_IMAGE_VIEW,
                      std::string("pc_hdr_swapchain_view_") + std::to_string(ctx->swapchainImageViews.size()));
        ctx->swapchainImageViews.push_back(imageView);
    }
}

// Render pass and framebuffers
static void createRenderPass(pc_hdr_context* ctx) {
    if (ctx->renderPass != VK_NULL_HANDLE) return;

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = ctx->swapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &colorAttachment;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;

    vk_check(vkCreateRenderPass(ctx->device, &rpInfo, nullptr, &ctx->renderPass),
             "vkCreateRenderPass");
    setObjectName(ctx,
                  reinterpret_cast<uint64_t>(ctx->renderPass),
                  VK_OBJECT_TYPE_RENDER_PASS,
                  "pc_hdr_render_pass");
}

static void createFramebuffers(pc_hdr_context* ctx) {
    for (auto fb : ctx->framebuffers) {
        vkDestroyFramebuffer(ctx->device, fb, nullptr);
    }
    ctx->framebuffers.clear();
    ctx->framebuffers.reserve(ctx->swapchainImageViews.size());

    for (auto view : ctx->swapchainImageViews) {
        VkImageView attachments[] = { view };

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = ctx->renderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = attachments;
        fbInfo.width = ctx->swapchainExtent.width;
        fbInfo.height = ctx->swapchainExtent.height;
        fbInfo.layers = 1;

        VkFramebuffer fb;
        vk_check(vkCreateFramebuffer(ctx->device, &fbInfo, nullptr, &fb),
                 "vkCreateFramebuffer");
        setObjectName(ctx,
                      reinterpret_cast<uint64_t>(fb),
                      VK_OBJECT_TYPE_FRAMEBUFFER,
                      std::string("pc_hdr_framebuffer_") + std::to_string(ctx->framebuffers.size()));
        ctx->framebuffers.push_back(fb);
    }
}

// Create Y/UV storage buffers
static void createBuffers(pc_hdr_context* ctx) {
    VkDeviceSize ySize = VkDeviceSize(ctx->videoWidth) * VkDeviceSize(ctx->videoHeight) * sizeof(uint32_t);
    VkDeviceSize uvSize = VkDeviceSize(ctx->videoWidth / 2) * VkDeviceSize(ctx->videoHeight / 2) * sizeof(uint32_t);

    ctx->yBufferElements = ctx->videoWidth * ctx->videoHeight;
    ctx->uvBufferElements = (ctx->videoWidth / 2) * (ctx->videoHeight / 2);

    auto createBuffer = [&](VkDeviceSize size, VkBuffer& outBuf, VkDeviceMemory& outMem, void** outMapped) {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = size;
        bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vk_check(vkCreateBuffer(ctx->device, &bufInfo, nullptr, &outBuf), "vkCreateBuffer");

        VkMemoryRequirements memReq{};
        vkGetBufferMemoryRequirements(ctx->device, outBuf, &memReq);

        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(ctx->physicalDevice, &memProps);

        uint32_t memTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReq.memoryTypeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags &
                 (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                memTypeIndex = i;
                break;
            }
        }
        if (memTypeIndex == UINT32_MAX) throw std::runtime_error("No HOST_VISIBLE memory type");

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memTypeIndex;

        vk_check(vkAllocateMemory(ctx->device, &allocInfo, nullptr, &outMem), "vkAllocateMemory");
        vk_check(vkBindBufferMemory(ctx->device, outBuf, outMem, 0), "vkBindBufferMemory");

        if (outMapped) {
            vk_check(vkMapMemory(ctx->device, outMem, 0, size, 0, outMapped), "vkMapMemory");
        }
    };

    createBuffer(ySize, ctx->yBuffer, ctx->yBufferMemory, &ctx->yBufferMapped);
    setObjectName(ctx,
                  reinterpret_cast<uint64_t>(ctx->yBuffer),
                  VK_OBJECT_TYPE_BUFFER,
                  "pc_hdr_y_buffer");
    createBuffer(uvSize, ctx->uvBuffer, ctx->uvBufferMemory, &ctx->uvBufferMapped);
    setObjectName(ctx,
                  reinterpret_cast<uint64_t>(ctx->uvBuffer),
                  VK_OBJECT_TYPE_BUFFER,
                  "pc_hdr_uv_buffer");
}

// Descriptor set (storage buffers)
static void createDescriptors(pc_hdr_context* ctx) {
    // Layout
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    vk_check(vkCreateDescriptorSetLayout(ctx->device, &layoutInfo, nullptr, &ctx->descriptorSetLayout),
             "vkCreateDescriptorSetLayout");
    setObjectName(ctx,
                  reinterpret_cast<uint64_t>(ctx->descriptorSetLayout),
                  VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                  "pc_hdr_descriptor_set_layout");

    // Pool
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    vk_check(vkCreateDescriptorPool(ctx->device, &poolInfo, nullptr, &ctx->descriptorPool),
             "vkCreateDescriptorPool");
    setObjectName(ctx,
                  reinterpret_cast<uint64_t>(ctx->descriptorPool),
                  VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                  "pc_hdr_descriptor_pool");

    // Allocate
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = ctx->descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &ctx->descriptorSetLayout;

    vk_check(vkAllocateDescriptorSets(ctx->device, &allocInfo, &ctx->descriptorSet),
             "vkAllocateDescriptorSets");
    setObjectName(ctx,
                  reinterpret_cast<uint64_t>(ctx->descriptorSet),
                  VK_OBJECT_TYPE_DESCRIPTOR_SET,
                  "pc_hdr_descriptor_set");

    // Update
    VkDescriptorBufferInfo yInfo{};
    yInfo.buffer = ctx->yBuffer;
    yInfo.offset = 0;
    yInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo uvInfo{};
    uvInfo.buffer = ctx->uvBuffer;
    uvInfo.offset = 0;
    uvInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[2]{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = ctx->descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &yInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = ctx->descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &uvInfo;

    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, nullptr);
}

static void createPipelineCache(pc_hdr_context* ctx) {
    if (ctx->pipelineCache != VK_NULL_HANDLE) {
        return;
    }
    auto cacheData = readFileIfExists(kPipelineCacheFilename);
    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = cacheData.size();
    cacheInfo.pInitialData = cacheData.empty() ? nullptr : cacheData.data();

    vk_check(vkCreatePipelineCache(ctx->device, &cacheInfo, nullptr, &ctx->pipelineCache),
             "vkCreatePipelineCache");
    setObjectName(ctx,
                  reinterpret_cast<uint64_t>(ctx->pipelineCache),
                  VK_OBJECT_TYPE_PIPELINE_CACHE,
                  "pc_hdr_pipeline_cache");
}

static void savePipelineCache(pc_hdr_context* ctx) {
    if (!ctx || ctx->pipelineCache == VK_NULL_HANDLE) {
        return;
    }
    size_t dataSize = 0;
    if (vkGetPipelineCacheData(ctx->device, ctx->pipelineCache, &dataSize, nullptr) != VK_SUCCESS ||
        dataSize == 0) {
        return;
    }
    std::vector<char> data(dataSize);
    if (vkGetPipelineCacheData(ctx->device, ctx->pipelineCache, &dataSize, data.data()) == VK_SUCCESS) {
        writeBinaryFile(kPipelineCacheFilename, data);
    }
}

static void chooseViewportForVideoAspect(pc_hdr_context* ctx,
                                         VkViewport& viewport,
                                         VkRect2D& scissor) {
    float videoAspect =
        ctx->videoHeight > 0
        ? static_cast<float>(ctx->videoWidth) / static_cast<float>(ctx->videoHeight)
        : 1.0f;

    float swapAspect =
        ctx->swapchainExtent.height > 0
        ? static_cast<float>(ctx->swapchainExtent.width) /
              static_cast<float>(ctx->swapchainExtent.height)
        : 1.0f;

    float vpWidth = static_cast<float>(ctx->swapchainExtent.width);
    float vpHeight = static_cast<float>(ctx->swapchainExtent.height);
    float vpX = 0.0f;
    float vpY = 0.0f;

    if (swapAspect > videoAspect) {
        // Window is wider than the video: pillarbox
        vpHeight = static_cast<float>(ctx->swapchainExtent.height);
        vpWidth = vpHeight * videoAspect;
        vpX = 0.5f * (static_cast<float>(ctx->swapchainExtent.width) - vpWidth);
        vpY = 0.0f;
    } else if (swapAspect < videoAspect) {
        // Window is taller than the video: letterbox
        vpWidth = static_cast<float>(ctx->swapchainExtent.width);
        vpHeight = vpWidth / videoAspect;
        vpX = 0.0f;
        vpY = 0.5f * (static_cast<float>(ctx->swapchainExtent.height) - vpHeight);
    }

    viewport.x = vpX;
    viewport.y = vpY;
    viewport.width = vpWidth;
    viewport.height = vpHeight;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    scissor.offset.x = static_cast<int32_t>(std::max(0.0f, vpX));
    scissor.offset.y = static_cast<int32_t>(std::max(0.0f, vpY));
    scissor.extent.width = static_cast<uint32_t>(
        std::min(vpWidth, static_cast<float>(ctx->swapchainExtent.width) - vpX));
    scissor.extent.height = static_cast<uint32_t>(
        std::min(vpHeight, static_cast<float>(ctx->swapchainExtent.height) - vpY));
}

// Graphics pipeline
static void createPipeline(pc_hdr_context* ctx) {
    createPipelineCache(ctx);
    // Load SPIR-V from files next to the DLL (adjust path as needed)
    auto vertCode = readFile(L"pc_hdr_vert.spv");
    auto fragCode = readFile(L"pc_hdr_frag.spv");

    VkShaderModule vertModule = createShaderModule(ctx->device, vertCode);
    VkShaderModule fragModule = createShaderModule(ctx->device, fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    VkRect2D scissor{};

    chooseViewportForVideoAspect(ctx, viewport, scissor);

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.pViewports = &viewport;
    vp.scissorCount = 1;
    vp.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cbAttach{};
    cbAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cbAttach.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cbAttach;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(pc_hdr_context::PushConstants);

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &ctx->descriptorSetLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;

    vk_check(vkCreatePipelineLayout(ctx->device, &plInfo, nullptr, &ctx->pipelineLayout),
             "vkCreatePipelineLayout");
    setObjectName(ctx,
                  reinterpret_cast<uint64_t>(ctx->pipelineLayout),
                  VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                  "pc_hdr_pipeline_layout");

    VkGraphicsPipelineCreateInfo gp{};
    gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb;
    gp.layout = ctx->pipelineLayout;
    gp.renderPass = ctx->renderPass;
    gp.subpass = 0;

    vk_check(vkCreateGraphicsPipelines(ctx->device,
                                       ctx->pipelineCache,
                                       1,
                                       &gp,
                                       nullptr,
                                       &ctx->pipeline),
             "vkCreateGraphicsPipelines");
    setObjectName(ctx,
                  reinterpret_cast<uint64_t>(ctx->pipeline),
                  VK_OBJECT_TYPE_PIPELINE,
                  "pc_hdr_pipeline");

    vkDestroyShaderModule(ctx->device, vertModule, nullptr);
    vkDestroyShaderModule(ctx->device, fragModule, nullptr);
}

// Command pool / buffers / sync
static void createCommands(pc_hdr_context* ctx) {
    if (ctx->commandPool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = ctx->graphicsQueueFamily;
        vk_check(vkCreateCommandPool(ctx->device, &poolInfo, nullptr, &ctx->commandPool),
                 "vkCreateCommandPool");
        setObjectName(ctx,
                      reinterpret_cast<uint64_t>(ctx->commandPool),
                      VK_OBJECT_TYPE_COMMAND_POOL,
                      "pc_hdr_command_pool");
    }

    if (!ctx->commandBuffers.empty()) {
        vkFreeCommandBuffers(ctx->device, ctx->commandPool,
                             static_cast<uint32_t>(ctx->commandBuffers.size()),
                             ctx->commandBuffers.data());
        ctx->commandBuffers.clear();
    }

    ctx->commandBuffers.resize(ctx->swapchainImages.size());

    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = ctx->commandPool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = static_cast<uint32_t>(ctx->commandBuffers.size());

    vk_check(vkAllocateCommandBuffers(ctx->device, &alloc, ctx->commandBuffers.data()),
             "vkAllocateCommandBuffers");
    for (size_t i = 0; i < ctx->commandBuffers.size(); ++i) {
        setObjectName(ctx,
                      reinterpret_cast<uint64_t>(ctx->commandBuffers[i]),
                      VK_OBJECT_TYPE_COMMAND_BUFFER,
                      std::string("pc_hdr_command_buffer_") + std::to_string(i));
    }

    for (auto fence : ctx->swapchainImageFences) {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(ctx->device, fence, nullptr);
        }
    }
    ctx->swapchainImageFences.clear();
    ctx->imageFenceInUse.assign(ctx->commandBuffers.size(), false);
    ctx->swapchainImageFences.resize(ctx->commandBuffers.size(), VK_NULL_HANDLE);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (size_t i = 0; i < ctx->swapchainImageFences.size(); ++i) {
        vk_check(vkCreateFence(ctx->device, &fenceInfo, nullptr, &ctx->swapchainImageFences[i]),
                 "vkCreateFence");
        setObjectName(ctx,
                      reinterpret_cast<uint64_t>(ctx->swapchainImageFences[i]),
                      VK_OBJECT_TYPE_FENCE,
                      std::string("pc_hdr_inflight_fence_") + std::to_string(i));
    }

    if (ctx->imageAvailableSemaphore == VK_NULL_HANDLE) {
        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vk_check(vkCreateSemaphore(ctx->device, &semInfo, nullptr, &ctx->imageAvailableSemaphore),
                 "vkCreateSemaphore");
        vk_check(vkCreateSemaphore(ctx->device, &semInfo, nullptr, &ctx->renderFinishedSemaphore),
                 "vkCreateSemaphore");
        setObjectName(ctx,
                      reinterpret_cast<uint64_t>(ctx->imageAvailableSemaphore),
                      VK_OBJECT_TYPE_SEMAPHORE,
                      "pc_hdr_image_available_semaphore");
        setObjectName(ctx,
                      reinterpret_cast<uint64_t>(ctx->renderFinishedSemaphore),
                      VK_OBJECT_TYPE_SEMAPHORE,
                      "pc_hdr_render_finished_semaphore");
    }
}

// Set HDR metadata for the swapchain (only when the extension is enabled).
static void setHdrMetadata(pc_hdr_context* ctx) {
    ctx->pfnSetHdrMetadataEXT = nullptr;
    if (!ctx->hdrSurfaceRequested || !ctx->hdrMetadataEnabled) {
        return;
    }

    ctx->pfnSetHdrMetadataEXT =
        reinterpret_cast<PFN_vkSetHdrMetadataEXT>(vkGetDeviceProcAddr(ctx->device, "vkSetHdrMetadataEXT"));
    if (!ctx->pfnSetHdrMetadataEXT) return;

    VkHdrMetadataEXT meta{};
    meta.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT;

    // BT.2020 primaries (approx.)
    meta.displayPrimaryRed   =  { 0.708f, 0.292f };
    meta.displayPrimaryGreen =  { 0.170f, 0.797f };
    meta.displayPrimaryBlue  =  { 0.131f, 0.046f };
    meta.whitePoint          =  { 0.3127f, 0.3290f };

    // Luminance in nits
    meta.maxLuminance = 1000.0f;
    meta.minLuminance = 0.001f;
    meta.maxContentLightLevel = 1000.0f;
    meta.maxFrameAverageLightLevel = 400.0f;

    ctx->pfnSetHdrMetadataEXT(ctx->device, 1, &ctx->swapchain, &meta);
}

// Record a command buffer for one imageIndex.
static void recordCommandBuffer(pc_hdr_context* ctx, uint32_t imageIndex) {
    VkCommandBuffer cmd = ctx->commandBuffers[imageIndex];

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");

    const float hdrLabelColor[4] = {0.35f, 0.65f, 0.95f, 1.0f};
    beginLabel(ctx, cmd, "pc_hdr_present", hdrLabelColor);

    VkClearValue clear{};
    clear.color = { {0.0f, 0.0f, 0.0f, 1.0f} };

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = ctx->renderPass;
    rpBegin.framebuffer = ctx->framebuffers[imageIndex];
    rpBegin.renderArea.offset = {0, 0};
    rpBegin.renderArea.extent = ctx->swapchainExtent;
    rpBegin.clearValueCount = 1;
    rpBegin.pClearValues = &clear;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->pipeline);

    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            ctx->pipelineLayout,
                            0, 1, &ctx->descriptorSet,
                            0, nullptr);

    vkCmdPushConstants(cmd,
                       ctx->pipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(ctx->pushConstants),
                       &ctx->pushConstants);

    vkCmdDraw(cmd, 3, 1, 0, 0); // fullscreen triangle

    vkCmdEndRenderPass(cmd);
    endLabel(ctx, cmd);

    vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
}

// Upload one frame worth of P010 into the storage buffers (repacking).
static void uploadP010ToBuffers(pc_hdr_context* ctx,
                                const std::uint16_t* yPlane,
                                const std::uint16_t* uvPlane,
                                int strideYBytes,
                                int strideUVBytes) {
    uint32_t W = ctx->videoWidth;
    uint32_t H = ctx->videoHeight;

    auto* yDst = reinterpret_cast<uint32_t*>(ctx->yBufferMapped);
    auto* uvDst = reinterpret_cast<uint32_t*>(ctx->uvBufferMapped);

    int strideY = strideYBytes / 2;   // 16-bit samples
    int strideUV = strideUVBytes / 2;

    traceUpload(ctx,
                "[pc_hdr] upload begin W=%u H=%u strideY=%d strideUV=%d\n",
                W,
                H,
                strideYBytes,
                strideUVBytes);

    // Y: one uint per pixel (lower 16 bits store the raw P010 sample)
    for (uint32_t y = 0; y < H; ++y) {
        const std::uint16_t* srcRow = yPlane + y * strideY;
        uint32_t* dstRow = yDst + y * W;
        for (uint32_t x = 0; x < W; ++x) {
            // P010: 16-bit word with 10-bit code in the upper bits; keep it as-is.
            dstRow[x] = static_cast<uint32_t>(srcRow[x]);
        }
    }

    // UV: each uint stores one interleaved UV pair (U in high 16, V in low 16)
    uint32_t Wc = W / 2;
    uint32_t Hc = H / 2;
    for (uint32_t y = 0; y < Hc; ++y) {
        const std::uint16_t* srcRow = uvPlane + y * strideUV;
        uint32_t* dstRow = uvDst + y * Wc;
        for (uint32_t x = 0; x < Wc; ++x) {
            // Store the raw 16-bit P010 chroma samples (U in high 16, V in low 16).
            std::uint16_t u = srcRow[2 * x + 0];
            std::uint16_t v = srcRow[2 * x + 1];
            dstRow[x] = (static_cast<uint32_t>(u) << 16) |
                        static_cast<uint32_t>(v);
        }
    }

    traceUpload(ctx, "[pc_hdr] upload finished\n");
}

/*
GLSL shaders (compile to SPIR-V as pc_hdr_vert.spv / pc_hdr_frag.spv):

// pc_hdr_vert.glsl
#version 450
layout(location = 0) out vec2 vUV;
void main() {
    const vec2 pos[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
    vUV = (gl_Position.xy * 0.5) + 0.5;
}

// pc_hdr_frag.glsl
#version 450
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(std430, binding = 0) readonly buffer YBuf {
    uint yData[];
};
layout(std430, binding = 1) readonly buffer UVBuf {
    uint uvData[];
};

layout(push_constant) uniform Push {
    int width;
    int height;
} pc;

// Simple BT.2020 YCbCr->RGB matrix (approx, non-constant luminance)
vec3 yuv_to_rgb_bt2020(float Y, float Cb, float Cr) {
    float R = Y + 1.4746 * Cr;
    float G = Y - 0.164553 * Cb - 0.571353 * Cr;
    float B = Y + 1.8814 * Cb;
    return vec3(R, G, B);
}

void main() {
    ivec2 coord = ivec2(vUV * vec2(pc.width, pc.height));
    coord = clamp(coord, ivec2(0), ivec2(pc.width - 1, pc.height - 1));

    int idxY = coord.y * pc.width + coord.x;
    int idxUV = (coord.y / 2) * (pc.width / 2) + (coord.x / 2);

    uint yRaw = yData[idxY];
    uint uvRaw = uvData[idxUV];

    float Y = float(yRaw & 0xFFFFu) / 65535.0; // PQ-coded luma, normalized
    uint uRaw = (uvRaw >> 16) & 0xFFFFu;
    uint vRaw = uvRaw & 0xFFFFu;

    float Cb = float(uRaw) / 65535.0 - 0.5;
    float Cr = float(vRaw) / 65535.0 - 0.5;

    vec3 rgb = yuv_to_rgb_bt2020(Y, Cb, Cr);
    outColor = vec4(rgb, 1.0);
}
*/

extern "C" {

pc_hdr_context* pc_hdr_init(HWND hwnd, int width, int height) {
    pc_hdr_context* ctx = nullptr;
    try {
        ctx = new pc_hdr_context();
        ctx->hwnd = hwnd;
        ctx->videoWidth = width;
        ctx->videoHeight = height;
        ctx->pushConstants.width = width;
        ctx->pushConstants.height = height;

        // Derive the initial widget dimensions from the window's client area to
        // avoid creating a zero-sized swapchain when the window is not yet
        // fully realized. If the client rect is empty, fall back to the video
        // size passed by the caller.
        RECT rc{};
        GetClientRect(hwnd, &rc);
        int widgetW = rc.right - rc.left;
        int widgetH = rc.bottom - rc.top;
        if (widgetW <= 0 || widgetH <= 0) {
            widgetW = width;
            widgetH = height;
        }
        // STRICT HDR: default to HDR10 swapchain when running in passthrough mode.
        // PC_HDR_SWAPCHAIN_HDR can only force-disable HDR (0/false/off).
        bool hdrReq = true;
        if (const char* envHdr = std::getenv("PC_HDR_SWAPCHAIN_HDR")) {
            // If the user explicitly sets this to 0/false/off, treat it as "no HDR".
            if (!env_truthy(envHdr)) {
                hdrReq = false;
            }
        }
        ctx->hdrSurfaceRequested = hdrReq;
        ctx->uploadTracingEnabled = env_truthy(std::getenv("PC_HDR_TRACE_UPLOAD"));
        if (ctx->uploadTracingEnabled) {
            OutputDebugStringA("[pc_hdr] PC_HDR_TRACE_UPLOAD=1 (upload tracing enabled)\n");
        }

        bool validationRequested = false;
#if defined(_DEBUG)
        validationRequested = true;
#endif
        if (const char* envValidation = std::getenv("PC_HDR_VALIDATION")) {
            validationRequested = env_truthy(envValidation);
        }

        bool debugUtilsRequested = validationRequested;
#if defined(_DEBUG)
        debugUtilsRequested = true;
#endif
        if (const char* envDebug = std::getenv("PC_HDR_DEBUG_UTILS")) {
            debugUtilsRequested = env_truthy(envDebug);
        }

        // Instance
        // Build instance extension list dynamically so we don't fail on drivers
        // that do not expose VK_EXT_swapchain_colorspace.
        std::vector<const char*> instanceExts;
        instanceExts.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
        instanceExts.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);

        bool swapchainColorSpaceSupported = false;
        bool debugUtilsSupported = false;

        uint32_t instExtCount = 0;
        VkResult r = vkEnumerateInstanceExtensionProperties(nullptr, &instExtCount, nullptr);
        if (r == VK_SUCCESS && instExtCount > 0) {
            std::vector<VkExtensionProperties> instExtProps(instExtCount);
            r = vkEnumerateInstanceExtensionProperties(nullptr, &instExtCount, instExtProps.data());
            if (r == VK_SUCCESS) {
                for (const auto& ext : instExtProps) {
                    if (std::strcmp(ext.extensionName, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) == 0) {
                        swapchainColorSpaceSupported = true;
                    }
                    if (std::strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
                        debugUtilsSupported = true;
                    }
                }
            }
        }
        if (swapchainColorSpaceSupported) {
            instanceExts.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
        }
        if (debugUtilsRequested && debugUtilsSupported) {
            instanceExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            ctx->debugUtilsEnabled = true;
        }

        std::vector<const char*> instanceLayers;
        if (validationRequested) {
            uint32_t layerCount = 0;
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
            if (layerCount > 0) {
                std::vector<VkLayerProperties> layerProps(layerCount);
                vkEnumerateInstanceLayerProperties(&layerCount, layerProps.data());
                for (const auto& layer : layerProps) {
                    if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                        instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
                        ctx->validationEnabled = true;
                        break;
                    }
                }
            }
        }
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "PersonCapture HDR";
        app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app.pEngineName = "pc_hdr_vulkan";
        app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;
        ci.enabledExtensionCount = static_cast<uint32_t>(instanceExts.size());
        ci.ppEnabledExtensionNames = instanceExts.data();
        ci.enabledLayerCount = static_cast<uint32_t>(instanceLayers.size());
        ci.ppEnabledLayerNames = instanceLayers.empty() ? nullptr : instanceLayers.data();

        vk_check(vkCreateInstance(&ci, nullptr, &ctx->instance), "vkCreateInstance");

        if (ctx->debugUtilsEnabled && ctx->validationEnabled) {
            auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(ctx->instance, "vkCreateDebugUtilsMessengerEXT"));
            ctx->pfnDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(ctx->instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (createMessenger) {
                VkDebugUtilsMessengerCreateInfoEXT dbgInfo{};
                dbgInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
                dbgInfo.messageSeverity =
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
                dbgInfo.messageType =
                    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                dbgInfo.pfnUserCallback = debugMessengerCallback;
                vk_check(createMessenger(ctx->instance, &dbgInfo, nullptr, &ctx->debugMessenger),
                         "vkCreateDebugUtilsMessengerEXT");
            }
        }

        // Surface
        VkWin32SurfaceCreateInfoKHR surfInfo{};
        surfInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        surfInfo.hinstance = GetModuleHandleW(nullptr);
        surfInfo.hwnd = hwnd;
        vk_check(vkCreateWin32SurfaceKHR(ctx->instance, &surfInfo, nullptr, &ctx->surface),
                 "vkCreateWin32SurfaceKHR");

        // Physical device
        uint32_t devCount = 0;
        vk_check(vkEnumeratePhysicalDevices(ctx->instance, &devCount, nullptr),
                 "vkEnumeratePhysicalDevices(count)");
        if (devCount == 0) throw std::runtime_error("No Vulkan physical devices");
        std::vector<VkPhysicalDevice> devices(devCount);
        vk_check(vkEnumeratePhysicalDevices(ctx->instance, &devCount, devices.data()),
                 "vkEnumeratePhysicalDevices(data)");
        // simple: pick first
        ctx->physicalDevice = devices[0];

        chooseSurfaceFormat(ctx);
        chooseQueueFamily(ctx);

        // Device + queue
        float qprio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = ctx->graphicsQueueFamily;
        qci.queueCount = 1;
        qci.pQueuePriorities = &qprio;

        // Always require swapchain; add optional extensions when supported.
        std::vector<const char*> deviceExts;
        deviceExts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

        uint32_t devExtCount = 0;
        vk_check(vkEnumerateDeviceExtensionProperties(ctx->physicalDevice, nullptr, &devExtCount, nullptr),
                 "vkEnumerateDeviceExtensionProperties(count)");
        std::vector<VkExtensionProperties> devExtProps(devExtCount);
        vk_check(vkEnumerateDeviceExtensionProperties(ctx->physicalDevice,
                                                      nullptr,
                                                      &devExtCount,
                                                      devExtProps.data()),
                 "vkEnumerateDeviceExtensionProperties(data)");

        bool hdrMetadataSupported = false;
        bool debugUtilsDeviceSupported = false;
        for (const auto& ext : devExtProps) {
            if (std::strcmp(ext.extensionName, VK_EXT_HDR_METADATA_EXTENSION_NAME) == 0) {
                hdrMetadataSupported = true;
            }
            if (std::strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
                debugUtilsDeviceSupported = true;
            }
        }

        ctx->hdrMetadataEnabled = false;
        if (ctx->hdrSurfaceRequested && hdrMetadataSupported) {
            deviceExts.push_back(VK_EXT_HDR_METADATA_EXTENSION_NAME);
            ctx->hdrMetadataEnabled = true;
        }
        if (ctx->debugUtilsEnabled && debugUtilsDeviceSupported) {
            deviceExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            ctx->debugUtilsDeviceEnabled = true;
        }

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<uint32_t>(deviceExts.size());
        dci.ppEnabledExtensionNames = deviceExts.data();

        vk_check(vkCreateDevice(ctx->physicalDevice, &dci, nullptr, &ctx->device),
                 "vkCreateDevice");

        vkGetDeviceQueue(ctx->device, ctx->graphicsQueueFamily, 0, &ctx->graphicsQueue);

        if (ctx->debugUtilsDeviceEnabled) {
            ctx->pfnSetDebugUtilsObjectNameEXT = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                vkGetDeviceProcAddr(ctx->device, "vkSetDebugUtilsObjectNameEXT"));
            ctx->pfnCmdBeginDebugUtilsLabelEXT = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
                vkGetDeviceProcAddr(ctx->device, "vkCmdBeginDebugUtilsLabelEXT"));
            ctx->pfnCmdEndDebugUtilsLabelEXT = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
                vkGetDeviceProcAddr(ctx->device, "vkCmdEndDebugUtilsLabelEXT"));
        }

        // Swapchain + render path
        createSwapchain(ctx, widgetW, widgetH);
        createRenderPass(ctx);
        createFramebuffers(ctx);
        createBuffers(ctx);
        createDescriptors(ctx);
        createPipeline(ctx);
        createCommands(ctx);
        setHdrMetadata(ctx);

        return ctx;
    } catch (...) {
        if (ctx) {
            try {
                pc_hdr_shutdown(ctx);
            } catch (...) {
            }
        }
        return nullptr;
    }
}

void pc_hdr_resize(pc_hdr_context* ctx, int width, int height) {
    if (!ctx || width <= 0 || height <= 0) return;
    try {
        vkDeviceWaitIdle(ctx->device);

        createSwapchain(ctx, width, height);
        createFramebuffers(ctx);
        setHdrMetadata(ctx);

        // Update viewport/scissor-based pipeline to new extent
        if (ctx->pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(ctx->device, ctx->pipeline, nullptr);
            ctx->pipeline = VK_NULL_HANDLE;
        }
        createPipeline(ctx);
        createCommands(ctx);
    } catch (...) {
    }
}

static void recreateSwapchainForCurrentWindow(pc_hdr_context* ctx) {
    if (!ctx) return;

    RECT rc{};
    int w = 0;
    int h = 0;

    if (ctx->hwnd && GetClientRect(ctx->hwnd, &rc)) {
        w = rc.right - rc.left;
        h = rc.bottom - rc.top;
    }

    if (w <= 0 || h <= 0) {
        // Fallback to last known extent, then to video size.
        if (ctx->swapchainExtent.width > 0 && ctx->swapchainExtent.height > 0) {
            w = static_cast<int>(ctx->swapchainExtent.width);
            h = static_cast<int>(ctx->swapchainExtent.height);
        } else if (ctx->videoWidth > 0 && ctx->videoHeight > 0) {
            w = ctx->videoWidth;
            h = ctx->videoHeight;
        } else {
            return;
        }
    }

    pc_hdr_resize(ctx, w, h);
}

void pc_hdr_upload_p010(
    pc_hdr_context* ctx,
    const std::uint16_t* yPlane,
    const std::uint16_t* uvPlane,
    int strideY,
    int strideUV
) {
    if (!ctx || !yPlane || !uvPlane) return;
    try {
        uploadP010ToBuffers(ctx, yPlane, uvPlane, strideY, strideUV);
    } catch (...) {
        traceUpload(ctx, "[pc_hdr] upload exception; dropping frame\n");
    }
}

void pc_hdr_present(pc_hdr_context* ctx) {
    if (!ctx) return;
    try {
        uint32_t imageIndex = 0;
        VkResult acq = vkAcquireNextImageKHR(ctx->device,
                                             ctx->swapchain,
                                             UINT64_MAX,
                                             ctx->imageAvailableSemaphore,
                                             VK_NULL_HANDLE,
                                             &imageIndex);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
            recreateSwapchainForCurrentWindow(ctx);
            return;  // skip this frame; next frame will use the rebuilt swapchain
        }
        vk_check(acq, "vkAcquireNextImageKHR");

        if (imageIndex >= ctx->swapchainImageFences.size()) {
            return;
        }
        VkFence frameFence = ctx->swapchainImageFences[imageIndex];
        if (ctx->imageFenceInUse[imageIndex]) {
            vkWaitForFences(ctx->device, 1, &frameFence, VK_TRUE, UINT64_MAX);
            ctx->imageFenceInUse[imageIndex] = false;
        }
        vkResetFences(ctx->device, 1, &frameFence);

        recordCommandBuffer(ctx, imageIndex);

        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &ctx->imageAvailableSemaphore;
        submit.pWaitDstStageMask = waitStages;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &ctx->commandBuffers[imageIndex];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &ctx->renderFinishedSemaphore;

        vk_check(vkQueueSubmit(ctx->graphicsQueue, 1, &submit, frameFence),
                 "vkQueueSubmit");
        ctx->imageFenceInUse[imageIndex] = true;

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &ctx->renderFinishedSemaphore;
        present.swapchainCount = 1;
        present.pSwapchains = &ctx->swapchain;
        present.pImageIndices = &imageIndex;

        VkResult pres = vkQueuePresentKHR(ctx->graphicsQueue, &present);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
            recreateSwapchainForCurrentWindow(ctx);
            return;
        }
        vk_check(pres, "vkQueuePresentKHR");
    } catch (...) {
    }
}

void pc_hdr_shutdown(pc_hdr_context* ctx) {
    if (!ctx) return;

    try {
        vkDeviceWaitIdle(ctx->device);
    } catch (...) {
    }

    savePipelineCache(ctx);

    if (ctx->yBufferMapped)  vkUnmapMemory(ctx->device, ctx->yBufferMemory);
    if (ctx->uvBufferMapped) vkUnmapMemory(ctx->device, ctx->uvBufferMemory);

    if (ctx->yBuffer != VK_NULL_HANDLE)        vkDestroyBuffer(ctx->device, ctx->yBuffer, nullptr);
    if (ctx->yBufferMemory != VK_NULL_HANDLE)  vkFreeMemory(ctx->device, ctx->yBufferMemory, nullptr);
    if (ctx->uvBuffer != VK_NULL_HANDLE)       vkDestroyBuffer(ctx->device, ctx->uvBuffer, nullptr);
    if (ctx->uvBufferMemory != VK_NULL_HANDLE) vkFreeMemory(ctx->device, ctx->uvBufferMemory, nullptr);

    if (ctx->descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx->device, ctx->descriptorPool, nullptr);
    if (ctx->descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx->device, ctx->descriptorSetLayout, nullptr);

    if (ctx->pipeline != VK_NULL_HANDLE)       vkDestroyPipeline(ctx->device, ctx->pipeline, nullptr);
    if (ctx->pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx->device, ctx->pipelineLayout, nullptr);
    if (ctx->pipelineCache != VK_NULL_HANDLE)  vkDestroyPipelineCache(ctx->device, ctx->pipelineCache, nullptr);

    for (auto fb : ctx->framebuffers) {
        vkDestroyFramebuffer(ctx->device, fb, nullptr);
    }
    ctx->framebuffers.clear();

    if (ctx->renderPass != VK_NULL_HANDLE)     vkDestroyRenderPass(ctx->device, ctx->renderPass, nullptr);

    if (ctx->commandPool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(ctx->device, ctx->commandPool,
                             static_cast<uint32_t>(ctx->commandBuffers.size()),
                             ctx->commandBuffers.data());
        vkDestroyCommandPool(ctx->device, ctx->commandPool, nullptr);
    }

    for (auto fence : ctx->swapchainImageFences) {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(ctx->device, fence, nullptr);
        }
    }
    ctx->swapchainImageFences.clear();
    ctx->imageFenceInUse.clear();

    if (ctx->imageAvailableSemaphore != VK_NULL_HANDLE)
        vkDestroySemaphore(ctx->device, ctx->imageAvailableSemaphore, nullptr);
    if (ctx->renderFinishedSemaphore != VK_NULL_HANDLE)
        vkDestroySemaphore(ctx->device, ctx->renderFinishedSemaphore, nullptr);

    for (auto iv : ctx->swapchainImageViews) {
        vkDestroyImageView(ctx->device, iv, nullptr);
    }
    ctx->swapchainImageViews.clear();

    if (ctx->swapchain != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(ctx->device, ctx->swapchain, nullptr);

    if (ctx->surface != VK_NULL_HANDLE)
        vkDestroySurfaceKHR(ctx->instance, ctx->surface, nullptr);

    if (ctx->device != VK_NULL_HANDLE)
        vkDestroyDevice(ctx->device, nullptr);

    if (ctx->debugMessenger != VK_NULL_HANDLE && ctx->pfnDestroyDebugUtilsMessengerEXT) {
        ctx->pfnDestroyDebugUtilsMessengerEXT(ctx->instance, ctx->debugMessenger, nullptr);
    }

    if (ctx->instance != VK_NULL_HANDLE)
        vkDestroyInstance(ctx->instance, nullptr);

    delete ctx;
}

} // extern "C"
