#pragma once

#include <windows.h>
#include <cstdint>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context type
struct pc_hdr_context;

// Initialize HDR preview on the given HWND.
// width/height are the decoded video frame size (not widget size).
__declspec(dllexport)
pc_hdr_context* pc_hdr_init(HWND hwnd, int width, int height);

// Notify about widget resize (swapchain recreation).
__declspec(dllexport)
void pc_hdr_resize(pc_hdr_context* ctx, int width, int height);

// Upload one P010 frame (yuv420p10le) for preview.
// yPlane / uvPlane: pointers to CPU-side planes.
// strideY / strideUV: strides in BYTES.
__declspec(dllexport)
void pc_hdr_upload_p010(
    pc_hdr_context* ctx,
    const std::uint16_t* yPlane,
    const std::uint16_t* uvPlane,
    int strideY,
    int strideUV
);

// Render the last uploaded frame to the HDR swapchain.
__declspec(dllexport)
void pc_hdr_present(pc_hdr_context* ctx);

// Destroy and free all Vulkan / OS resources.
__declspec(dllexport)
void pc_hdr_shutdown(pc_hdr_context* ctx);

#ifdef __cplusplus
}
#endif
