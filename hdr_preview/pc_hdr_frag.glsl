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
    int videoWidth;
    int videoHeight;
    int outWidth;
    int outHeight;
} pc;

// Simple BT.2020 YCbCr->RGB matrix (approx, non-constant luminance)
vec3 yuv_to_rgb_bt2020(float Y, float Cb, float Cr) {
    float R = Y + 1.4746 * Cr;
    float G = Y - 0.164553 * Cb - 0.571353 * Cr;
    float B = Y + 1.8814 * Cb;
    return vec3(R, G, B);
}

void main() {
    float videoAspect = float(pc.videoWidth) / float(pc.videoHeight);
    float outAspect   = float(pc.outWidth) / float(pc.outHeight);

    vec2 uv = vUV;

    if (outAspect > videoAspect) {
        float scale = videoAspect / outAspect;
        uv.x = (uv.x - 0.5) / scale + 0.5;
    } else if (outAspect < videoAspect) {
        float scale = outAspect / videoAspect;
        uv.y = (uv.y - 0.5) / scale + 0.5;
    }

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    ivec2 coord = ivec2(uv * vec2(pc.videoWidth, pc.videoHeight));
    coord = clamp(coord, ivec2(0), ivec2(pc.videoWidth - 1, pc.videoHeight - 1));

    int idxY  = coord.y * pc.videoWidth + coord.x;
    int idxUV = (coord.y / 2) * (pc.videoWidth / 2) + (coord.x / 2);

    uint yRaw = yData[idxY];
    uint uvRaw = uvData[idxUV];

    // P010: 10-bit code in the upper bits of a 16-bit word.
    uint y10 = (yRaw & 0xFFFFu) >> 6;

    // Map HDR10 limited range [64, 940] in 10-bit to [0, 1] PQ code.
    float Ycode = float(y10);
    float Yp = clamp((Ycode - 64.0) / (940.0 - 64.0), 0.0, 1.0);

    // Chroma: also 10-bit codes in upper bits of 16-bit.
    uint u16 = (uvRaw >> 16) & 0xFFFFu;
    uint v16 =  uvRaw        & 0xFFFFu;

    uint u10 = u16 >> 6;
    uint v10 = v16 >> 6;

    // Center around 0, normalizing the HDR10 limited chroma range (64–960 codes).
    float Cb = (float(u10) - 512.0) / 896.0;
    float Cr = (float(v10) - 512.0) / 896.0;

    // Convert BT.2020 Y′CbCr (non-constant luminance) to BT.2020 RGB′ in PQ code space.
    vec3 rgbPQ = yuv_to_rgb_bt2020(Yp, Cb, Cr);

    // Clamp to valid PQ code range.
    rgbPQ = clamp(rgbPQ, vec3(0.0), vec3(1.0));

    // Output PQ-coded BT.2020 RGB′ directly; swapchain is tagged HDR10/ST.2084.
    outColor = vec4(rgbPQ, 1.0);
}
