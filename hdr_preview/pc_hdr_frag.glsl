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

// NOTE: Strict HDR10 path for passthrough mode:
// - P010 10-bit luma codes (limited range) -> PQ code
// - PQ code -> linear relative luminance (SMPTE ST.2084)
// - Limited-range BT.2020 YCbCr -> BT.2020 RGB (linear)
// This expects an HDR10 ST.2084 swapchain, not SDR.

// Simple BT.2020 YCbCr->RGB matrix (approx, non-constant luminance)
vec3 yuv_to_rgb_bt2020(float Y, float Cb, float Cr) {
    float R = Y + 1.4746 * Cr;
    float G = Y - 0.164553 * Cb - 0.571353 * Cr;
    float B = Y + 1.8814 * Cb;
    return vec3(R, G, B);
}

// PQ (ST.2084) EOTF: code -> linear relative luminance (0..1 relative to 10,000 nits)
float pq_to_linear(float x) {
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    x = clamp(x, 1e-6, 1.0);
    float xp = pow(x, 1.0 / m2);
    float num = max(xp - c1, 0.0);
    float den = c2 - c3 * xp;
    return pow(num / max(den, 1e-6), 1.0 / m1);
}

// PQ (ST.2084) OETF: linear relative luminance -> PQ code (0..1)
float linear_to_pq(float L) {
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    // Clamp to a sensible HDR range relative to 10,000 nits.
    L = clamp(L, 0.0, 1.0);
    float Lm1 = pow(L, m1);
    float num = c1 + c2 * Lm1;
    float den = 1.0 + c3 * Lm1;
    return pow(num / max(den, 1e-6), m2);
}

void main() {
    ivec2 coord = ivec2(vUV * vec2(pc.width, pc.height));
    coord = clamp(coord, ivec2(0), ivec2(pc.width - 1, pc.height - 1));

    int idxY = coord.y * pc.width + coord.x;
    int idxUV = (coord.y / 2) * (pc.width / 2) + (coord.x / 2);

    uint yRaw = yData[idxY];
    uint uvRaw = uvData[idxUV];

    // Recover 10-bit code from stored 16-bit P010 sample (bits 6..15).
    uint y10 = (yRaw & 0xFFFFu) >> 6;
    float Ycode = float(y10);
    // HDR10 limited range for luma: Y [64, 940] in 10-bit
    float Yn = clamp((Ycode - 64.0) / (940.0 - 64.0), 0.0, 1.0);
    // Decode PQ to linear relative luminance.
    float Ylin = pq_to_linear(Yn);

    // Chroma: use full 16-bit planes, center around 0.
    uint uRaw = (uvRaw >> 16) & 0xFFFFu;
    uint vRaw = uvRaw & 0xFFFFu;
    float Cb = (float(uRaw) / 65535.0) - 0.5;
    float Cr = (float(vRaw) / 65535.0) - 0.5;

    // Convert to linear BT.2020 RGB.
    vec3 rgbLinear = yuv_to_rgb_bt2020(Ylin, Cb, Cr);
    rgbLinear = max(rgbLinear, vec3(0.0));

    // Encode linear BT.2020 RGB back to PQ for HDR10 swapchain.
    vec3 rgbPQ = vec3(
        linear_to_pq(rgbLinear.r),
        linear_to_pq(rgbLinear.g),
        linear_to_pq(rgbLinear.b)
    );

    outColor = vec4(rgbPQ, 1.0);
}
