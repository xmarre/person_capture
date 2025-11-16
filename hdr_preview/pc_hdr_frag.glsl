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

// NOTE: This is a pragmatic passthrough preview: luma is normalized and a basic
// BT.2020-style matrix is applied. It is not a reference-accurate HDR10 PQ transform yet.
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

    // Raw P010 stores 10-bit codes in the upper bits of 16-bit words; shift down here.
    float yCode = float((yRaw >> 6) & 0x3FFu);
    float uCode = float(((uvRaw >> 16) >> 6) & 0x3FFu);
    float vCode = float((uvRaw >> 6) & 0x3FFu);

    // HDR10 limited range (10-bit):
    //   Y:    [64, 940]
    //   Cb/Cr:[64, 960] with 512 as center
    float Y = clamp((yCode - 64.0) / 876.0, 0.0, 1.0);
    float Cb = clamp((uCode - 512.0) / 896.0, -0.5, 0.5);
    float Cr = clamp((vCode - 512.0) / 896.0, -0.5, 0.5);

    vec3 rgb = yuv_to_rgb_bt2020(Y, Cb, Cr);
    outColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
