
#version 450

layout(location = 0) in vec3 in_world_pos;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec2 in_lm_uv;
layout(location = 3) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D lightmap;
layout(set = 1, binding = 0) uniform sampler2D base_color;

layout(push_constant) uniform PerObj
{
    mat4 model_to_world;
    mat4 normal_mat;
    mat4 world_to_proj;
    vec2 lm_uv_offset;
    float lm_uv_scale;
    uint bicubic;  // b32
} per_obj;

vec4 cubic(float v)
{
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}

// From: https://stackoverflow.com/questions/13501081/efficient-bicubic-filtering-code-in-glsl
vec4 sample_texture_bicubic(sampler2D image, vec2 texCoords)
{
    vec2 texSize = textureSize(image, 0);
    vec2 invTexSize = 1.0 / texSize;

    texCoords = texCoords * texSize - 0.5;

    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = texCoords.xxyy + vec2 (-0.5, +1.5).xyxy;

    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

    offset *= invTexSize.xxyy;

    vec4 sample0 = texture(image, offset.xz);
    vec4 sample1 = texture(image, offset.yz);
    vec4 sample2 = texture(image, offset.xw);
    vec4 sample3 = texture(image, offset.yw);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(
       mix(sample3, sample2, sx), mix(sample1, sample0, sx)
    , sy);
}

vec4 uv_checkerboard(vec2 uv)
{
    //float checkerSize = 0.05f;
    float checkerSize = 0.005f;
    float total = floor(uv.x / checkerSize) + floor(uv.y / checkerSize);
    bool isEven = mod(total, 2.0) == 0.0;
    return isEven? vec4(0.05f) : vec4(0.9f);
}

void main()
{
    vec3 world_normal = normalize(in_world_normal);

    vec4 lm_sample = vec4(0.0f);
    if(per_obj.bicubic != 0)
        lm_sample = sample_texture_bicubic(lightmap, in_lm_uv);
    else
        lm_sample = texture(lightmap, in_lm_uv);

    vec4 albedo = texture(base_color, in_uv);

    //out_color = vec4(world_normal * 0.5f + 0.5f, 1);
    //out_color = vec4(in_lm_uv, 0.0f, 1.0f);
    //out_color = texture(lightmap, in_lm_uv);
    //out_color = uv_checkerboard(in_lm_uv);
    //out_color = lm_sample;
    //out_color = albedo;
    out_color = lm_sample * albedo;
}
