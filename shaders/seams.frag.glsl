
#version 450

layout(location = 0) in flat vec2 in_uv_0;
layout(location = 1) in flat vec2 in_uv_1;
layout(location = 2) in float in_t;

layout(location = 0) out vec4 out_color;

layout(set = 1, binding = 0) uniform sampler2D src_image;

void main()
{
    vec2 in_uv = mix(in_uv_0, in_uv_1, in_t);

    out_color = vec4(texture(src_image, in_uv).rgb, 0.5f);
    //float step = 0.0001f;
    //out_color = vec4(mod(in_uv, step) / step, 1.0f, 0.5f);
    //out_color = vec4(vec3(in_t), 1.0f);
    //out_color = vec4(0.0f, 1.0f, 0, 0.5f);
}
