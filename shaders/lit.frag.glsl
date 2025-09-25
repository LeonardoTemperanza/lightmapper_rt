
#version 450

layout(location = 0) in vec3 in_world_pos;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec2 in_lm_uv;

layout(location = 0) out vec4 out_color;

void main()
{
    vec3 world_normal = normalize(in_world_normal);
    //out_color = vec4(world_normal * 0.5f + 0.5f, 1);
    out_color = vec4(in_lm_uv, 0.0f, 1.0f);
}
