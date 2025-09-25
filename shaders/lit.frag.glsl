
#version 450

layout(location = 0) in vec3 in_world_pos;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec3 in_uv;
layout(location = 3) in vec3 in_lm_uv;

layout(location = 0) out vec4 out_color;

void main()
{
    vec3 world_normal = normalize(in_world_normal);
    out_color = vec4(normalize(world_normal) * 0.5f + 0.5f, 1);
}
