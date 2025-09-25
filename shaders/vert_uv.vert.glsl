
#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_uv;
layout(location = 3) in vec3 in_lm_uv;

layout(location = 0) out vec3 out_world_pos;
layout(location = 1) out vec3 out_world_normal;

void main()
{
    vec4 world_pos = push.model_to_world * vec4(in_pos, 1.0f);

    out_world_pos = world_pos;
    out_normal = in_normal;

    gl_Position = in_lm_uv;
}
