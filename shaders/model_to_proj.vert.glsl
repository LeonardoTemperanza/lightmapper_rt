
#version 450

layout(push_constant) uniform PerObj
{
    mat4 model_to_world;
    mat4 normal_mat;
    mat4 world_to_view;
    mat4 view_to_proj;
} per_obj;

/*
layout(set = 0, binding = 0) buffer PerView
{
    mat4 world_to_view;
    mat4 view_to_proj;
} per_view;
*/

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_uv;
layout(location = 3) in vec3 in_lm_uv;

layout(location = 0) out vec3 out_world_pos;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec3 out_uv;
layout(location = 3) out vec3 out_lm_uv;

void main()
{
    vec4 world_pos = per_obj.model_to_world * vec4(in_pos, 1.0f);
    vec4 view_pos  = per_obj.world_to_view * world_pos;
    vec4 proj_pos  = per_obj.view_to_proj * view_pos;
    proj_pos.y *= -1.0f;

    out_world_pos = in_pos;
    out_world_normal = mat3(per_obj.normal_mat) * in_normal;
    out_uv = in_uv;
    out_lm_uv = in_lm_uv;

    gl_Position = proj_pos;
}
