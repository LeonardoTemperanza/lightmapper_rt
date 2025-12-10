
#version 450

layout(push_constant) uniform PerObj
{
    mat4 model_to_world;
    mat4 normal_mat;
    mat4 world_to_proj;
    vec2 lm_uv_offset;
    float lm_uv_scale;
    uint bicubic;  // b32
} per_obj;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_lm_uv;
layout(location = 3) in vec2 in_uv;

layout(location = 0) out vec3 out_world_pos;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec2 out_lm_uv;
layout(location = 3) out vec2 out_uv;

void main()
{
    vec4 world_pos = per_obj.model_to_world * vec4(in_pos, 1.0f);
    vec4 proj_pos  = per_obj.world_to_proj * world_pos;
    proj_pos.y *= -1.0f;

    out_world_pos = in_pos;
    out_world_normal = mat3(per_obj.normal_mat) * in_normal;
    out_lm_uv = per_obj.lm_uv_scale * in_lm_uv + per_obj.lm_uv_offset;
    out_uv = in_uv;

    gl_Position = proj_pos;
}
