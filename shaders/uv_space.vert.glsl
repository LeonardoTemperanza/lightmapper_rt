
#version 450

layout(push_constant) uniform PerObj
{
    mat4 model_to_world;
    mat4 normal_mat;
    vec2 lm_uv_offset;
    float lm_uv_scale;
} per_obj;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_lm_uv;

layout(location = 0) out vec3 out_world_pos;
layout(location = 1) out vec3 out_world_normal;

void main()
{
    vec4 world_pos = per_obj.model_to_world * vec4(in_pos, 1.0f);
    vec3 world_normal = mat3(per_obj.normal_mat) * in_normal;

    //vec2 trans_lm_uv = in_lm_uv * per_obj.lm_uv_scale + per_obj.lm_uv_offset;
    vec2 trans_lm_uv = in_lm_uv;

    out_world_pos = world_pos.xyz;
    out_world_normal = world_normal;
    gl_Position = vec4(trans_lm_uv * 2.0f - 1.0f, 0.0f, 1.0f);
}
