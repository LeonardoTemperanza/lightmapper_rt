#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_debug_printf : require
bool bool_ZERO;
int int_ZERO;
uint uint_ZERO;
float float_ZERO;
vec2 vec2_ZERO;
vec3 vec3_ZERO;
vec4 vec4_ZERO;
mat4 mat4_ZERO;
uint texture_id_ZERO;
uint sampler_id_ZERO;
uint bvh_id_ZERO;

layout(location = 0) out vec4 _res_out_loc0_;
layout(location = 1) in vec2 _res_in_loc1_;
layout(location = 0) in vec4 _res_in_loc0_;
layout(location = 2) in vec4 _res_in_loc2_;

layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Data
{
    uint base_color_map_;
    uint base_color_map_sampler_;
    uint metallic_roughness_map_;
    uint metallic_roughness_map_sampler_;
    uint normal_map_;
    uint normal_map_sampler_;
};
Data Data_ZERO;
void main();
float fwidth_(float v_);
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_ptr_Data { Data _res_; };
_res_ptr_Data _res_ptr_Data_ZERO;

layout(set = 0, binding = 0) uniform texture2D _res_textures_[];
layout(set = 1, binding = 0) uniform image2D _res_textures_rw_[];
layout(set = 2, binding = 0) uniform sampler _res_samplers_[];

layout(push_constant, scalar) uniform Push
{
    _res_ptr_Data _res_vert_data_;
    _res_ptr_Data _res_frag_data_;
    _res_ptr_void _res_indirect_data_;
};

void main()
{
    vec4 normal_vert_ = _res_in_loc0_;
    vec2 uv_ = _res_in_loc1_;
    vec4 world_pos_ = _res_in_loc2_;
    _res_ptr_Data data_ = _res_frag_data_;
    vec4 base_color_ = vec4_ZERO;
    vec4 normal_map_sample_ = vec4_ZERO;
    vec3 world_normal_ = vec3_ZERO;
    vec4 packed_normal_ = vec4_ZERO;
    vec4 metallic_roughness_sample_ = vec4_ZERO;
    vec4 albedo_ = vec4_ZERO;
    base_color_ = texture(sampler2D(_res_textures_[nonuniformEXT(data_._res_.base_color_map_)], _res_samplers_[nonuniformEXT(data_._res_.base_color_map_sampler_)]), uv_);
    base_color_.a = (((base_color_.a - 0.3) / max(fwidth_(base_color_.a), 0.0001)) + 0.5);
    normal_map_sample_ = texture(sampler2D(_res_textures_[nonuniformEXT(data_._res_.normal_map_)], _res_samplers_[nonuniformEXT(data_._res_.normal_map_sampler_)]), uv_);
    world_normal_ = normalize(normal_vert_.xyz);
    packed_normal_ = vec4(((world_normal_ * 0.5) + 0.5), 1.0);
    metallic_roughness_sample_ = texture(sampler2D(_res_textures_[nonuniformEXT(data_._res_.metallic_roughness_map_)], _res_samplers_[nonuniformEXT(data_._res_.metallic_roughness_map_sampler_)]), uv_);
    albedo_ = base_color_;
    _res_out_loc0_ = albedo_;
}

float fwidth_(float v_)
{
    return (abs(dFdxFine(v_)) + abs(dFdyFine(v_)));
}

