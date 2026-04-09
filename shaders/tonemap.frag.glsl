#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_debug_printf : require
layout(set = 0, binding = 0) uniform texture2D _res_textures_[];
layout(set = 1, binding = 0) uniform image2D _res_textures_rw_[];
layout(set = 2, binding = 0) uniform sampler _res_samplers_[];


// Intrinsics:

vec2 texture_size(uint t, uint s, int lod)
{
   return textureSize(sampler2D(_res_textures_[nonuniformEXT(t)], _res_samplers_[nonuniformEXT(s)]), lod);
}

// Intrinsics end.

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
layout(location = 0) in vec2 _res_in_loc0_;

layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Data
{
    uint texture_id_;
    uint sampler_id_;
};
Data Data_ZERO;
void main();
vec4 linear_to_srgb(vec4 color_);
vec3 filmic(vec3 x_);
vec4 hdr_to_ldr(vec4 color_);
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_ptr_Data { Data _res_; };
_res_ptr_Data _res_ptr_Data_ZERO;

layout(push_constant, scalar) uniform Push
{
    _res_ptr_Data _res_vert_data_;
    _res_ptr_Data _res_frag_data_;
    _res_ptr_void _res_indirect_data_;
};

void main()
{
    vec2 uv_ = _res_in_loc0_;
    _res_ptr_Data data_ = _res_frag_data_;
    vec4 linear_ = vec4_ZERO;
    linear_ = texture(sampler2D(_res_textures_[nonuniformEXT(data_._res_.texture_id_)], _res_samplers_[nonuniformEXT(data_._res_.sampler_id_)]), uv_);
    _res_out_loc0_ = linear_to_srgb(hdr_to_ldr(max(vec4(0, 0, 0, 0), linear_)));
}

vec4 linear_to_srgb(vec4 color_)
{
    vec3 cutoff_ = vec3_ZERO;
    vec3 higher_ = vec3_ZERO;
    vec3 lower_ = vec3_ZERO;
    cutoff_ = vec3(float((color_.x < 0.0031308)), float((color_.y < 0.0031308)), float((color_.z < 0.0031308)));
    higher_ = ((vec3(1.055) * pow(color_.xyz, vec3((1.0 / 2.4)))) - vec3(0.055));
    lower_ = (color_.xyz * vec3(12.92));
    return vec4(mix(higher_, lower_, cutoff_), color_.w);
}

vec3 filmic(vec3 x_)
{
    vec3 X_ = vec3_ZERO;
    vec3 result_ = vec3_ZERO;
    X_ = max(vec3(0.0), (x_ - 0.004));
    result_ = ((X_ * ((6.2 * X_) + 0.5)) / ((X_ * ((6.2 * X_) + 1.7)) + 0.06));
    return pow(result_, vec3(2.2));
}

vec4 hdr_to_ldr(vec4 color_)
{
    return vec4(filmic(color_.xyz), color_.w);
}

