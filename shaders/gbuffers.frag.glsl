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
layout(location = 1) out vec4 _res_out_loc1_;
layout(location = 1) in vec3 _res_in_loc1_;
layout(location = 0) in vec3 _res_in_loc0_;
layout(location = 2) in vec2 _res_in_loc2_;

layout(buffer_reference) readonly buffer _res_ptr_void;

struct Output
{
    vec4 world_pos_;
    vec4 world_normal_;
};
Output Output_ZERO;
void main();
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };

layout(set = 0, binding = 0) uniform texture2D _res_textures_[];
layout(set = 1, binding = 0) uniform image2D _res_textures_rw_[];
layout(set = 2, binding = 0) uniform sampler _res_samplers_[];

layout(push_constant, scalar) uniform Push
{
    _res_ptr_void _res_vert_data_;
    _res_ptr_void _res_frag_data_;
    _res_ptr_void _res_indirect_data_;
};

void main()
{
    vec3 world_pos_ = _res_in_loc0_;
    vec3 world_normal_ = _res_in_loc1_;
    vec2 uv_ = _res_in_loc2_;
    Output out_ = Output_ZERO;
    out_.world_pos_ = vec4(1, 0, 0, 1);
    _res_out_loc0_ = out_.world_pos_; _res_out_loc1_ = out_.world_normal_; 
}

