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
layout(location = 1) out vec2 _res_out_loc1_;
layout(location = 2) out vec2 _res_out_loc2_;
layout(location = 3) out vec4 _res_out_loc3_;

layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_slice_vec3;
layout(buffer_reference) readonly buffer _res_slice_vec2;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Data
{
    _res_slice_vec3 positions_;
    _res_slice_vec3 normals_;
    _res_slice_vec2 uvs_;
    _res_slice_vec2 lm_uvs_;
    mat4 model_to_world_;
    mat4 model_to_world_normal_;
    mat4 world_to_view_;
    mat4 view_to_proj_;
};
Data Data_ZERO;
struct Output
{
    vec4 pos_;
    vec4 normal_;
    vec2 uv_;
    vec2 lm_uv_;
    vec4 world_pos_;
};
Output Output_ZERO;
void main();
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_slice_vec3 { vec3 _res_[]; };
_res_slice_vec3 _res_slice_vec3_ZERO;
layout(buffer_reference, scalar) readonly buffer _res_slice_vec2 { vec2 _res_[]; };
_res_slice_vec2 _res_slice_vec2_ZERO;
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
    uint vert_id_ = gl_VertexIndex;
    _res_ptr_Data data_ = _res_vert_data_;
    vec4 clip_pos_ = vec4_ZERO;
    vec4 world_pos_ = vec4_ZERO;
    vec4 world_normal_ = vec4_ZERO;
    Output vert_out_ = Output_ZERO;
    clip_pos_ = vec4(data_._res_.positions_._res_[vert_id_].xyz, 1.0);
    world_pos_ = (data_._res_.model_to_world_ * clip_pos_);
    clip_pos_ = (data_._res_.world_to_view_ * world_pos_);
    clip_pos_ = (data_._res_.view_to_proj_ * clip_pos_);
    clip_pos_.y = (0.0 - clip_pos_.y);
    world_normal_ = (data_._res_.model_to_world_normal_ * vec4(data_._res_.normals_._res_[vert_id_], 1));
    vert_out_.pos_ = clip_pos_;
    vert_out_.normal_ = world_normal_;
    vert_out_.uv_ = data_._res_.uvs_._res_[vert_id_];
    vert_out_.lm_uv_ = data_._res_.lm_uvs_._res_[vert_id_];
    vert_out_.world_pos_ = world_pos_;
    gl_Position = vert_out_.pos_; _res_out_loc0_ = vert_out_.normal_; _res_out_loc1_ = vert_out_.uv_; _res_out_loc2_ = vert_out_.lm_uv_; _res_out_loc3_ = vert_out_.world_pos_; 
}

