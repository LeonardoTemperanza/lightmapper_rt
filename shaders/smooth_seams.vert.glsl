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

#define texture_sample(t, s, uv)       texture(sampler2D(_res_textures_[nonuniformEXT(t)], _res_samplers_[nonuniformEXT(s)]), uv)
#define texture_load(t, coord)         imageLoad(_res_textures_rw_[nonuniformEXT(t)], ivec2(coord))
#define texture_store(t, coord, value) imageStore(_res_textures_rw_[nonuniformEXT(t)], ivec2(coord), value)
#define texture_size(t, s, lod)        textureSize(sampler2D(_res_textures_[nonuniformEXT(t)], _res_samplers_[nonuniformEXT(s)]), lod)

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

layout(location = 0) out vec2 _res_out_loc0_;

layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_slice_vec2;
layout(buffer_reference) readonly buffer _res_slice_Seam;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Output
{
    vec4 pos_;
    vec2 to_sample_;
};
Output Output_ZERO;
struct Data
{
    _res_slice_vec2 lm_uvs_;
    _res_slice_Seam seams_;
    vec2 resolution_;
    bool a_to_b_;
    uint _res_padding_0;
};
Data Data_ZERO;
struct Seam
{
    uint line_a_0_;
    uint line_a_1_;
    uint line_b_0_;
    uint line_b_1_;
};
Seam Seam_ZERO;
void main();
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_slice_vec2 { vec2 _res_[]; };
_res_slice_vec2 _res_slice_vec2_ZERO;
layout(buffer_reference, scalar) readonly buffer _res_slice_Seam { Seam _res_[]; };
_res_slice_Seam _res_slice_Seam_ZERO;
layout(buffer_reference, scalar) readonly buffer _res_ptr_Data { Data _res_; };
_res_ptr_Data _res_ptr_Data_ZERO;
struct _res_array_6_int { int data[6]; };
_res_array_6_int _res_array_6_int_ZERO;

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
    uint segment_ = uint_ZERO;
    uint local_ = uint_ZERO;
    vec2 p0_ = vec2_ZERO;
    vec2 p1_ = vec2_ZERO;
    vec2 other_p0_ = vec2_ZERO;
    vec2 other_p1_ = vec2_ZERO;
    vec2 dir_ = vec2_ZERO;
    float thickness_ = float_ZERO;
    vec2 normal_ = vec2_ZERO;
    _res_array_6_int corner_x_ = _res_array_6_int_ZERO;
    _res_array_6_int corner_y_ = _res_array_6_int_ZERO;
    vec2 cap_ = vec2_ZERO;
    vec2 base_ = vec2_ZERO;
    float side_ = float_ZERO;
    vec2 pos_ = vec2_ZERO;
    Output out_ = Output_ZERO;
    segment_ = (vert_id_ / 6);
    local_ = (vert_id_ % 6);
    if(data_._res_.a_to_b_)
    {
        p0_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_a_0_];
        other_p0_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_b_0_];
        p1_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_a_1_];
        other_p1_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_b_1_];
    }
    else
    {
        p0_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_b_0_];
        other_p0_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_a_0_];
        p1_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_b_1_];
        other_p1_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_a_1_];
p0_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_b_0_];other_p0_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_a_0_];p1_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_b_1_];other_p1_ = data_._res_.lm_uvs_._res_[data_._res_.seams_._res_[segment_].line_a_1_];    }

    dir_ = normalize((p1_ - p0_));
    thickness_ = 1.5;
    normal_ = (vec2((-dir_.y), dir_.x) * (thickness_ / data_._res_.resolution_));
    corner_x_.data[0] = 0;
    corner_x_.data[1] = 0;
    corner_x_.data[2] = 1;
    corner_x_.data[3] = 1;
    corner_x_.data[4] = 1;
    corner_x_.data[5] = 0;
    corner_y_.data[0] = 0;
    corner_y_.data[1] = 1;
    corner_y_.data[2] = 0;
    corner_y_.data[3] = 1;
    corner_y_.data[4] = 0;
    corner_y_.data[5] = 1;
    cap_ = (dir_ * (0.5 / data_._res_.resolution_));
    base_ = mix((p0_ - cap_), (p1_ + cap_), vec2(float(corner_x_.data[local_])));
    side_ = (((corner_y_.data[local_] == 0)) ? ((-1.0)) : (1.0));
    pos_ = (base_ + (side_ * normal_));
    pos_ = ((pos_ * 2) - vec2(1));
    out_.pos_ = vec4(pos_, 0.0, 1.0);
    out_.to_sample_ = (((corner_x_.data[local_] == 0)) ? (other_p0_) : (other_p1_));
    gl_Position = out_.pos_; _res_out_loc0_ = out_.to_sample_; 
}

