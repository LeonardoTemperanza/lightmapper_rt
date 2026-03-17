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
uint textureid_ZERO;
uint samplerid_ZERO;
uint bvh_id_ZERO;

layout(location = 1) out vec2 _res_out_loc1_;
layout(location = 0) out vec4 _res_out_loc0_;

layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_slice_Im_Draw_Vert;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Im_Draw_Vert
{
    vec2 pos;
    vec2 uv;
    uint col;
};
Im_Draw_Vert Im_Draw_Vert_ZERO;
struct Data
{
    _res_slice_Im_Draw_Vert verts;
    uint vert_offset;
    vec2 scale;
    vec2 translate;
};
Data Data_ZERO;
struct Output
{
    vec4 pos;
    vec4 color;
    vec2 uv;
};
Output Output_ZERO;
void main();
vec4 uint_to_rgba8(uint v);
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_slice_Im_Draw_Vert { Im_Draw_Vert _res_[]; };
_res_slice_Im_Draw_Vert _res_slice_Im_Draw_Vert_ZERO;
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
    _res_ptr_Data data_ = _res_vert_data_;
    uint vert_id_ = gl_VertexIndex;
    uint vert_id_global_ = uint_ZERO;
    Im_Draw_Vert vert_ = Im_Draw_Vert_ZERO;
    Output out_ = Output_ZERO;
    vert_id_global_ = (vert_id_ + data_._res_.vert_offset);
    vert_ = data_._res_.verts._res_[vert_id_global_];
    out_.color = uint_to_rgba8(vert_.col);
    out_.pos = vec4(((vert_.pos * data_._res_.scale) + data_._res_.translate), 0, 1);
    out_.uv = vert_.uv;
    gl_Position = out_.pos; _res_out_loc0_ = out_.color; _res_out_loc1_ = out_.uv; 
}

vec4 uint_to_rgba8(uint v_)
{
    return (vec4(float((v_ & 0xFF)), float(((v_ >> 8) & 0xFF)), float(((v_ >> 16) & 0xFF)), float(((v_ >> 24) & 0xFF))) / 255.0);
}

