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

layout(location = 1) out vec3 _res_out_loc1_;
layout(location = 0) out vec3 _res_out_loc0_;
layout(location = 2) out vec2 _res_out_loc2_;

layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_slice_vec3;
layout(buffer_reference) readonly buffer _res_slice_vec2;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Output
{
    vec4 out_pos;
    vec3 world_pos;
    vec3 world_normal;
    vec2 uv;
};
Output Output_ZERO;
struct Data
{
    _res_slice_vec3 pos;
    _res_slice_vec3 normals;
    _res_slice_vec2 uvs;
    _res_slice_vec2 lightmap_uvs;
    mat4 model_to_world;
    mat4 model_to_world_normals;
};
Data Data_ZERO;
void main();
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_slice_vec3 { vec3 _res_[]; };
_res_slice_vec3 _res_slice_vec3_ZERO;
layout(buffer_reference, scalar) readonly buffer _res_slice_vec2 { vec2 _res_[]; };
_res_slice_vec2 _res_slice_vec2_ZERO;
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
    uint vert_id_ = gl_VertexIndex;
    _res_ptr_Data data_ = _res_vert_data_;
    Output vert_out_ = Output_ZERO;
    vert_out_.out_pos = vec4(data_._res_.lightmap_uvs._res_[vert_id_], 0, 1);
    vert_out_.world_pos = (data_._res_.model_to_world * vec4(data_._res_.pos._res_[vert_id_], 1)).xyz;
    vert_out_.world_normal = (data_._res_.model_to_world_normals * vec4(data_._res_.normals._res_[vert_id_], 0)).xyz;
    vert_out_.uv = data_._res_.uvs._res_[vert_id_];
    gl_Position = vert_out_.out_pos; _res_out_loc0_ = vert_out_.world_pos; _res_out_loc1_ = vert_out_.world_normal; _res_out_loc2_ = vert_out_.uv; 
}

