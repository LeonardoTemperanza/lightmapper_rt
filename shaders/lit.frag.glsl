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

layout(location = 0) out vec4 _res_out_loc0_;
layout(location = 3) in vec4 _res_in_loc3_;
layout(location = 2) in centroid vec2 _res_in_loc2_;
layout(location = 0) in vec4 _res_in_loc0_;
layout(location = 1) in centroid vec2 _res_in_loc1_;

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
    uint lightmap_;
    uint lightmap_sampler_;
    bool do_bicubic_sampling_;
    bool sample_lightmap_;
    bool sample_diffuse_;
    bool select_lm_pixel_;
    vec2 selected_lm_pixel_;
};
Data Data_ZERO;
void main();
float fwidth_(float v_);
vec4 cubic(float v_);
vec4 texture_sample_bicubic(uint texture_, uint linear_sampler_, vec2 coords_);
float PI = 3.14159265359;
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
    vec4 normal_vert_ = _res_in_loc0_;
    vec2 uv_ = _res_in_loc1_;
    vec2 lm_uv_ = _res_in_loc2_;
    vec4 world_pos_ = _res_in_loc3_;
    _res_ptr_Data data_ = _res_frag_data_;
    vec4 base_color_ = vec4_ZERO;
    vec4 normal_map_sample_ = vec4_ZERO;
    vec3 world_normal_ = vec3_ZERO;
    vec4 packed_normal_ = vec4_ZERO;
    vec4 metallic_roughness_sample_ = vec4_ZERO;
    vec3 irradiance_ = vec3_ZERO;
    vec4 lm_sample_ = vec4_ZERO;
    vec3 out_ = vec3_ZERO;
    base_color_ = texture_sample(data_._res_.base_color_map_, data_._res_.base_color_map_sampler_, uv_);
    base_color_.a = (((base_color_.a - 0.3) / max(fwidth_(base_color_.a), 0.0001)) + 0.5);
    normal_map_sample_ = texture_sample(data_._res_.normal_map_, data_._res_.normal_map_sampler_, uv_);
    world_normal_ = normalize(normal_vert_.xyz);
    packed_normal_ = vec4(((world_normal_ * 0.5) + 0.5), 1.0);
    metallic_roughness_sample_ = texture_sample(data_._res_.metallic_roughness_map_, data_._res_.metallic_roughness_map_sampler_, uv_);
    if(data_._res_.do_bicubic_sampling_)
    {
        vec4 lm_sample_;
        lm_sample_ = texture_sample_bicubic(data_._res_.lightmap_, data_._res_.lightmap_sampler_, lm_uv_);
        irradiance_ = lm_sample_.rgb;
        if((lm_sample_.a < 0.5))
        {
            _res_out_loc0_ = vec4(0, 0, 1, 1);
        }

    }
    else
    {
        lm_sample_ = texture_sample(data_._res_.lightmap_, data_._res_.lightmap_sampler_, lm_uv_);
        irradiance_ = lm_sample_.rgb;
        if((lm_sample_.a < 0.5))
        {
            _res_out_loc0_ = vec4(0, 0, 1, 1);
        }

lm_sample_ = texture_sample(data_._res_.lightmap_, data_._res_.lightmap_sampler_, lm_uv_);irradiance_ = lm_sample_.rgb;if((lm_sample_.a < 0.5))
        {
            _res_out_loc0_ = vec4(0, 0, 1, 1);
        }
    }

    out_ = vec3(1);
    if((data_._res_.sample_diffuse_ && data_._res_.sample_lightmap_))
    {
        out_ = ((base_color_.rgb / PI) * irradiance_);
    }
    else
    {
if((data_._res_.sample_diffuse_ && (!data_._res_.sample_lightmap_)))
        {
            out_ = base_color_.rgb;
        }
        else
        {
if(((!data_._res_.sample_diffuse_) && data_._res_.sample_lightmap_))
            {
                out_ = irradiance_;
            }
        }
    }

    if(data_._res_.select_lm_pixel_)
    {
        vec2 lm_size_;
        vec2 texel_;
        lm_size_ = texture_size(data_._res_.lightmap_, data_._res_.lightmap_sampler_, 0);
        texel_ = (lm_size_ * lm_uv_);
        if(((((texel_.x >= data_._res_.selected_lm_pixel_.x) && (texel_.x <= (data_._res_.selected_lm_pixel_.x + 1))) && (texel_.y >= data_._res_.selected_lm_pixel_.y)) && (texel_.y <= (data_._res_.selected_lm_pixel_.y + 1))))
        {
            out_ = vec3(1, 0, 1);
        }

    }

    _res_out_loc0_ = vec4(out_, base_color_.a);
}

float fwidth_(float v_)
{
    return (abs(dFdxFine(v_)) + abs(dFdyFine(v_)));
}

vec4 cubic(float v_)
{
    vec4 n_ = vec4_ZERO;
    vec4 s_ = vec4_ZERO;
    float x_ = float_ZERO;
    float y_ = float_ZERO;
    float z_ = float_ZERO;
    float w_ = float_ZERO;
    n_ = (vec4(1.0, 2.0, 3.0, 4.0) - v_);
    s_ = ((n_ * n_) * n_);
    x_ = s_.x;
    y_ = (s_.y - (4.0 * s_.x));
    z_ = ((s_.z - (4.0 * s_.y)) + (6.0 * s_.x));
    w_ = (((6.0 - x_) - y_) - z_);
    return (vec4(x_, y_, z_, w_) * (1.0 / 6.0));
}

vec4 texture_sample_bicubic(uint texture_, uint linear_sampler_, vec2 coords_)
{
    vec2 tex_size_ = vec2_ZERO;
    vec2 inv_tex_size_ = vec2_ZERO;
    vec2 tex_coords_ = vec2_ZERO;
    vec2 fxy_ = vec2_ZERO;
    vec4 xcubic_ = vec4_ZERO;
    vec4 ycubic_ = vec4_ZERO;
    vec4 c_ = vec4_ZERO;
    vec4 s_ = vec4_ZERO;
    vec4 offset_ = vec4_ZERO;
    vec4 sample0_ = vec4_ZERO;
    vec4 sample1_ = vec4_ZERO;
    vec4 sample2_ = vec4_ZERO;
    vec4 sample3_ = vec4_ZERO;
    float sx_ = float_ZERO;
    float sy_ = float_ZERO;
    tex_size_ = texture_size(texture_, linear_sampler_, 0);
    inv_tex_size_ = (1.0 / tex_size_);
    tex_coords_ = ((coords_ * tex_size_) - 0.5);
    fxy_ = fract(tex_coords_);
    tex_coords_ -= fxy_;
    xcubic_ = cubic(fxy_.x);
    ycubic_ = cubic(fxy_.y);
    c_ = (tex_coords_.xxyy + vec2((-0.5), (+1.5)).xyxy);
    s_ = vec4((xcubic_.xz + xcubic_.yw), (ycubic_.xz + ycubic_.yw));
    offset_ = (c_ + (vec4(xcubic_.yw, ycubic_.yw) / s_));
    offset_ *= inv_tex_size_.xxyy;
    sample0_ = texture_sample(texture_, linear_sampler_, offset_.xz);
    sample1_ = texture_sample(texture_, linear_sampler_, offset_.yz);
    sample2_ = texture_sample(texture_, linear_sampler_, offset_.xw);
    sample3_ = texture_sample(texture_, linear_sampler_, offset_.yw);
    sx_ = (s_.x / (s_.x + s_.y));
    sy_ = (s_.z / (s_.z + s_.w));
    return mix(mix(sample3_, sample2_, vec4(sx_)), mix(sample1_, sample0_, vec4(sx_)), vec4(sy_));
}

