#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_debug_printf : require
#extension GL_EXT_ray_query : require
layout(local_size_x_id = 13370, local_size_y_id = 13371, local_size_z_id = 13372) in;

// Raytracing intrinsics:

layout(set = 3, binding = 0) uniform accelerationStructureEXT _res_bvhs_[];

mat4 _res_mat4_from_mat4x3(mat4x3 m)
{
    // GLSL is column-major: m[col][row]
    return mat4(
        vec4(m[0], 0.0),
        vec4(m[1], 0.0),
        vec4(m[2], 0.0),
        vec4(m[3], 1.0)
    );
}

struct Ray_Desc
{
    uint flags;
    uint cull_mask;
    float t_min;
    float t_max;
    vec3 origin;
    vec3 dir;
};
Ray_Desc Ray_Desc_ZERO;

struct Ray_Result
{
    uint kind;
    float t;
    uint instance_idx;
    uint primitive_idx;
    vec2 barycentrics;
    bool front_face;
    mat4 object_to_world;
    mat4 world_to_object;
};
Ray_Result Ray_Result_ZERO;

Ray_Result rayquery_result(rayQueryEXT rq)
{
    Ray_Result res;
    res.kind = rayQueryGetIntersectionTypeEXT(rq, true);
    res.t = rayQueryGetIntersectionTEXT(rq, true);
    res.instance_idx  = rayQueryGetIntersectionInstanceIdEXT(rq, true);
    res.primitive_idx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    res.front_face    = rayQueryGetIntersectionFrontFaceEXT(rq, true);
    res.object_to_world = _res_mat4_from_mat4x3(rayQueryGetIntersectionObjectToWorldEXT(rq, true));
    res.world_to_object = _res_mat4_from_mat4x3(rayQueryGetIntersectionWorldToObjectEXT(rq, true));
    res.barycentrics    = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    return res;
}

Ray_Result rayquery_candidate(rayQueryEXT rq)
{
    Ray_Result res;
    res.kind = rayQueryGetIntersectionTypeEXT(rq, false);
    res.t = rayQueryGetIntersectionTEXT(rq, false);
    res.instance_idx  = rayQueryGetIntersectionInstanceIdEXT(rq, false);
    res.primitive_idx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
    res.front_face    = rayQueryGetIntersectionFrontFaceEXT(rq, false);
    res.object_to_world = _res_mat4_from_mat4x3(rayQueryGetIntersectionObjectToWorldEXT(rq, false));
    res.world_to_object = _res_mat4_from_mat4x3(rayQueryGetIntersectionWorldToObjectEXT(rq, false));
    res.barycentrics    = rayQueryGetIntersectionBarycentricsEXT(rq, false);
    return res;
}

void rayquery_init(rayQueryEXT rq, Ray_Desc desc, uint bvh)
{
    rayQueryInitializeEXT(rq,
                          _res_bvhs_[nonuniformEXT(bvh)],
                          desc.flags,
                          desc.cull_mask,
                          desc.origin,
                          desc.t_min,
                          desc.dir,
                          desc.t_max);
}

bool rayquery_proceed(rayQueryEXT rq)
{
    return rayQueryProceedEXT(rq);
}

// Raytracing intrinsics end.

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


layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_slice_Instance;
layout(buffer_reference) readonly buffer _res_slice_Mesh;
layout(buffer_reference) readonly buffer _res_slice_vec4;
layout(buffer_reference) readonly buffer _res_slice_vec2;
layout(buffer_reference) readonly buffer _res_slice_uint;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Scene
{
    _res_slice_Instance instances;
    _res_slice_Mesh meshes;
};
Scene Scene_ZERO;
struct Mesh
{
    _res_slice_vec4 pos;
    _res_slice_vec4 normal;
    _res_slice_vec2 uvs;
    _res_slice_uint indices;
};
Mesh Mesh_ZERO;
struct Instance
{
    uint mesh_idx;
};
Instance Instance_ZERO;
struct Data
{
    uint output_texture_id;
    uint tlas_id;
    Scene scene;
    vec2 resolution;
    uint accum_counter;
    mat4 camera_to_world;
};
Data Data_ZERO;
struct Ray
{
    vec3 ori;
    vec3 dir;
};
Ray Ray_ZERO;
struct Hit_Info
{
    bool hit;
    float t;
    vec3 normal;
    uint mesh_idx;
    uint tri_idx;
    vec2 uv;
};
Hit_Info Hit_Info_ZERO;
struct Material_Point
{
    vec3 color;
};
Material_Point Material_Point_ZERO;
void main();
vec3 pathtrace(Ray start_ray, Scene scene, uint tlas_id);
Hit_Info ray_scene_intersection(Ray ray, Scene scene, uint tlas_id);
uint hash_u32(uint seed);
void init_rng(uint global_id, uint accum_counter);
uint random_u32();
float random_f32();
vec2 random_vec2();
float copysignf(float mag, float sgn);
mat4 basis_fromz(vec3 v);
vec3 sample_hemisphere_cos(vec3 normal, vec2 ruv);
float sample_hemisphere_cos_pdf(vec3 normal, vec3 direction);
Material_Point get_material_point(Scene scene, Hit_Info hit);
vec3 sample_matte(vec3 color, vec3 normal, vec3 outgoing, vec2 rn);
vec3 eval_matte(vec3 color, vec3 normal, vec3 outgoing, vec3 incoming);
float sample_matte_pdf(vec3 color, vec3 normal, vec3 outgoing, vec3 incoming);
bool is_finite(vec3 v);
float pi = 3.1415;
uint RNG_STATE;
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_slice_Instance { Instance _res_[]; };
_res_slice_Instance _res_slice_Instance_ZERO;
layout(buffer_reference, scalar) readonly buffer _res_slice_Mesh { Mesh _res_[]; };
_res_slice_Mesh _res_slice_Mesh_ZERO;
layout(buffer_reference, scalar) readonly buffer _res_slice_vec4 { vec4 _res_[]; };
_res_slice_vec4 _res_slice_vec4_ZERO;
layout(buffer_reference, scalar) readonly buffer _res_slice_vec2 { vec2 _res_[]; };
_res_slice_vec2 _res_slice_vec2_ZERO;
layout(buffer_reference, scalar) readonly buffer _res_slice_uint { uint _res_[]; };
_res_slice_uint _res_slice_uint_ZERO;
layout(buffer_reference, scalar) readonly buffer _res_ptr_Data { Data _res_; };
_res_ptr_Data _res_ptr_Data_ZERO;

layout(set = 0, binding = 0) uniform texture2D _res_textures_[];
layout(set = 1, binding = 0) uniform image2D _res_textures_rw_[];
layout(set = 2, binding = 0) uniform sampler _res_samplers_[];

layout(push_constant, scalar) uniform Push
{
    _res_ptr_Data _res_compute_data_;
};

void main()
{
    _res_ptr_Data data_ = _res_compute_data_;
    vec3 global_invocation_id_ = gl_GlobalInvocationID;
    vec2 uv_ = vec2_ZERO;
    vec2 coord_ = vec2_ZERO;
    vec3 world_camera_pos_ = vec3_ZERO;
    vec3 camera_lookat_ = vec3_ZERO;
    vec3 world_camera_lookat_ = vec3_ZERO;
    Ray camera_ray_ = Ray_ZERO;
    vec3 color_ = vec3_ZERO;
    init_rng(uint(((global_invocation_id_.y * data_._res_.resolution.x) + global_invocation_id_.x)), data_._res_.accum_counter);
    uv_ = (global_invocation_id_.xy / data_._res_.resolution);
    coord_ = ((2.0 * uv_) - 1.0);
    coord_ = (coord_ * tan((((90.0 * pi) / 180.0) / 2.0)));
    coord_.y = ((coord_.y * data_._res_.resolution.y) / data_._res_.resolution.x);
    world_camera_pos_ = (data_._res_.camera_to_world * vec4(0, 0, 0, 1)).xyz;
    camera_lookat_ = normalize(vec3(coord_, 1));
    world_camera_lookat_ = normalize((data_._res_.camera_to_world * vec4(camera_lookat_, 0.0))).xyz;
    camera_ray_.ori = world_camera_pos_;
    camera_ray_.dir = world_camera_lookat_;
    color_ = pathtrace(camera_ray_, data_._res_.scene, data_._res_.tlas_id);
    if(((global_invocation_id_.x < data_._res_.resolution.x) && (global_invocation_id_.y < data_._res_.resolution.y)))
    {
        vec2 output_pixel_;
        output_pixel_ = vec2(global_invocation_id_.x, (data_._res_.resolution.y - global_invocation_id_.y));
        if((data_._res_.accum_counter > 1))
        {
            float weight_;
            vec3 prev_color_;
            weight_ = (1.0 / float(data_._res_.accum_counter));
            prev_color_ = imageLoad(_res_textures_rw_[nonuniformEXT(data_._res_.output_texture_id)], ivec2(output_pixel_)).xyz;
            color_ = ((prev_color_ * (1 - weight_)) + (color_ * weight_));
            color_ = max(color_, vec3(0, 0, 0));
        }

        imageStore(_res_textures_rw_[nonuniformEXT(data_._res_.output_texture_id)], ivec2(output_pixel_), vec4(color_, 1));
    }

}

vec3 pathtrace(Ray start_ray_, Scene scene_, uint tlas_id_)
{
    vec3 radiance_ = vec3_ZERO;
    vec3 weight_ = vec3_ZERO;
    Ray ray_ = Ray_ZERO;
    int max_bounces_ = int_ZERO;
    radiance_ = vec3(0, 0, 0);
    weight_ = vec3(1, 1, 1);
    ray_ = start_ray_;
    max_bounces_ = 5;
    // for construct
    {
        int bounce_;
        Hit_Info hit_;
        Material_Point mat_point_;
        vec3 outgoing_;
        vec3 incoming_;
        vec2 rnd_;
        float prob_;
        for(bounce_ = 0; (bounce_ <= max_bounces_); bounce_ = (bounce_ + 1))
        {
            hit_ = ray_scene_intersection(ray_, scene_, tlas_id_);
            if((!hit_.hit))
            {
                vec2 coords_;
                vec3 emission_;
                coords_ = vec2((atan(ray_.dir.x, ray_.dir.z) / (2.0 * 3.1415)), (acos(clamp(ray_.dir.y, (-1.0), 1.0)) / 3.1415));
                emission_ = (mix(vec3(0.8, 0.7, 0.1), vec3(0.1, 0.2, 0.8), vec3(coords_.y)) * 5.0);
                radiance_ += (emission_ * weight_);
                break;
            }

            mat_point_ = get_material_point(scene_, hit_);
            outgoing_ = (-ray_.dir);
            incoming_ = vec3(0, 0, 0);
            rnd_ = random_vec2();
            incoming_ = sample_matte(mat_point_.color, hit_.normal, outgoing_, rnd_);
            if((incoming_ == vec3(0, 0, 0)))
            {
                break;
            }

            prob_ = sample_matte_pdf(mat_point_.color, hit_.normal, outgoing_, incoming_);
            weight_ *= (eval_matte(mat_point_.color, hit_.normal, outgoing_, incoming_) / prob_);
            ray_.ori = (ray_.ori + (ray_.dir * hit_.t));
            ray_.dir = incoming_;
            if(((weight_ == vec3(0, 0, 0)) || (!is_finite(weight_))))
            {
                break;
            }

            if((bounce_ > 3))
            {
                float survive_prob_;
                survive_prob_ = min(0.99, max(weight_.x, max(weight_.y, weight_.z)));
                if((random_f32() >= survive_prob_))
                {
                    break;
                }

                weight_ = (weight_ / survive_prob_);
            }

        }
    }

    return radiance_;
}

Hit_Info ray_scene_intersection(Ray ray_, Scene scene_, uint tlas_id_)
{
    uint Ray_Flags_Opaque_ = uint_ZERO;
    uint Ray_Flags_Terminate_On_First_Hit_ = uint_ZERO;
    uint Ray_Flags_Skip_Closest_Hit_Shader_ = uint_ZERO;
    uint Ray_Result_Kind_Miss_ = uint_ZERO;
    uint Ray_Result_Kind_Hit_Mesh_ = uint_ZERO;
    uint Ray_Result_Kind_Hit_AABB_ = uint_ZERO;
    Hit_Info hit_info_ = Hit_Info_ZERO;
    Ray_Desc desc_ = Ray_Desc_ZERO;
    rayQueryEXT rq_;
    Ray_Result hit_ = Ray_Result_ZERO;
    Instance instance_ = Instance_ZERO;
    Mesh mesh_ = Mesh_ZERO;
    _res_slice_uint indices_ = _res_slice_uint_ZERO;
    uint base_idx_ = uint_ZERO;
    float w_ = float_ZERO;
    vec4 n0_ = vec4_ZERO;
    vec4 n1_ = vec4_ZERO;
    vec4 n2_ = vec4_ZERO;
    vec4 normal_ = vec4_ZERO;
    vec4 world_normal_ = vec4_ZERO;
    vec2 bary_ = vec2_ZERO;
    Ray_Flags_Opaque_ = 1;
    Ray_Flags_Terminate_On_First_Hit_ = 4;
    Ray_Flags_Skip_Closest_Hit_Shader_ = 8;
    Ray_Result_Kind_Miss_ = 0;
    Ray_Result_Kind_Hit_Mesh_ = 1;
    Ray_Result_Kind_Hit_AABB_ = 2;
    desc_.flags = Ray_Flags_Opaque_;
    desc_.cull_mask = 0xFF;
    desc_.t_min = 0.001;
    desc_.t_max = 1000000000.0;
    desc_.origin = ray_.ori;
    desc_.dir = ray_.dir;
    rayquery_init(rq_, desc_, tlas_id_);
    rayquery_proceed(rq_);
    hit_ = rayquery_result(rq_);
    if((hit_.kind != Ray_Result_Kind_Hit_Mesh_))
    {
        hit_info_.hit = false;
        return hit_info_;
    }

    instance_ = scene_.instances._res_[hit_.instance_idx];
    mesh_ = scene_.meshes._res_[instance_.mesh_idx];
    indices_ = mesh_.indices;
    base_idx_ = (hit_.primitive_idx * 3);
    w_ = ((1.0 - hit_.barycentrics.x) - hit_.barycentrics.y);
    n0_ = mesh_.normal._res_[indices_._res_[(base_idx_ + 0)]];
    n1_ = mesh_.normal._res_[indices_._res_[(base_idx_ + 1)]];
    n2_ = mesh_.normal._res_[indices_._res_[(base_idx_ + 2)]];
    normal_ = normalize((((n0_ * w_) + (n1_ * hit_.barycentrics.x)) + (n2_ * hit_.barycentrics.y)));
    if(hit_.front_face)
    {
        normal_ *= (-1.0);
    }

    world_normal_ = normalize((transpose(hit_.world_to_object) * vec4(normal_.xyz, 1)));
    bary_ = hit_.barycentrics;
    hit_info_.hit = true;
    hit_info_.t = hit_.t;
    hit_info_.normal = world_normal_.xyz;
    hit_info_.mesh_idx = instance_.mesh_idx;
    hit_info_.tri_idx = hit_.primitive_idx;
    hit_info_.uv = hit_.barycentrics;
    return hit_info_;
}

uint hash_u32(uint seed_)
{
    uint x_ = uint_ZERO;
    x_ = seed_;
    x_ = (x_ ^ (x_ >> 17));
    x_ = (x_ * 0xed5ad4bb);
    x_ = (x_ ^ (x_ >> 11));
    x_ = (x_ * 0xac4c1b51);
    x_ = (x_ ^ (x_ >> 15));
    x_ = (x_ * 0x31848bab);
    x_ = (x_ ^ (x_ >> 14));
    return x_;
}

void init_rng(uint global_id_, uint accum_counter_)
{
    uint seed_ = uint_ZERO;
    seed_ = 0;
    RNG_STATE = hash_u32(((global_id_ * 19349663) ^ (accum_counter_ * 83492791)));
}

uint random_u32()
{
    uint result_ = uint_ZERO;
    RNG_STATE = ((RNG_STATE * 747796405) + 2891336453);
    result_ = (((RNG_STATE >> ((RNG_STATE >> 28) + 4)) ^ RNG_STATE) * 277803737);
    result_ = ((result_ >> 22) ^ result_);
    return result_;
}

float random_f32()
{
    uint result_ = uint_ZERO;
    RNG_STATE = ((RNG_STATE * 747796405) + 2891336453);
    result_ = (((RNG_STATE >> ((RNG_STATE >> 28) + 4)) ^ RNG_STATE) * 277803737);
    result_ = ((result_ >> 22) ^ result_);
    return (float(result_) / 4294967295.0);
}

vec2 random_vec2()
{
    float rnd0_ = float_ZERO;
    float rnd1_ = float_ZERO;
    rnd0_ = random_f32();
    rnd1_ = random_f32();
    return vec2(rnd0_, rnd1_);
}

float copysignf(float mag_, float sgn_)
{
    return ((sgn_ > 0)) ? (mag_) : ((-mag_));
}

mat4 basis_fromz(vec3 v_)
{
    vec3 z_ = vec3_ZERO;
    float sign_ = float_ZERO;
    float a_ = float_ZERO;
    float b_ = float_ZERO;
    vec3 x_ = vec3_ZERO;
    vec3 y_ = vec3_ZERO;
    z_ = normalize(v_);
    sign_ = copysignf(1.0, z_.z);
    a_ = ((-1.0) / (sign_ + z_.z));
    b_ = ((z_.x * z_.y) * a_);
    x_ = vec3((1.0 + (((sign_ * z_.x) * z_.x) * a_)), (sign_ * b_), ((-sign_) * z_.x));
    y_ = vec3(b_, (sign_ + ((z_.y * z_.y) * a_)), (-z_.y));
    return mat4(vec4(x_, 0), vec4(y_, 0), vec4(z_, 0), vec4(0, 0, 0, 0));
}

vec3 sample_hemisphere_cos(vec3 normal_, vec2 ruv_)
{
    float z_ = float_ZERO;
    float r_ = float_ZERO;
    float phi_ = float_ZERO;
    vec3 local_direction_ = vec3_ZERO;
    z_ = sqrt(ruv_.y);
    r_ = sqrt((1 - (z_ * z_)));
    phi_ = ((2 * pi) * ruv_.x);
    local_direction_ = vec3((r_ * cos(phi_)), (r_ * sin(phi_)), z_);
    return normalize((basis_fromz(normal_) * vec4(local_direction_, 0))).xyz;
}

float sample_hemisphere_cos_pdf(vec3 normal_, vec3 direction_)
{
    float cosw_ = float_ZERO;
    cosw_ = dot(normal_, direction_);
    if((cosw_ <= 0))
    {
        return 0;
    }
    else
    {
        return (cosw_ / pi);
return (cosw_ / pi);    }

}

Material_Point get_material_point(Scene scene_, Hit_Info hit_)
{
    vec4 color_sample_ = vec4_ZERO;
    Material_Point mat_point_ = Material_Point_ZERO;
    color_sample_ = vec4(1);
    if(true)
    {
        Mesh mesh_;
        _res_slice_uint indices_;
        uint base_idx_;
        vec2 uv0_;
        vec2 uv1_;
        vec2 uv2_;
        float w_;
        vec2 texcoords_;
        mesh_ = scene_.meshes._res_[hit_.mesh_idx];
        indices_ = mesh_.indices;
        base_idx_ = (hit_.tri_idx * 3);
        uv0_ = mesh_.uvs._res_[indices_._res_[(base_idx_ + 0)]];
        uv1_ = mesh_.uvs._res_[indices_._res_[(base_idx_ + 1)]];
        uv2_ = mesh_.uvs._res_[indices_._res_[(base_idx_ + 2)]];
        w_ = ((1.0 - hit_.uv.x) - hit_.uv.y);
        texcoords_ = (((uv0_ * w_) + (uv1_ * hit_.uv.x)) + (uv2_ * hit_.uv.y));
        if(true)
        {
            color_sample_ = texture(sampler2D(_res_textures_[nonuniformEXT(0)], _res_samplers_[nonuniformEXT(0)]), texcoords_);
        }

    }

    mat_point_.color = color_sample_.rgb;
    return mat_point_;
}

vec3 sample_matte(vec3 color_, vec3 normal_, vec3 outgoing_, vec2 rn_)
{
    return sample_hemisphere_cos(normal_, rn_);
}

vec3 eval_matte(vec3 color_, vec3 normal_, vec3 outgoing_, vec3 incoming_)
{
    return ((color_ / pi) * abs(dot(normal_, incoming_)));
}

float sample_matte_pdf(vec3 color_, vec3 normal_, vec3 outgoing_, vec3 incoming_)
{
    return sample_hemisphere_cos_pdf(normal_, incoming_);
}

bool is_finite(vec3 v_)
{
    bool is_x_finite_ = bool_ZERO;
    bool is_y_finite_ = bool_ZERO;
    bool is_z_finite_ = bool_ZERO;
    is_x_finite_ = ((floatBitsToInt(v_.x) & 0x7F800000) != 0x7F800000);
    is_y_finite_ = ((floatBitsToInt(v_.y) & 0x7F800000) != 0x7F800000);
    is_z_finite_ = ((floatBitsToInt(v_.z) & 0x7F800000) != 0x7F800000);
    return ((is_x_finite_ && is_y_finite_) && is_z_finite_);
}

