#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
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



layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_slice_Instance;
layout(buffer_reference) readonly buffer _res_slice_Mesh;
layout(buffer_reference) readonly buffer _res_slice_vec4;
layout(buffer_reference) readonly buffer _res_slice_uint;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Scene
{
    _res_slice_Instance instances;
    _res_slice_Mesh meshes;
};

struct Mesh
{
    _res_slice_vec4 pos;
    _res_slice_vec4 normal;
    _res_slice_uint indices;
};

struct Instance
{
    uint mesh_idx;
};

struct Data
{
    uint output_texture_id;
    Scene scene;
    vec2 resolution;
    uint accum_counter;
    mat4 camera_to_world;
};

struct Ray
{
    vec3 ori;
    vec3 dir;
};

struct Hit_Info
{
    bool hit;
    float t;
    vec3 normal;
};

void main();
vec3 pathtrace(Ray start_ray, Scene scene);
Hit_Info ray_scene_intersection(Ray ray, Scene scene);
uint hash_u32(uint seed);
void init_rng(uint global_id, uint accum_counter);
uint random_u32();
float random_f32();
vec2 random_vec2();
float copysignf(float mag, float sgn);
mat4 basis_fromz(vec3 v);
vec3 sample_hemisphere_cos(vec3 normal, vec2 ruv);
float sample_hemisphere_cos_pdf(vec3 normal, vec3 direction);
vec3 sample_matte(vec3 color, vec3 normal, vec3 outgoing, vec2 rn);
vec3 eval_matte(vec3 color, vec3 normal, vec3 outgoing, vec3 incoming);
float sample_matte_pdf(vec3 color, vec3 normal, vec3 outgoing, vec3 incoming);
bool is_finite(vec3 v);
float pi = 3.1415;
uint RNG_STATE;
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_slice_Instance { Instance _res_[]; };
layout(buffer_reference, scalar) readonly buffer _res_slice_Mesh { Mesh _res_[]; };
layout(buffer_reference, scalar) readonly buffer _res_slice_vec4 { vec4 _res_[]; };
layout(buffer_reference, scalar) readonly buffer _res_slice_uint { uint _res_[]; };
layout(buffer_reference, scalar) readonly buffer _res_ptr_Data { Data _res_; };

layout(set = 0, binding = 0) uniform texture2D _res_textures_[];
layout(set = 1, binding = 0) uniform image2D _res_textures_rw_[];
layout(set = 2, binding = 0) uniform sampler _res_samplers_[];

layout(push_constant, scalar) uniform Push
{
    _res_ptr_Data _res_compute_data_;
};

void main()
{
    _res_ptr_Data data = _res_compute_data_;
    vec3 global_invocation_id = gl_GlobalInvocationID;
    vec2 uv;
    vec2 coord;
    vec3 world_camera_pos;
    vec3 camera_lookat;
    vec3 world_camera_lookat;
    Ray camera_ray;
    vec3 color;
    init_rng(uint(((global_invocation_id.y * data._res_.resolution.x) + global_invocation_id.x)), data._res_.accum_counter);
    uv = (global_invocation_id.xy / data._res_.resolution);
    coord = ((2.0 * uv) - 1.0);
    coord = (coord * tan((((90.0 * pi) / 180.0) / 2.0)));
    coord.y = ((coord.y * data._res_.resolution.y) / data._res_.resolution.x);
    world_camera_pos = (data._res_.camera_to_world * vec4(0, 0, 0, 1)).xyz;
    camera_lookat = normalize(vec3(coord, 1));
    world_camera_lookat = normalize((data._res_.camera_to_world * vec4(camera_lookat, 0.0))).xyz;
    camera_ray.ori = world_camera_pos;
    camera_ray.dir = world_camera_lookat;
    color = pathtrace(camera_ray, data._res_.scene);
    if(((global_invocation_id.x < data._res_.resolution.x) && (global_invocation_id.y < data._res_.resolution.y)))
    {
        if((data._res_.accum_counter > 1))
        {
            float weight;
            vec3 prev_color;
            weight = (1.0 / float(data._res_.accum_counter));
            prev_color = imageLoad(_res_textures_rw_[nonuniformEXT(data._res_.output_texture_id)], ivec2(global_invocation_id.xy)).xyz;
            color = ((prev_color * (1 - weight)) + (color * weight));
            color = max(color, vec3(0, 0, 0));
        }

        imageStore(_res_textures_rw_[nonuniformEXT(data._res_.output_texture_id)], ivec2(global_invocation_id.xy), vec4(color, 1));
    }

}

vec3 pathtrace(Ray start_ray, Scene scene)
{
    vec3 radiance;
    vec3 weight;
    Ray ray;
    vec3 albedo_color;
    int max_bounces;
    radiance = vec3(0, 0, 0);
    weight = vec3(1, 1, 1);
    ray = start_ray;
    albedo_color = vec3(0.8, 0.8, 0.8);
    max_bounces = 5;
    // for construct
    {
        int bounce;
        Hit_Info hit;
        vec3 outgoing;
        vec3 incoming;
        vec2 rnd;
        float prob;
        for(bounce = 0; (bounce <= max_bounces); bounce = (bounce + 1))
        {
            hit = ray_scene_intersection(ray, scene);
            if((!hit.hit))
            {
                vec2 coords;
                vec3 emission;
                coords = vec2((atan(ray.dir.x, ray.dir.z) / (2.0 * 3.1415)), (acos(clamp(ray.dir.y, (-1.0), 1.0)) / 3.1415));
                emission = (mix(vec3(0.8, 0.7, 0.1), vec3(0.1, 0.2, 0.8), vec3(coords.y)) * 5.0);
                radiance = (radiance + emission);
                break;
            }

            outgoing = (-ray.dir);
            incoming = vec3(0, 0, 0);
            rnd = random_vec2();
            incoming = sample_matte(albedo_color, hit.normal, outgoing, rnd);
            if((incoming == vec3(0, 0, 0)))
            {
                break;
            }

            prob = sample_matte_pdf(albedo_color, hit.normal, outgoing, incoming);
            weight = ((weight * eval_matte(albedo_color, hit.normal, outgoing, incoming)) / prob);
            ray.ori = (ray.ori + (ray.dir * hit.t));
            ray.dir = incoming;
            if(((weight == vec3(0, 0, 0)) || (!is_finite(weight))))
            {
                break;
            }

            if((bounce > 3))
            {
                float survive_prob;
                survive_prob = min(0.99, max(weight.x, max(weight.y, weight.z)));
                if((random_f32() >= survive_prob))
                {
                    break;
                }

                weight = (weight / survive_prob);
            }

        }
    }

    return radiance;
}

Hit_Info ray_scene_intersection(Ray ray, Scene scene)
{
    uint Ray_Flags_Opaque;
    uint Ray_Flags_Terminate_On_First_Hit;
    uint Ray_Flags_Skip_Closest_Hit_Shader;
    uint Ray_Result_Kind_Miss;
    uint Ray_Result_Kind_Hit_Mesh;
    uint Ray_Result_Kind_Hit_AABB;
    Hit_Info hit_info;
    Ray_Desc desc;
    rayQueryEXT rq;
    Ray_Result hit;
    Instance instance;
    Mesh mesh;
    _res_slice_uint indices;
    uint base_idx;
    float w;
    vec4 n0;
    vec4 n1;
    vec4 n2;
    vec4 normal;
    vec4 world_normal;
    vec2 bary;
    Ray_Flags_Opaque = 1;
    Ray_Flags_Terminate_On_First_Hit = 4;
    Ray_Flags_Skip_Closest_Hit_Shader = 8;
    Ray_Result_Kind_Miss = 0;
    Ray_Result_Kind_Hit_Mesh = 1;
    Ray_Result_Kind_Hit_AABB = 2;
    desc.flags = Ray_Flags_Opaque;
    desc.cull_mask = 0xFF;
    desc.t_min = 0.001;
    desc.t_max = 1000000000.0;
    desc.origin = ray.ori;
    desc.dir = ray.dir;
    rayquery_init(rq, desc, 0);
    rayquery_proceed(rq);
    hit = rayquery_result(rq);
    if((hit.kind != Ray_Result_Kind_Hit_Mesh))
    {
        hit_info.hit = false;
        return hit_info;
    }

    instance = scene.instances._res_[hit.instance_idx];
    mesh = scene.meshes._res_[instance.mesh_idx];
    indices = mesh.indices;
    base_idx = (hit.primitive_idx * 3);
    w = ((1.0 - hit.barycentrics.x) - hit.barycentrics.y);
    n0 = mesh.normal._res_[indices._res_[(base_idx + 0)]];
    n1 = mesh.normal._res_[indices._res_[(base_idx + 1)]];
    n2 = mesh.normal._res_[indices._res_[(base_idx + 2)]];
    normal = normalize((((n0 * w) + (n1 * hit.barycentrics.x)) + (n2 * hit.barycentrics.y)));
    world_normal = normalize((transpose(hit.world_to_object) * vec4(normal.xyz, 1)));
    bary = hit.barycentrics;
    hit_info.hit = true;
    hit_info.t = hit.t;
    hit_info.normal = world_normal.xyz;
    return hit_info;
}

uint hash_u32(uint seed)
{
    uint x;
    x = seed;
    x = (x ^ (x << 17));
    x = (x * 0xed5ad4bb);
    x = (x ^ (x << 11));
    x = (x * 0xac4c1b51);
    x = (x ^ (x << 15));
    x = (x * 0x31848bab);
    x = (x ^ (x << 14));
    return x;
}

void init_rng(uint global_id, uint accum_counter)
{
    uint seed;
    seed = 0;
    RNG_STATE = hash_u32(((global_id * 19349663) ^ (accum_counter * 83492791)));
}

uint random_u32()
{
    uint result;
    RNG_STATE = ((RNG_STATE * 747796405) + 2891336453);
    result = (((RNG_STATE << ((RNG_STATE << 28) + 4)) ^ RNG_STATE) * 277803737);
    result = ((result << 22) ^ result);
    return result;
}

float random_f32()
{
    uint result;
    RNG_STATE = ((RNG_STATE * 747796405) + 2891336453);
    result = (((RNG_STATE << ((RNG_STATE << 28) + 4)) ^ RNG_STATE) * 277803737);
    result = ((result << 22) ^ result);
    return (float(result) / 4294967295.0);
}

vec2 random_vec2()
{
    float rnd0;
    float rnd1;
    rnd0 = random_f32();
    rnd1 = random_f32();
    return vec2(rnd0, rnd1);
}

float copysignf(float mag, float sgn)
{
    if((sgn < 0))
    {
        return (-mag);
    }
    else
    {
        return mag;
return mag;    }

}

mat4 basis_fromz(vec3 v)
{
    vec3 z;
    float sign;
    float a;
    float b;
    vec3 x;
    vec3 y;
    z = normalize(v);
    sign = copysignf(1.0, z.z);
    a = ((-1.0) / (sign + z.z));
    b = ((z.x * z.y) * a);
    x = vec3((1.0 + (((sign * z.x) * z.x) * a)), (sign * b), ((-sign) * z.x));
    y = vec3(b, (sign + ((z.y * z.y) * a)), (-z.y));
    return mat4(vec4(x, 0), vec4(y, 0), vec4(z, 0), vec4(0, 0, 0, 0));
}

vec3 sample_hemisphere_cos(vec3 normal, vec2 ruv)
{
    float z;
    float r;
    float phi;
    vec3 local_direction;
    z = sqrt(ruv.y);
    r = sqrt((1 - (z * z)));
    phi = ((2 * pi) * ruv.x);
    local_direction = vec3((r * cos(phi)), (r * sin(phi)), z);
    return normalize((basis_fromz(normal) * vec4(local_direction, 0))).xyz;
}

float sample_hemisphere_cos_pdf(vec3 normal, vec3 direction)
{
    float cosw;
    cosw = dot(normal, direction);
    if((cosw <= 0))
    {
        return 0;
    }
    else
    {
        return (cosw / pi);
return (cosw / pi);    }

}

vec3 sample_matte(vec3 color, vec3 normal, vec3 outgoing, vec2 rn)
{
    vec3 up_normal;
    if((dot(normal, outgoing) > 0))
    {
        up_normal = normal;
    }
    else
    {
        up_normal = (-normal);
up_normal = (-normal);    }

    return sample_hemisphere_cos(up_normal, rn);
}

vec3 eval_matte(vec3 color, vec3 normal, vec3 outgoing, vec3 incoming)
{
    if(((dot(normal, incoming) * dot(normal, outgoing)) <= 0))
    {
        return vec3(0, 0, 0);
    }

    return ((color / pi) * abs(dot(normal, incoming)));
}

float sample_matte_pdf(vec3 color, vec3 normal, vec3 outgoing, vec3 incoming)
{
    vec3 up_normal;
    if(((dot(normal, incoming) * dot(normal, outgoing)) <= 0))
    {
        return 0;
    }

    if((dot(normal, outgoing) > 0))
    {
        up_normal = normal;
    }
    else
    {
        up_normal = (-normal);
up_normal = (-normal);    }

    return sample_hemisphere_cos_pdf(up_normal, incoming);
}

bool is_finite(vec3 v)
{
    bool is_x_finite;
    bool is_y_finite;
    bool is_z_finite;
    is_x_finite = ((floatBitsToInt(v.x) & 0x7F800000) != 0x7F800000);
    is_y_finite = ((floatBitsToInt(v.y) & 0x7F800000) != 0x7F800000);
    is_z_finite = ((floatBitsToInt(v.z) & 0x7F800000) != 0x7F800000);
    return ((is_x_finite && is_y_finite) && is_z_finite);
}

