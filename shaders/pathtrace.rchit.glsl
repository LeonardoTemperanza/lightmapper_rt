
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

const uint MAX_U32 = 4294967295;
const uint SENTINEL_IDX = MAX_U32;

layout(buffer_reference, std430) readonly buffer Uvs
{
    float buf[];
};

layout(buffer_reference, std430) readonly buffer Normals
{
    float buf[];
};

layout(buffer_reference, std430) readonly buffer Indices
{
    uint buf[];
};

struct Geometry
{
    Normals normals;
    Indices indices;
    Uvs uvs;
};

layout(set = 0, binding = 5) readonly buffer Geometries
{
    Geometry geometries[];
};

// Static scene resources
layout(set = 1, binding = 0) uniform sampler2D textures[];

// Dynamic scene resources
struct Instance
{
    uint albedo_tex_idx;
};

layout(set = 2, binding = 0) readonly buffer Instances
{
    Instance instances[];
};

struct HitInfo
{
    bool hit;
    bool first_bounce;
    bool hit_backface;
    vec3 world_pos;
    vec3 world_normal;
    vec3 albedo;
    vec3 emission;
};

layout(location = 0) rayPayloadInEXT HitInfo hit_info;

hitAttributeEXT vec2 attribs;

void main()
{
    uint instance_idx = gl_InstanceCustomIndexEXT;
    Instance instance = instances[instance_idx];

    Geometry geom = geometries[instance_idx];

    uint idx_base = gl_PrimitiveID * 3;
    uvec3 indices = uvec3(geom.indices.buf[idx_base], geom.indices.buf[idx_base+1], geom.indices.buf[idx_base+2]);

    float w = 1.0f - attribs.x - attribs.y;

    vec3 n0 = vec3(geom.normals.buf[indices.x * 3 + 0], geom.normals.buf[indices.x * 3 + 1], geom.normals.buf[indices.x * 3 + 2]);
    vec3 n1 = vec3(geom.normals.buf[indices.y * 3 + 0], geom.normals.buf[indices.y * 3 + 1], geom.normals.buf[indices.y * 3 + 2]);
    vec3 n2 = vec3(geom.normals.buf[indices.z * 3 + 0], geom.normals.buf[indices.z * 3 + 1], geom.normals.buf[indices.z * 3 + 2]);
    vec3 normal = normalize(n0*w + n1*attribs.x + n2*attribs.y);
    vec3 world_normal = normalize(transpose(mat3(gl_WorldToObjectEXT)) * normal);

    vec2 uv0 = vec2(geom.uvs.buf[indices.x * 2 + 0], geom.uvs.buf[indices.x * 2 + 1]);
    vec2 uv1 = vec2(geom.uvs.buf[indices.y * 2 + 0], geom.uvs.buf[indices.y * 2 + 1]);
    vec2 uv2 = vec2(geom.uvs.buf[indices.z * 2 + 0], geom.uvs.buf[indices.z * 2 + 1]);
    vec2 uv = uv0*w + uv1*attribs.x + uv2*attribs.y;

    vec4 albedo_sample = vec4(1.0f, 1.0f, 1.0f, 1.0f);
    if(instance.albedo_tex_idx != SENTINEL_IDX)
        albedo_sample = texture(textures[nonuniformEXT(instance.albedo_tex_idx)], uv);

    vec4 albedo = albedo_sample;
    //vec4 albedo = world_normal * 0.5f + 0.5f;
    //vec4 albedo = vec4(vec3(0.7f), 1.0f);

    vec3 world_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    bool hit_backface = gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT;

    hit_info.hit = true;
    hit_info.world_pos = world_pos;
    hit_info.world_normal = world_normal;
    hit_info.hit_backface = hit_backface;
    hit_info.albedo = albedo.rgb;
    hit_info.emission = vec3(0.0f);
}
