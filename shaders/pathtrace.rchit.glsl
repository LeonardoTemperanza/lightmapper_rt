
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require

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
};

layout(set = 0, binding = 5) readonly buffer Geometries
{
    Geometry geometries[];
};

// Static scene resources
//layout(set = 1, binding = 0) uniform sampler2D textures[];

// Dynamic scene resources
//layout(set = 2, binding = 0) readonly buffer

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
    Geometry geom = geometries[gl_InstanceCustomIndexEXT];

    uint idx_base = gl_PrimitiveID * 3;
    uvec3 indices = uvec3(geom.indices.buf[idx_base], geom.indices.buf[idx_base+1], geom.indices.buf[idx_base+2]);

    vec3 n0 = vec3(geom.normals.buf[indices.x * 3 + 0], geom.normals.buf[indices.x * 3 + 1], geom.normals.buf[indices.x * 3 + 2]);
    vec3 n1 = vec3(geom.normals.buf[indices.y * 3 + 0], geom.normals.buf[indices.y * 3 + 1], geom.normals.buf[indices.y * 3 + 2]);
    vec3 n2 = vec3(geom.normals.buf[indices.z * 3 + 0], geom.normals.buf[indices.z * 3 + 1], geom.normals.buf[indices.z * 3 + 2]);
    float w = 1.0f - attribs.x - attribs.y;
    vec3 normal = normalize(n0*w + n1*attribs.x + n2*attribs.y);
    vec3 world_normal = normalize(transpose(mat3(gl_WorldToObjectEXT)) * normal);

    vec3 world_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    //vec3 albedo = world_normal * 0.5f + 0.5f;
    vec3 albedo = vec3(0.7f);
    bool hit_backface = gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT;
    hit_info = HitInfo(true, hit_info.first_bounce, hit_backface, world_pos, world_normal, albedo, vec3(0.0f));
}
