
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_ray_tracing_position_fetch : require

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

layout(set = 0, binding = 4) readonly buffer Geometries
{
    Geometry geometries[];
};

struct HitInfo
{
    bool hit_backface;
    vec3 adjusted_pos;
    float dst;
};

layout(location = 0) rayPayloadInEXT HitInfo hit_info;

const float BIAS = 0.001f;

hitAttributeEXT vec2 attribs;

void main()
{
    if(gl_HitKindEXT != gl_HitKindBackFacingTriangleEXT)
    {
        hit_info.hit_backface = false;
        return;
    }

    vec3 v0 = gl_HitTriangleVertexPositionsEXT[0];
    vec3 v1 = gl_HitTriangleVertexPositionsEXT[1];
    vec3 v2 = gl_HitTriangleVertexPositionsEXT[2];
    vec3 geom_normal = normalize(cross(v1 - v0, v2 - v0));
    vec3 geom_normal_out = faceforward(geom_normal, -gl_WorldRayDirectionEXT, geom_normal);

    vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    hit_info.hit_backface = true;
    hit_info.adjusted_pos = hit_pos + geom_normal_out * BIAS;
    hit_info.dst = gl_HitTEXT;
}
