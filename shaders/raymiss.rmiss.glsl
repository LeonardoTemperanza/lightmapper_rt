
#version 460
#extension GL_EXT_ray_tracing : require

struct HitInfo
{
    bool hit;
    bool hit_backface;
    vec3 world_pos;
    vec3 world_normal;
    vec3 albedo;
    vec3 emission;
};

layout(location = 0) rayPayloadInEXT HitInfo hit_info;

void main()
{
    hit_info = HitInfo(false, false, vec3(0.0f), vec3(0.0f), vec3(0.0f), vec3(10.0f));
}
