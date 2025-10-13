
#version 460
#extension GL_EXT_ray_tracing : require

struct HitInfo
{
    bool hit;
    vec3 world_pos;
    vec3 world_normal;
    vec3 albedo;
    vec3 emission;
};

layout(location = 0) rayPayloadInEXT HitInfo hit_info;

void main()
{
    hit_info = HitInfo(false, vec3(0.0f), vec3(0.0f), vec3(0.0f), vec3(50.0f));
}
