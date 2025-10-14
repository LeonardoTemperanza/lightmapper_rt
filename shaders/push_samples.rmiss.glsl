
#version 460
#extension GL_EXT_ray_tracing : require

struct HitInfo
{
    bool hit_backface;
    vec3 adjusted_pos;
};

layout(location = 0) rayPayloadInEXT HitInfo hit_info;

void main()
{
    hit_info.hit_backface = false;
}
