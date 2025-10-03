
#version 460
#extension GL_EXT_ray_tracing : require

struct HitInfo
{
    bool hit;
    vec3 color;
};

layout(location = 0) rayPayloadInEXT HitInfo hit_info;

void main()
{
    hit_info.hit = false;
    hit_info.color = vec3(1.0f);
}
