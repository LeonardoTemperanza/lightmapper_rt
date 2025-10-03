
#version 460
#extension GL_EXT_ray_tracing : require

struct HitInfo
{
    bool hit;
    vec3 color;
};

layout(location = 0) rayPayloadInEXT HitInfo hit_info;

hitAttributeEXT vec2 attribs;

void main()
{
    //vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    //hit_info.color = normalize(bary);
    hit_info.hit = true;
    hit_info.color = vec3(0.0f);
}
