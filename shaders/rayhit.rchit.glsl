
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

hitAttributeEXT vec2 attribs;

void main()
{
    //vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    //hit_info.color = normalize(bary);

    hit_info = HitInfo(true, vec3(0.0f), vec3(0.0f), vec3(1.0f), vec3(0.0f));
}
