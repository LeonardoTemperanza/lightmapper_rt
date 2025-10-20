
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
    vec3 emission = vec3(10.0f);

    vec3 dir = gl_WorldRayDirectionEXT;
    vec2 coords = vec2(atan(dir.x, dir.z) / (2.0f * 3.1415f), acos(clamp(dir.y, -1.0f, 1.0f)) / 3.1415f);
    emission = mix(vec3(0.8, 0.7, 0.1), vec3(0.1, 0.2, 0.8), coords.y) * 20.0f;

    hit_info = HitInfo(false, false, vec3(0.0f), vec3(0.0f), vec3(0.0f), emission);
}
