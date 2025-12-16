
#version 460
#extension GL_EXT_ray_tracing : require

#define PI      3.1415926

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

layout(push_constant, std140) uniform Push
{
    uint accum_counter;
    uint seed;
    uint use_dir_light;
    float dir_light_angle;
    vec3 dir_light;
    uint padding;
    vec3 dir_light_emission;
} push;

layout(location = 0) rayPayloadInEXT HitInfo hit_info;

float sun_disk_falloff(vec3 ray_dir, vec3 sun_dir, float angular_radius)
{
    float cos_theta = dot(ray_dir, sun_dir);
    float cos_inner = cos(angular_radius);
    float cos_outer = cos(angular_radius * 1.5);

    return smoothstep(cos_outer, cos_inner, cos_theta);
}

//vec3 dir_light = normalize(vec3(0.2f, -1.0f, -0.1f));
//vec3 dir_light_emission = vec3(200000.0f, 184000.0f, 164000.0f);
//float dir_light_angle = 0.2 * (PI/180);

void main()
{
    const bool indirect_only = false;

    vec3 dir = gl_WorldRayDirectionEXT;
    vec2 coords = vec2(atan(dir.x, dir.z) / (2.0f * 3.1415f), acos(clamp(dir.y, -1.0f, 1.0f)) / 3.1415f);
    vec3 emission = vec3(0.57, 0.79, 1.09);

    emission += sun_disk_falloff(dir, -push.dir_light, push.dir_light_angle) * push.dir_light_emission;

    if(hit_info.first_bounce && indirect_only)
        emission = vec3(0.0f);

    hit_info = HitInfo(false, hit_info.first_bounce, false, vec3(0.0f), vec3(0.0f), vec3(0.0f), emission);
}
