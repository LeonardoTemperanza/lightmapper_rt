
#version 460
#extension GL_EXT_ray_tracing : require

#define PI      3.1415926

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

float sun_disk_falloff(vec3 ray_dir, vec3 sun_dir, float angular_radius)
{
    float cos_theta = dot(ray_dir, sun_dir);
    float cos_inner = cos(angular_radius);
    float cos_outer = cos(angular_radius * 1.5);

    return smoothstep(cos_outer, cos_inner, cos_theta);
}

vec3 dir_light = normalize(vec3(0.2f, -1.0f, -0.1f));
vec3 dir_light_emission = vec3(1000.0f, 920.0f, 820.0f);
float dir_light_angle = 0.2 * (PI/180);

void main()
{
    vec3 dir = gl_WorldRayDirectionEXT;
    vec2 coords = vec2(atan(dir.x, dir.z) / (2.0f * 3.1415f), acos(clamp(dir.y, -1.0f, 1.0f)) / 3.1415f);
    vec3 emission = vec3(0.57, 0.79, 1.09);

    emission += sun_disk_falloff(dir, -dir_light, dir_light_angle) * dir_light_emission;

    hit_info = HitInfo(false, false, vec3(0.0f), vec3(0.0f), vec3(0.0f), emission);
}
