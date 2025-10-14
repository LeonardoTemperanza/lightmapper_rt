
#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba16f) uniform image2D lightmap;
layout(set = 0, binding = 2, rgba32f) uniform image2D gbuf_worldpos;
layout(set = 0, binding = 3, rgba8) uniform image2D gbuf_worldnormals;

struct HitInfo
{
    bool hit_backface;
    vec3 adjusted_pos;
};

layout(location = 0) rayPayloadEXT HitInfo hit_info;

#define PI      3.1415926
#define DEG2RAD PI / 180.0f;

const float T_MIN = 0.001;

struct Ray
{
    vec3 ori;
    vec3 dir;
};

// Stores result in payload.
void ray_scene_intersection(Ray ray, float t_max)
{
    uint ray_flags = gl_RayFlagsOpaqueEXT;
    uint cull_mask = 0xFF;
    uint sbt_record_offset = 0;
    uint sbt_record_stride = 0;
    uint miss_index = 0;
    vec3 origin = ray.ori;
    float t_min = T_MIN;
    vec3 direction = ray.dir;
    float _t_max = t_max;
    const int payload_loc = 0;
    traceRayEXT(tlas, ray_flags, cull_mask, sbt_record_offset, sbt_record_stride, miss_index, origin, t_min, direction, _t_max, payload_loc);
}

void main()
{
    ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);
    ivec2 size = ivec2(gl_LaunchSizeEXT.xy);
    pixel.y = size.y - 1 - pixel.y;

    vec4 gbuf_worldnormals_sample = imageLoad(gbuf_worldnormals, pixel);
    vec3 world_normal = normalize(gbuf_worldnormals_sample.xyz * 2.0f - 1.0f);
    float validity = gbuf_worldnormals_sample.a;
    if(validity == 0.0f) return;

    vec4 gbuf_worldpos_sample = imageLoad(gbuf_worldpos, pixel);
    vec3 world_pos = gbuf_worldpos_sample.xyz;
    float texel_size = gbuf_worldpos_sample.w;

    vec3 right = normalize(cross(world_normal, vec3(0.0f, 1.0f, 0.0f)));
    vec3 up = normalize(cross(right, world_normal));
    float ray_length = texel_size * 1.25f;

    const uint NUM_RAYS = 8;
    vec3 dirs[NUM_RAYS] = {
        right,
        (right + up),
        up,
        (-right + up),
        -right,
        (-right - up),
        -up,
        (-up + right),
    };

    hit_info.hit_backface = false;
    for(int i = 0; i < NUM_RAYS; ++i)
    {
        ray_scene_intersection(Ray(world_pos, dirs[i]), ray_length);
        if(hit_info.hit_backface) break;
    }

    if(hit_info.hit_backface) {
        imageStore(gbuf_worldpos, pixel, vec4(hit_info.adjusted_pos, texel_size));
    }
}
