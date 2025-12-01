
#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba16f) uniform image2D lightmap;
layout(set = 0, binding = 2, rgba32f) uniform image2D gbuf_worldpos;
layout(set = 0, binding = 3, rgba8) uniform image2D gbuf_worldnormals;
layout(set = 0, binding = 4, rgba8) uniform image2D gbuf_worldgeomnormals;

struct Payload
{
    bool hit_backface;
    vec3 adjusted_pos;
    float dst;
};

layout(location = 0) rayPayloadEXT Payload payload;

#define PI      3.1415926
#define DEG2RAD PI / 180.0f;

const float T_MIN = 0.001f;
const float OVERSHOOT_FACTOR = 1.25f;

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

    vec4 gbuf_worldgeomnormals_sample = imageLoad(gbuf_worldgeomnormals, pixel);
    vec3 world_geom_normal = normalize(gbuf_worldgeomnormals_sample.xyz * 2.0f - 1.0f);

    vec4 gbuf_worldpos_sample = imageLoad(gbuf_worldpos, pixel);
    vec3 world_pos = gbuf_worldpos_sample.xyz;
    float texel_size = gbuf_worldpos_sample.w;

    //uint tri_idx = imageLoad(gbuf_tri_idx, pixel).x;

    vec3 right = normalize(cross(world_geom_normal, vec3(0.0f, 1.0f, 0.0f)));
    vec3 up = normalize(cross(right, world_geom_normal));
    float ray_length = texel_size * 0.5f * OVERSHOOT_FACTOR;

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

    bool found = false;
    float min_dst = 1000000000.0f;
    vec3 adjusted = vec3(0.0f);
    payload.hit_backface = false;
    for(int i = 0; i < NUM_RAYS; ++i)
    {
        ray_scene_intersection(Ray(world_pos + world_normal * 0.001f, normalize(dirs[i])), ray_length * length(dirs[i]));
        if(payload.hit_backface)
        {
            if(min_dst > payload.dst)
            {
                min_dst = payload.dst;
                adjusted = payload.adjusted_pos;
            }

            found = true;
        }
    }

    if(found) {
        imageStore(gbuf_worldpos, pixel, vec4(adjusted, -1.0f));
    }
}
