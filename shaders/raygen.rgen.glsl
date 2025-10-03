
#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba16f) uniform image2D lightmap;
layout(set = 0, binding = 2, rgba32f) uniform image2D gbuf_worldpos;
layout(set = 0, binding = 3, rgba8) uniform image2D gbuf_worldnormals;

layout(push_constant) uniform Push
{
    uint accum_counter;
    uint seed;
} push;

struct HitInfo
{
    bool hit;
    vec3 color;
};

layout(location = 0) rayPayloadEXT HitInfo hit_info;

#define PI      3.1415926
#define DEG2RAD PI / 180.0f;

const float T_MIN = 0.001;
const float T_MAX = 10000.0f;

uint RNG_STATE = 0;

struct Ray
{
    vec3 ori;
    vec3 dir;
};

uint hash_u32(uint seed)
{
    uint x = seed;
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return x;
}

void init_rng(uint global_id)
{
    uint seed = 0;
    RNG_STATE = hash_u32(global_id * 19349663 ^ push.accum_counter * 83492791 ^ push.seed * 73856093);
}

// PCG Random number generator.
// From: www.pcg-random.org and www.shadertoy.com/view/XlGcRh
uint random_u32()
{
    RNG_STATE = RNG_STATE * 747796405u + 2891336453u;
    uint result = ((RNG_STATE >> ((RNG_STATE >> 28u) + 4)) ^ RNG_STATE) * 277803737;
    result = (result >> 22u) ^ result;
    return result;
}

// From 0 (inclusive) to 1 (exclusive)
float random_f32()
{
    RNG_STATE = RNG_STATE * 747796405 + 2891336453;
    uint result = ((RNG_STATE >> ((RNG_STATE >> 28) + 4)) ^ RNG_STATE) * 277803737;
    result = (result >> 22) ^ result;
    return float(result) / 4294967295.0f;
}

vec2 random_vec2()
{
    // Enforce evaluation order.
    float rnd0 = random_f32();
    float rnd1 = random_f32();
    return vec2(rnd0, rnd1);
}

float copysignf(float mag, float sgn) { return sgn < 0.0f ? -mag : mag; }

mat3 basis_fromz(vec3 v)
{
    // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    vec3 z   = normalize(v);
    float sign = copysignf(1.0f, z.z);
    float a    = -1.0f / (sign + z.z);
    float b    = z.x * z.y * a;
    vec3 x     = vec3(1.0f + sign * z.x * z.x * a, sign * b, -sign * z.x);
    vec3 y     = vec3(b, sign + z.y * z.y * a, -z.y);
    return mat3(x, y, z);
}

vec3 sample_hemisphere_cos(vec3 normal, vec2 ruv)
{
    float z              = sqrt(ruv.y);
    float r              = sqrt(1 - z * z);
    float phi            = 2 * PI * ruv.x;
    vec3 local_direction = vec3(r * cos(phi), r * sin(phi), z);
    return normalize(basis_fromz(normal) * local_direction);
}

vec3 pathtrace(vec3 world_pos, vec3 world_normal);

void main()
{
    ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);
    ivec2 size = ivec2(gl_LaunchSizeEXT.xy);
    pixel.y = size.y - 1 - pixel.y;

    vec4 gbuf_worldnormals_sample = imageLoad(gbuf_worldnormals, pixel);
    vec3 world_normal = normalize(gbuf_worldnormals_sample.xyz * 2.0f - 1.0f);
    float validity = gbuf_worldnormals_sample.a;
    if(validity == 0.0f)
    {
        imageStore(lightmap, pixel, vec4(vec3(0.0f), 1.0f));
        return;
    }

    vec4 gbuf_worldpos_sample = imageLoad(gbuf_worldpos, pixel);
    vec3 world_pos = gbuf_worldpos_sample.xyz;

    init_rng(pixel.y * size.x + pixel.x);

    vec3 color = pathtrace(world_pos, world_normal);
    imageStore(lightmap, pixel, vec4(color, 1.0f));
}

vec3 sample_matte(vec3 color, vec3 normal, vec3 outgoing, vec2 rn)
{
    vec3 up_normal = dot(normal, outgoing) > 0.0f ? normal : -normal;
    return sample_hemisphere_cos(up_normal, rn);
}

// Stores result in payload.
void ray_scene_intersection(Ray ray)
{
    uint ray_flags = gl_RayFlagsOpaqueEXT;
    uint cull_mask = 0xFF;
    uint sbt_record_offset = 0;
    uint sbt_record_stride = 0;
    uint miss_index = 0;
    vec3 origin = ray.ori;
    float t_min = T_MIN;
    vec3 direction = ray.dir;
    float t_max = T_MAX;
    const int payload_loc = 0;
    traceRayEXT(tlas, ray_flags, cull_mask, sbt_record_offset, sbt_record_stride, miss_index, origin, t_min, direction, t_max, payload_loc);
}

vec3 pathtrace(vec3 start_pos, vec3 world_normal)
{
    Ray ray = Ray(start_pos, -world_normal);

    const uint MAX_BOUNCES = 1;
    for(uint bounce = 0; bounce < MAX_BOUNCES; ++bounce)
    {
        vec3 outgoing = -ray.dir;

        vec3 incoming = sample_matte(vec3(0.8f), world_normal, outgoing, random_vec2());
        Ray ray = Ray(start_pos, incoming);
        ray_scene_intersection(ray);
        // TODO: Modify ray ori and dir according to hit
    }

    return hit_info.color;
}
