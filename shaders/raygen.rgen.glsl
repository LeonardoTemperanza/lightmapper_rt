
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
    bool hit_backface;
    vec3 world_pos;
    vec3 world_normal;
    vec3 albedo;
    vec3 emission;
};

layout(location = 0) rayPayloadEXT HitInfo hit_info;

#define PI      3.1415926
#define DEG2RAD PI / 180.0f;

const float T_MIN = 0.01;
const float T_MAX = 1000000.0f;

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

float sample_hemisphere_cos_pdf(vec3 normal, vec3 direction)
{
    float cosw = dot(normal, direction);
    return cosw <= 0.0f ? 0.0f : cosw / PI;
}

vec3 sample_matte(vec3 color, vec3 normal, vec3 outgoing, vec2 rn)
{
    vec3 up_normal = dot(normal, outgoing) > 0.0f ? normal : -normal;
    return sample_hemisphere_cos(up_normal, rn);
}

vec3 eval_matte(vec3 color, vec3 normal, vec3 outgoing, vec3 incoming)
{
    if(dot(normal, incoming) * dot(normal, outgoing) <= 0) return vec3(0.0f);
    return color / PI * abs(dot(normal, incoming));
}

float sample_matte_pdf(vec3 color, vec3 normal, vec3 outgoing, vec3 incoming)
{
    if(dot(normal, incoming) * dot(normal, outgoing) <= 0.0f) return 0.0f;
    vec3 up_normal = dot(normal, outgoing) <= 0.0f ? -normal : normal;
    return sample_hemisphere_cos_pdf(up_normal, incoming);
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

// Alpha stores the validity of this sample, in [0, 1].
vec4 pathtrace(vec3 start_pos, vec3 world_normal)
{
    vec3 radiance = vec3(0.0f);
    vec3 weight = vec3(1.0f);
    Ray ray = Ray(start_pos, world_normal);
    vec3 outgoing = world_normal;

    // Initialize the first hit to be
    hit_info.hit = true;
    hit_info.albedo = vec3(0.7f);
    hit_info.emission = vec3(0.0f);
    hit_info.world_normal = world_normal;
    hit_info.world_pos = start_pos;

    vec3 hit_pos = start_pos;

    const uint MAX_BOUNCES = 5;
    uint backface_hits_count = 0;
    for(uint bounce = 0; bounce <= MAX_BOUNCES; ++bounce)
    {
        if(bounce != 0)
            ray_scene_intersection(ray);
        if(!hit_info.hit)
        {
            radiance += hit_info.emission * weight;
            break;
        }

        if(bounce > 0 && hit_info.hit_backface) ++backface_hits_count;

        if(bounce != 0)
        {
            hit_pos = hit_info.world_pos;
            outgoing = -ray.dir;
        }

        // Accumulate emission
        radiance += weight * hit_info.emission;

        vec3 incoming = sample_matte(hit_info.albedo, hit_info.world_normal, outgoing, random_vec2());
        if(incoming == vec3(0.0f)) break;
        weight *= eval_matte(hit_info.albedo, hit_info.world_normal, outgoing, incoming) /
                  sample_matte_pdf(hit_info.albedo, hit_info.world_normal, outgoing, incoming);

        // Update ray
        ray.ori = hit_pos;
        ray.dir = incoming;
    }

    //float validity = 1.0f - (float(backface_hits_count) / float(MAX_BOUNCES - 1));
    //float validity = backface_hits_count > 0 ? 0.0f : 1.0f;
    float validity = 1.0f;
    return vec4(radiance, validity);
}

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
        imageStore(lightmap, pixel, vec4(vec3(0.0f), 0.0f));
        return;
    }

    vec4 gbuf_worldpos_sample = imageLoad(gbuf_worldpos, pixel);
    vec3 world_pos = gbuf_worldpos_sample.xyz;

    init_rng(pixel.y * size.x + pixel.x);

    uint NUM_SAMPLES = 1;
    vec4 color = vec4(0.0f);
    for(int i = 0; i < NUM_SAMPLES; ++i)
        color += pathtrace(world_pos, world_normal);
    color /= NUM_SAMPLES;

    // Progressive pathtracing.
    if(push.accum_counter != 0)
    {
        float weight = 1.0f / float(push.accum_counter);
        vec4 prev_color = imageLoad(lightmap, pixel);
        color = prev_color * (1.0f - weight) + color * weight;
        color = max(color, vec4(0.0f));
    }

    imageStore(lightmap, pixel, color);
}
