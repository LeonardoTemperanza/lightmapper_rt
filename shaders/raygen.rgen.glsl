
#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba16f) uniform image2D lightmap;
layout(set = 0, binding = 2, rgba32f) uniform image2D gbuf_worldpos;
layout(set = 0, binding = 3, rgba8) uniform image2D gbuf_worldnormals;

/*
layout(push_constant) uniform Push
{
    uint sbt_offset;
    uint sbt_stride;
}
*/

struct Payload
{
    vec3 color;
};

layout(location = 0) rayPayloadEXT Payload payload;

void main()
{
    ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);
    ivec2 size = ivec2(gl_LaunchSizeEXT.xy);
    pixel.y = size.y - 1 - pixel.y;

    vec4 gbuf_worldpos_sample = imageLoad(gbuf_worldpos, pixel);
    vec3 world_pos = gbuf_worldpos_sample.xyz;
    float validity = gbuf_worldpos_sample.a;
    if(validity == 0.0f)
    {
        imageStore(lightmap, pixel, vec4(vec3(0.0f), 1.0f));
        return;
    }

    vec4 gbuf_worldnormals_sample = imageLoad(gbuf_worldnormals, pixel);
    vec3 world_normal = normalize(gbuf_worldnormals_sample.xyz * 2.0f - 1.0f);

    payload.color = vec3(1.0f, 0.0f, 1.0f);  // Initialize to a known value

    uint ray_flags = gl_RayFlagsOpaqueEXT;
    uint cull_mask = 0xFF;
    uint sbt_record_offset = 0;
    uint sbt_record_stride = 0;
    uint miss_index = 0;
    vec3 origin = world_pos;
    float t_min = 0.001f;
    vec3 direction = world_normal;
    //vec3 direction = vec3(0.0f, 1.0f, 0.0f);
    float t_max = 10000.0f;
    const int payload_loc = 0;
    traceRayEXT(tlas, ray_flags, cull_mask, sbt_record_offset, sbt_record_stride, miss_index, origin, t_min, direction, t_max, payload_loc);

    imageStore(lightmap, pixel, vec4(payload.color, 1.0f));
}
