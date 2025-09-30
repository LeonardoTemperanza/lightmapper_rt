
#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba32f) uniform image2D outputImage;

struct Payload
{
    vec3 color;
};

layout(location = 0) rayPayloadEXT Payload payload;

void main()
{
    ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);
    ivec2 size = ivec2(gl_LaunchSizeEXT.xy);

    imageStore(outputImage, pixel, vec4(1.0f, 0.0f, 1.0f, 1.0f));

    /*
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    vec3 origin = vec3(0, 0, -5);
    vec3 dir = normalize(vec3(uv - 0.5, 1.0));

    payload.color = vec3(0);

    traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, origin, 0.001, dir, 10000.0, 0);
    vec3 result = payload.color;
    imageStore(outputImage, pixel, vec4(result, 1.0));
    */
}
