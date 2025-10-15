
#version 460
#extension GL_EXT_ray_tracing : require

struct Payload
{
    // Input
    vec3 geom_normal;

    // Output
    bool hit_backface;
    vec3 adjusted_pos;
};

layout(location = 0) rayPayloadInEXT Payload payload;

void main()
{
    payload.hit_backface = false;
}
