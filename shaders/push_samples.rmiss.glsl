
#version 460
#extension GL_EXT_ray_tracing : require

struct Payload
{
    bool hit_backface;
    vec3 adjusted_pos;
    float dst;
};

layout(location = 0) rayPayloadInEXT Payload payload;

void main()
{
    payload.hit_backface = false;
}
