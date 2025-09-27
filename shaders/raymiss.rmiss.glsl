
#version 460
#extension GL_EXT_ray_tracing : require

struct Payload
{
    vec3 color;
    int hit;
};

layout(location = 0) rayPayloadInEXT Payload payload;

void main()
{
    payload.color = vec3(0.0f);
    payload.hit = 0;
}
