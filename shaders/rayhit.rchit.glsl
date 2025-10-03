
#version 460
#extension GL_EXT_ray_tracing : require

struct Payload
{
    vec3 color;
};

layout(location = 0) rayPayloadInEXT Payload payload;

hitAttributeEXT vec2 attribs;

void main()
{
    //vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    //payload.color = normalize(bary);
    payload.color = vec3(0.0f);
}
