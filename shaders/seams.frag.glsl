
#version 450

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 1) uniform sampler2D src_image;

void main()
{
    out_color = vec4(texture(src_image, in_uv).rbg, 0.1f);
    //out_color = vec4(0.0f, 1.0f, 0, 0.5f);
}
