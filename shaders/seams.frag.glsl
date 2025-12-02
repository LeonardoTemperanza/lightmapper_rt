
#version 450

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D src_image;

void main()
{
    out_color = vec4(texture(src_image, in_uv).rbg, 0.5f);
}
