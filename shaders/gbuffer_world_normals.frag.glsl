
#version 450

layout(location = 1) in vec3 in_world_normal;
layout(location = 0) out vec4 out_color;

void main()
{
    out_color = vec4(normalize(in_world_normal) * 0.5f + 0.5f, 1.0f);
}
