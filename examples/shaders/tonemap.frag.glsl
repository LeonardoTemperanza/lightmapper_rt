
#version 450

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D src;

vec4 linear_to_srgb(vec4 color)
{
    bvec3 cutoff = lessThan(color.rgb, vec3(0.0031308));
    vec3 higher = vec3(1.055) * pow(color.rgb, vec3(1.0/2.4)) - vec3(0.055);
    vec3 lower = color.rgb * vec3(12.92);

    return vec4(mix(higher, lower, cutoff), color.a);
}

vec3 filmic(vec3 x)
{
    vec3 X = max(vec3(0.0f), x - 0.004f);
    vec3 result = (X * (6.2 * X + 0.5)) / (X * (6.2 * X + 1.7) + 0.06);
    return pow(result, vec3(2.2));
}

vec4 hdr_to_ldr(vec4 color)
{
    return vec4(filmic(color.rgb), color.a);
}

void main()
{
    vec4 color = texture(src, in_uv);
    //color *= pow(2.0, 5.0);
    out_color = linear_to_srgb(hdr_to_ldr(max(vec4(0.0f), color)));
}
