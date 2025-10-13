
#version 450

layout(location = 0) in vec3 in_world_pos;
layout(location = 0) out vec4 out_color;

void main()
{
    // Approximate texel world size using derivatives. Assuming
    // there is no heavy distortion on lightmap uvs.
    // From: https://ndotl.wordpress.com/2018/08/29/baking-artifact-free-lightmaps/
    vec3 d_uv = max(abs(dFdxFine(in_world_pos)), abs(dFdyFine(in_world_pos)));
    float texel_world_size = max(max(d_uv.x, d_uv.y), d_uv.z) * sqrt(2.0f);

    out_color = vec4(in_world_pos, texel_world_size);
}
