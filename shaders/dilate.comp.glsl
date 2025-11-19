
#version 450

layout(local_size_x = 8) in;
layout(local_size_y = 8) in;
layout(local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D src_image;
layout(set = 0, binding = 1) writeonly uniform image2D dst_image;

const float VALIDITY_THRESHOLD = 0.8f;

void main(void)
{
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 tex_size = min(textureSize(src_image, 0), imageSize(dst_image));
    if(coord.x >= tex_size.x || coord.y >= tex_size.y)
        return;

    vec4 src_color = texelFetch(src_image, coord, 0);
    if(src_color.a > VALIDITY_THRESHOLD)
    {
        imageStore(dst_image, coord, src_color);
        return;
    }

    // Search neighbors for a valid pixel.
    vec4 color_sum = vec4(0.0f);
    int count = 0;
    for(int dy = -1; dy <= 1; ++dy)
    {
        for(int dx = -1; dx <= 1; ++dx)
        {
            ivec2 neighbor_coord = clamp(coord + ivec2(dx, dy), ivec2(0), ivec2(tex_size));
            vec4 neighbor_color = texelFetch(src_image, neighbor_coord, 0);
            if(neighbor_color.a > VALIDITY_THRESHOLD)
            {
                color_sum += neighbor_color;
                ++count;
            }
        }
    }

    if(count > 0)
        imageStore(dst_image, coord, vec4(color_sum / float(count)));
    else
        imageStore(dst_image, coord, src_color);
}
