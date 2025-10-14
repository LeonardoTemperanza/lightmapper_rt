
#version 450

layout(location = 0) out uint out_tri_idx;

void main()
{
    out_tri_idx = gl_PrimitiveID;
}
