
#version 450

// Meant to be used with no vertex buffer, 6 indices.
// Assumes counter-clockwise order.

layout(location = 0) out vec2 out_uv;

void main()
{
    vec4 verts[6] = {
        vec4(-1.0f,  1.0f, 0.0f, 1.0f),  // Bottom-left tri
        vec4(-1.0f, -1.0f, 0.0f, 1.0f),
        vec4( 1.0f, -1.0f, 0.0f, 1.0f),
        vec4(-1.0f,  1.0f, 0.0f, 1.0f),  // Top-right tri
        vec4( 1.0f, -1.0f, 0.0f, 1.0f),
        vec4( 1.0f,  1.0f, 0.0f, 1.0f),
    };

    vec2 uvs[6] = {
        vec2(0.0f, 0.0f),
        vec2(0.0f, 1.0f),
        vec2(1.0f, 1.0f),
        vec2(0.0f, 0.0f),
        vec2(1.0f, 1.0f),
        vec2(1.0f, 0.0f),
    };

    gl_Position = verts[gl_VertexIndex];
    out_uv = vec2(uvs[gl_VertexIndex].x, 1.0f - uvs[gl_VertexIndex].y);
}
