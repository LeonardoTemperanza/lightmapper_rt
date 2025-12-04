
#version 450

layout(location = 0) out vec2 out_uv;

struct Line
{
    vec2 p[2];
};

struct Seam
{
    Line lines[2];
};

layout(set = 0, binding = 0) readonly buffer Seams
{
    Seam seams[];
};

layout(push_constant) uniform Push
{
    // Used for ping-pong buffering.
    uint render_to_line0;  // b32
    float target_size;
};

void main()
{
    uint line_id = gl_VertexIndex / 3;

    float texel_scale = 1.0f / target_size;

    Line target_line = render_to_line0 != 0 ? seams[line_id].lines[0] : seams[line_id].lines[1];
    Line other_line  = render_to_line0 != 0 ? seams[line_id].lines[1] : seams[line_id].lines[0];

    vec2 v =       (gl_VertexIndex % 2) == 0 ? target_line.p[0] : target_line.p[1];
    vec2 other_v = (gl_VertexIndex % 2) == 0 ? other_line.p[0] : other_line.p[1];

    vec2 final_v = v * 2.0f - 1.0f;
    gl_Position = vec4(final_v, 0.0f, 1.0f);
    out_uv = vec2(other_v.x, 1.0f - other_v.y);
}
