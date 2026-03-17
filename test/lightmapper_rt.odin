
package test

import "../no_gfx_api/gpu"

Context :: struct
{

}

init :: proc()
{

}

cleanup :: proc()
{

}

Bake :: struct
{
    shaders: Shaders,
    gbufs: GBuffers,
    scene: Scene,
}

bake_begin :: proc(#any_int lightmap_size: i64, scene: Scene) -> Bake
{
    bake: Bake
    bake.shaders = shaders_create()
    bake.gbufs = gbufs_create(lightmap_size)
    bake.scene = scene

    cmd_buf := gpu.commands_begin(.Main)
    gbufs_render(cmd_buf, &bake.gbufs, bake.shaders, scene)
    gpu.cmd_barrier(cmd_buf, .All, .All, {})
    gpu.queue_submit(.Main, { cmd_buf })
    return bake
}

bake_iteration :: proc(bake: ^Bake)
{
    cmd_buf := gpu.commands_begin(.Main)
    // pathtrace(cmd_buf, bake.gbufs, bake.shaders, bake.scene)
    gpu.queue_submit(.Main, { cmd_buf })
}

bake_end :: proc(bake: ^Bake)
{

}

// internal

Scene :: struct
{
    instances: [dynamic]Instance,
    meshes: [dynamic]Mesh,
}

Scene_Shader :: struct
{
    instances: rawptr,
    meshes: rawptr,
}

Instance :: struct
{
    mesh_idx: u32,
    transform: matrix[4, 4]f32,
}

Mesh :: struct
{
    positions: gpu.slice_t([3]f32),
    normals: gpu.slice_t([3]f32),
    uvs: gpu.slice_t([2]f32),
    lm_uvs: gpu.slice_t([2]f32),
}

Mesh_Shader :: struct
{
    positions: rawptr,
    normals: rawptr,
    uvs: rawptr,
    lm_uvs: rawptr,
}

Shaders :: struct
{
    uv_space: gpu.Shader,
    gbuffers: gpu.Shader,
    pathtrace: gpu.Shader
}

shaders_create :: proc() -> Shaders
{
    res: Shaders
    res.uv_space = gpu.shader_create(#load("../shaders/uv_space.vert.spv", []u32), .Vertex)
    res.gbuffers = gpu.shader_create(#load("../shaders/gbuffers.frag.spv", []u32), .Fragment)
    res.pathtrace = gpu.shader_create_compute(#load("../shaders/pathtrace.comp.spv", []u32), 8, 8, 1)
    return res
}

shader_destroy :: proc(shaders: ^Shaders)
{
    gpu.shader_destroy(shaders.uv_space)
    gpu.shader_destroy(shaders.gbuffers)
    gpu.shader_destroy(shaders.pathtrace)
    shaders^ = {}
}

GBuffers :: struct
{
    world_pos: gpu.Owned_Texture,
    world_normals: gpu.Owned_Texture,
}

gbufs_create :: proc(#any_int lightmap_size: i64) -> GBuffers
{
    gbufs: GBuffers
    gbufs.world_pos = gpu.texture_alloc_and_create({
        dimensions = { u32(lightmap_size), u32(lightmap_size), 1 },
        format = .RGBA32_Float,
    })
    gbufs.world_normals = gpu.texture_alloc_and_create({
        dimensions = { u32(lightmap_size), u32(lightmap_size), 1 },
        format = .RGBA8_Unorm,
    })
    return gbufs
}

gbufs_destroy :: proc(gbufs: ^GBuffers)
{
    gpu.texture_free_and_destroy(&gbufs.world_pos)
    gpu.texture_free_and_destroy(&gbufs.world_normals)
    gbufs^ = {}
}

gbufs_render :: proc(cmd_buf: gpu.Command_Buffer, gbufs: ^GBuffers, shaders: Shaders, scene: Scene)
{
    gpu.cmd_scoped_render_pass(cmd_buf, {
        color_attachments = {
            { texture = gbufs.world_pos, clear_color = { 0, 0, 0, 1 } },
            { texture = gbufs.world_normals, clear_color = { 0, 0, 0, 1 } }
        }
    })

    gpu.cmd_set_shaders(cmd_buf, shaders.uv_space, shaders.gbuffers)

    // Render the entire scene
    for instance in scene.instances
    {
        mesh := scene.meshes[instance.mesh_idx]

        Vertex_Data :: struct {
            // scene: Scene_Mesh,
        }
    }
}

pathtrace :: proc(cmd_buf: gpu.Command_Buffer, output: gpu.Texture, shaders: Shaders)
{
    // gpu.cmd_set_compute_shader(cmd_buf,
}
