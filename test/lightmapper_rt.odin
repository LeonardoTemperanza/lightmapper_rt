
#+feature using-stmt

package test

import "../no_gfx_api/gpu"
import "core:math/linalg"
import "core:math"
import intr "base:intrinsics"
import "core:slice"

Handle :: struct { idx: u32, gen: u32 }
Mesh_Handle :: distinct Handle
Lightmap_UV_Handle :: distinct Handle

Context :: struct
{
    shaders: Shaders,

    desc_pool: ^gpu.Descriptor_Pool,

    // Upload resources
    bvh_scratch_arena: gpu.Arena,  // GPU local
    upload_arena: gpu.Arena,  // CPU mapped

    // Global resources
    meshes: [dynamic]Mesh,
    lm_uvs: [dynamic]gpu.slice_t([2]f32),
}

init :: proc(desc_pool: ^gpu.Descriptor_Pool) -> Context
{
    ctx: Context
    ctx.shaders = shaders_create()
    ctx.bvh_scratch_arena = gpu.arena_init(mem_type = gpu.Memory.GPU)
    ctx.upload_arena = gpu.arena_init()
    ctx.desc_pool = desc_pool
    return ctx
}

cleanup :: proc(ctx: ^Context)
{
    shaders_destroy(&ctx.shaders)
    gpu.arena_destroy(&ctx.bvh_scratch_arena)
    gpu.arena_destroy(&ctx.upload_arena)
    ctx^ = {}
}

Mesh_Desc :: struct
{
    // TODO: Do we need these here?
    // Temporary, can be changed/freed after this call.
    positions_cpu: [][3]f32,
    normals_cpu:   [][3]f32,
    uvs_cpu:       [][2]f32,
    indices_cpu:   []u32,

    // Must stay alive until the removal of this Mesh_Handle.
    positions_gpu: gpu.slice_t([3]f32),
    normals_gpu:   gpu.slice_t([3]f32),
    uvs_gpu:       gpu.slice_t([2]f32),
    indices_gpu:   gpu.slice_t(u32),
}

add_mesh :: proc(using ctx: ^Context, cmd_buf: gpu.Command_Buffer, desc: Mesh_Desc) -> Mesh_Handle
{
    bvh := build_blas(&bvh_scratch_arena,
                      cmd_buf,
                      desc.positions_gpu,
                      desc.indices_gpu,
                      u32(len(desc.indices_cpu)),
                      u32(len(desc.positions_cpu)))

    append(&meshes, Mesh {
        positions = desc.positions_gpu,
        normals   = desc.normals_gpu,
        uvs       = desc.uvs_gpu,
        indices   = desc.indices_gpu,
        bvh       = bvh,
    })

    return Mesh_Handle { idx = u32(len(meshes) - 1), gen = 0 }
}

remove_mesh :: proc(ctx: ^Context, handle: ^Mesh_Handle)
{

}

Lightmap_UVs_Desc :: struct
{
    // Temporary, can be changed/freed after this call.
    positions_cpu: [][3]f32,
    normals_cpu:   [][3]f32,
    uvs_cpu:       [][2]f32,
    lm_uvs_cpu:    [][2]f32,

    // Must stay alive until the removal of this Lightmap_UV_Handle.
    lm_uvs_gpu:    gpu.slice_t([2]f32),
}

add_lightmap_uvs :: proc(ctx: ^Context, desc: Lightmap_UVs_Desc) -> Lightmap_UV_Handle
{
    append(&ctx.lm_uvs, desc.lm_uvs_gpu)
    return {}
}

remove_lightmap_uvs :: proc(ctx: ^Context, handle: ^Lightmap_UV_Handle)
{

}

Bake :: struct
{
    ctx: ^Context,
    gbufs: GBuffers,
    instances: [dynamic]Instance,
    scene_gpu: Scene_GPU,
}

Instance :: struct
{
    mesh_handle: Mesh_Handle,
    // You might want different instances of the same mesh to have a completely different set of UVs.
    lm_uvs_handle: Lightmap_UV_Handle,
    transform: matrix[4, 4]f32,
    lm_uvs_offset: [2]f32,
    lm_uvs_scale: [2]f32,

    // Material properties
    albedo_tex_id: u32,
    albedo: [3]f32,
}

bake_begin :: proc(ctx: ^Context, #any_int lightmap_size: i64, instances: []Instance) -> Bake
{
    bake: Bake
    bake.ctx = ctx
    bake.gbufs = gbufs_create(lightmap_size)
    bake.instances = slice.clone_to_dynamic(instances)

    cmd_buf := gpu.commands_begin(.Main)

    meshes_gpu := gpu.arena_alloc(&ctx.upload_arena, Mesh_Shader, len(ctx.meshes))
    for &mesh, i in meshes_gpu.cpu {
        mesh.positions = ctx.meshes[i].positions.gpu.ptr
        mesh.normals   = ctx.meshes[i].normals.gpu.ptr
        mesh.uvs       = ctx.meshes[i].uvs.gpu.ptr
        mesh.indices   = ctx.meshes[i].indices.gpu.ptr
    }
    bake.scene_gpu.meshes_shader = gpu.mem_alloc(Mesh_Shader, len(ctx.meshes), gpu.Memory.GPU)
    gpu.cmd_mem_copy(cmd_buf, bake.scene_gpu.meshes_shader, meshes_gpu, len(ctx.meshes))

    instances_gpu := gpu.arena_alloc(&ctx.upload_arena, Instance_Shader, len(instances))
    for &instance, i in instances_gpu.cpu {
        instance = {
            mesh_idx = instances[i].mesh_handle.idx,
            albedo_tex_id = instances[i].albedo_tex_id,
        }
    }
    bake.scene_gpu.instances = gpu.mem_alloc(Instance_Shader, len(instances), gpu.Memory.GPU)
    gpu.cmd_mem_copy(cmd_buf, bake.scene_gpu.instances, instances_gpu, len(instances))
    gpu.cmd_barrier(cmd_buf, .All, .All)

    bake.scene_gpu.instances_bvh = upload_bvh_instances(&ctx.upload_arena, cmd_buf, instances, ctx.meshes[:])
    gpu.cmd_barrier(cmd_buf, .Transfer, .Build_BVH)
    bake.scene_gpu.bvh = build_tlas(&ctx.upload_arena, cmd_buf, bake.scene_gpu.instances_bvh, u32(len(instances)))
    gpu.cmd_barrier(cmd_buf, .Build_BVH, .All)

    bake.scene_gpu.bvh_id = gpu.desc_pool_alloc_bvh(ctx.desc_pool, gpu.bvh_descriptor(bake.scene_gpu.bvh))

    gbufs_render(cmd_buf, &ctx.upload_arena, &bake.gbufs, ctx.shaders, instances, ctx.meshes[:], ctx.lm_uvs[:])
    gpu.cmd_barrier(cmd_buf, .All, .All, {})

    gpu.queue_submit(.Main, { cmd_buf })
    return bake
}

bake_scene_changed :: proc(bake: ^Bake, instances: []Instance) -> bool
{
    return false
}

bake_reset :: proc(bake: ^Bake)
{

}

bake_iteration :: proc(bake: ^Bake)
{
    cmd_buf := gpu.commands_begin(.Main)
    // pathtrace(cmd_buf, bake.gbufs, bake.shaders, bake.scene)
    gpu.queue_submit(.Main, { cmd_buf })
}

bake_get_gbuffer_world_pos :: proc(bake: ^Bake) -> gpu.Texture
{
    return bake.gbufs.world_pos
}

bake_get_gbuffer_world_normals :: proc(bake: ^Bake) -> gpu.Texture
{
    return bake.gbufs.world_normals
}

bake_debug_ground_truth :: proc(bake: ^Bake, cmd_buf: gpu.Command_Buffer, frame_arena: ^gpu.Arena, camera_to_world: matrix[4, 4]f32, texture_rw_id: u32, sampler_id: u32, resolution: [2]f32)
{
    Compute_Data :: struct #all_or_none {
        output_texture_id: u32,
        tlas_id: u32,
        linear_sampler: u32,
        scene: Scene_Shader,
        resolution: [2]f32,
        accum_counter: u32,
        camera_to_world: [16]f32,
    }

    compute_data := gpu.arena_alloc(frame_arena, Compute_Data)
    compute_data.cpu^ = {
        output_texture_id = texture_rw_id,
        tlas_id = bake.scene_gpu.bvh_id,
        linear_sampler = sampler_id,
        scene = {
            instances = bake.scene_gpu.instances.gpu.ptr,
            meshes = bake.scene_gpu.meshes_shader.gpu.ptr,
            lights = {
                dir_light_dir   = linalg.normalize([3]f32 { 0.2, -1.0, -0.2 }),
                dir_light_angle = math.RAD_PER_DEG * 0.2,
                dir_light_emission = [3]f32 { 2000000.0, 1840000.0, 1640000.0 },
            }
        },
        accum_counter = 0,
        resolution = resolution,
        camera_to_world = intr.matrix_flatten(camera_to_world),
    }

    gpu.cmd_set_compute_shader(cmd_buf, bake.ctx.shaders.pathtrace)

    num_groups_x := (u32(resolution.x) + 8 - 1) / 8
    num_groups_y := (u32(resolution.y) + 8 - 1) / 8
    num_groups_z := u32(1)
    gpu.cmd_dispatch(cmd_buf, compute_data.gpu, num_groups_x, num_groups_y, num_groups_z)

    gpu.cmd_barrier(cmd_buf, .Compute, .Fragment_Shader, {})
    gpu.cmd_barrier(cmd_buf, .Compute, .Compute, {})
}

bake_end :: proc(bake: ^Bake)
{

}

bake_destroy :: proc(bake: ^Bake)
{
    gbufs_destroy(&bake.gbufs)
    bake^ = {}
}

// internal

Scene_GPU :: struct #all_or_none
{
    bvh: gpu.Owned_BVH,
    bvh_id: u32,
    instances_bvh: gpu.slice_t(gpu.BVH_Instance),

    // Shader view
    instances: gpu.slice_t(Instance_Shader),
    meshes_shader: gpu.slice_t(Mesh_Shader),
}

Scene_Shader :: struct
{
    instances: rawptr,
    meshes: rawptr,
    lights: Lights_Shader,
}

Lights_Shader :: struct
{
    dir_light_dir: [3]f32,
    dir_light_angle: f32,
    dir_light_emission: [3]f32,
}

Mesh :: struct
{
    positions: gpu.slice_t([3]f32),
    normals: gpu.slice_t([3]f32),
    uvs: gpu.slice_t([2]f32),
    indices: gpu.slice_t(u32),
    bvh: gpu.Owned_BVH,
}

Instance_Shader :: struct
{
    mesh_idx: u32,
    albedo_tex_id: u32,
}

Mesh_Shader :: struct
{
    positions: rawptr,
    normals: rawptr,
    uvs: rawptr,
    lm_uvs: rawptr,
    indices: rawptr,
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

shaders_destroy :: proc(shaders: ^Shaders)
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
        usage = { .Color_Attachment, .Sampled, .Storage },
    })
    gbufs.world_normals = gpu.texture_alloc_and_create({
        dimensions = { u32(lightmap_size), u32(lightmap_size), 1 },
        format = .RGBA8_Unorm,
        usage = { .Color_Attachment, .Sampled, .Storage }
    })
    return gbufs
}

gbufs_destroy :: proc(gbufs: ^GBuffers)
{
    gpu.texture_free_and_destroy(&gbufs.world_pos)
    gpu.texture_free_and_destroy(&gbufs.world_normals)
    gbufs^ = {}
}

gbufs_render :: proc(cmd_buf: gpu.Command_Buffer, upload_arena: ^gpu.Arena, gbufs: ^GBuffers, shaders: Shaders, instances: []Instance, meshes: []Mesh, lm_uvs: []gpu.slice_t([2]f32))
{
    gpu.cmd_scoped_render_pass(cmd_buf, {
        color_attachments = {
            { texture = gbufs.world_pos, clear_color = { 0, 0, 0, 1 } },
            { texture = gbufs.world_normals, clear_color = { 0, 0, 0, 1 } }
        }
    })

    gpu.cmd_set_shaders(cmd_buf, shaders.uv_space, shaders.gbuffers)

    // Render the entire scene
    for instance in instances
    {
        mesh := meshes[instance.mesh_handle.idx]
        lightmap_uvs := lm_uvs[instance.lm_uvs_handle.idx]

        Vertex_Data :: struct #all_or_none {
            pos: rawptr,
            normals: rawptr,
            uvs: rawptr,
            lightmap_uvs: rawptr,
            model_to_world: [16]f32,
            model_to_world_normals: [16]f32,
        }
        vert_data := gpu.arena_alloc(upload_arena, Vertex_Data)
        vert_data.cpu^ = Vertex_Data {
            pos = mesh.positions.gpu.ptr,
            normals = mesh.normals.gpu.ptr,
            uvs = mesh.normals.gpu.ptr,
            lightmap_uvs = lightmap_uvs.gpu.ptr,
            model_to_world = intr.matrix_flatten(instance.transform),
            model_to_world_normals = intr.matrix_flatten(linalg.transpose(linalg.inverse(instance.transform))),
        }


        gpu.cmd_draw_indexed_instanced(cmd_buf, vert_data, {}, mesh.indices, u32(len(mesh.indices.cpu)))
    }
}

pathtrace :: proc(cmd_buf: gpu.Command_Buffer, output: gpu.Texture, shaders: Shaders)
{
    // gpu.cmd_set_compute_shader(cmd_buf,
}

build_blas :: proc(bvh_scratch_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer, positions: gpu.slice_t([3]f32), indices: gpu.slice_t(u32), idx_count: u32, vert_count: u32) -> gpu.Owned_BVH
{
    assert(idx_count % 3 == 0)

    desc := gpu.BLAS_Desc {
        hint = .Prefer_Fast_Trace,
        shapes = {
            gpu.BVH_Mesh_Desc {
                vertex_stride = size_of(positions.cpu[0]),
                max_vertex = vert_count - 1,
                tri_count = idx_count / 3,
            }
        }
    }
    bvh := gpu.bvh_alloc_and_create(desc)
    scratch := gpu.bvh_alloc_build_scratch_buffer(bvh_scratch_arena, desc)
    gpu.cmd_build_blas(cmd_buf, bvh, bvh.mem, scratch, { gpu.BVH_Mesh { verts = positions.gpu.ptr, indices = indices.gpu.ptr } })
    return bvh
}

build_tlas :: proc(bvh_scratch_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer, instances: gpu.gpuptr, instance_count: u32) -> gpu.Owned_BVH
{
    desc := gpu.TLAS_Desc {
        hint = .Prefer_Fast_Trace,
        instance_count = instance_count
    }
    bvh := gpu.bvh_alloc_and_create(desc)
    scratch := gpu.bvh_alloc_build_scratch_buffer(bvh_scratch_arena, desc)
    gpu.cmd_build_tlas(cmd_buf, bvh, bvh.mem, scratch, instances)
    return bvh
}

upload_bvh_instances :: proc(upload_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer, instances: []Instance, meshes: []Mesh) -> gpu.slice_t(gpu.BVH_Instance)
{
    instances_staging := gpu.arena_alloc(upload_arena, gpu.BVH_Instance, len(instances))
    for &instance, i in instances_staging.cpu
    {
        instance = {
            transform = transform_to_gpu_transform(instances[i].transform),
            blas_root = gpu.bvh_root_ptr(meshes[instances[i].mesh_handle.idx].bvh),
            disable_culling = true,
            flip_facing = true,
            mask = 1,
        }
    }
    instances_local := gpu.mem_alloc(gpu.BVH_Instance, len(instances), mem_type = gpu.Memory.GPU)
    gpu.cmd_mem_copy(cmd_buf, instances_local, instances_staging, len(instances))
    return instances_local
}

transform_to_gpu_transform :: proc(transform: matrix[4, 4]f32) -> [12]f32
{
    transform_row_major := intr.transpose(transform)
    flattened := linalg.matrix_flatten(transform_row_major)
    return [12]f32 { flattened[0], flattened[1], flattened[2], flattened[3], flattened[4], flattened[5], flattened[6], flattened[7], flattened[8], flattened[9], flattened[10], flattened[11], }
}
