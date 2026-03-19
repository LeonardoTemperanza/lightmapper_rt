
package main

import intr "base:intrinsics"
import "base:runtime"
import "core:container/queue"
import "core:fmt"
import "core:image"
import "core:image/jpeg"
import "core:image/png"
import log "core:log"
import "core:math"
import "core:math/linalg"
import "core:strings"
import "core:sync"
import "core:thread"
import "core:sys/info"
import "core:time"

import sdl "vendor:sdl3"

import shared "shared"
import gltf2 "shared/gltf2"

import vk "vendor:vulkan"
import "no_gfx_api/gpu"

import imgui "odin-imgui"
import imgui_impl_sdl3 "odin-imgui/imgui_impl_sdl3"
import imgui_impl_nogfx "odin-imgui/imgui_impl_nogfx"

import lm "test"

Frames_In_Flight :: 3
Example_Name_Format :: "Right-click + WASD for first-person controls. Left click to toggle texture type. Current: %v"

Sponza_Scene :: #load("shared/assets/sponza.glb")

Num_Async_Worker_Threads := clamp(info.cpu.logical_cores - 1, 2, 6)

// How many textures to load in a single batch / command buffer
Loader_Chunk_Size :: 16

// Index used for the color target texture
COLOR_TARGET_IDX: u32 = 0
COLOR_TARGET_RW_IDX: u32 = 0

// Textures can be loaded/unloaded on different threads, so we need to synchronize access to loaded_textures, image_to_texture and image_uploaded
mutex: sync.Mutex
// Every texture from loaded_textures array needs to be freed when we are done
loaded_textures: [dynamic]gpu.Owned_Texture
// Enables asynchronous cancellation of texture loading
cancel_loading_textures: bool
next_texture_idx: u32 = shared.MISSING_TEXTURE_ID + 1
// Cache for image_index -> texture mapping, reused across texture loading chunks
image_to_texture: map[int]struct {
    texture:     gpu.Owned_Texture,
    texture_idx: u32,
}
image_uploaded: map[int]^sync.One_Shot_Event

upload_sem: gpu.Semaphore
upload_sem_val: u64

main :: proc()
{
    ok_i := sdl.Init({.VIDEO})
    assert(ok_i)

    console_logger := log.create_console_logger()
    defer log.destroy_console_logger(console_logger)
    context.logger = console_logger

    ts_freq := sdl.GetPerformanceFrequency()
    max_delta_time: f32 = 1.0 / 10.0 // 10fps

    window_flags :: sdl.WindowFlags { .HIGH_PIXEL_DENSITY, .VULKAN, .RESIZABLE, .MAXIMIZED }
    window := sdl.CreateWindow("test_lightmapper", 1000, 1000, window_flags)
    ensure(window != nil)

    window_size_x: i32
    window_size_y: i32
    sdl.GetWindowSize(window, &window_size_x, &window_size_y)

    ok := gpu.init()
    ensure(ok)
    defer gpu.cleanup()

    gpu.swapchain_init_from_sdl(window, Frames_In_Flight)

    pathtrace_shader := gpu.shader_create_compute(#load("shaders/pathtrace.comp.spv", []u32), 8, 8, 1)
    defer gpu.shader_destroy(pathtrace_shader)

    vert_shader_lit := gpu.shader_create(#load("shaders/lit.vert.spv", []u32), .Vertex)
    frag_shader_lit := gpu.shader_create(#load("shaders/lit.frag.spv", []u32), .Fragment)
    defer {
        gpu.shader_destroy(vert_shader_lit)
        gpu.shader_destroy(frag_shader_lit)
    }

    vert_shader_tonemap := gpu.shader_create(#load("shaders/tonemap.vert.spv", []u32), .Vertex)
    frag_shader_tonemap := gpu.shader_create(#load("shaders/tonemap.frag.spv", []u32), .Fragment)
    defer {
        gpu.shader_destroy(vert_shader_tonemap)
        gpu.shader_destroy(frag_shader_tonemap)
    }

    upload_arena := gpu.arena_init()
    defer gpu.arena_destroy(&upload_arena)
    bvh_scratch_arena := gpu.arena_init(mem_type = .GPU)
    defer gpu.arena_destroy(&bvh_scratch_arena)

    upload_sem = gpu.semaphore_create()
    defer gpu.semaphore_destroy(upload_sem)

    upload_cmd_buf := gpu.commands_begin(.Main)

    fsq_verts, fsq_indices := create_fullscreen_quad(
        &upload_arena,
        upload_cmd_buf,
    )
    defer {
        gpu.mem_free(fsq_verts)
        gpu.mem_free(fsq_indices)
    }

    desc_pool := gpu.desc_pool_create()
    defer gpu.desc_pool_destroy(&desc_pool)

    gltf_scene, texture_infos, gltf_data := shared.load_scene_gltf(Sponza_Scene)
    defer {
        shared.destroy_scene(&gltf_scene)
        gltf2.unload(gltf_data)
    }

    defer {
        // Clean up loaded textures
        sync.guard(&mutex)
        for &tex in loaded_textures {
            gpu.texture_free_and_destroy(&tex)
        }
    }

    // Spawn and wait for loading threads
    {
        worker_threads: [dynamic]^thread.Thread
        defer {
            cancel_loading_textures = true
            for t in worker_threads {
                thread.terminate(t, 0)
            }
        }

        Texture_Loader_Data :: struct {
            texture_infos: []shared.Gltf_Texture_Info,
            gltf_data:     ^gltf2.Data,
            scene:         ^shared.Scene,
            desc_pool:     ^gpu.Descriptor_Pool,
            logger:        log.Logger,
            current_chunk: ^int,
        }
        loader_data := Texture_Loader_Data {
            texture_infos = texture_infos,
            gltf_data     = gltf_data,
            scene         = &gltf_scene,
            desc_pool     = &desc_pool,
            logger        = console_logger,
            current_chunk = new(int),
        }

        texture_loader_thread_proc :: proc(thread: ^thread.Thread) {
            data := cast(^Texture_Loader_Data)thread.data
            context.logger = data.logger

            for !cancel_loading_textures {
                current_chunk_start := sync.atomic_add(data.current_chunk, Loader_Chunk_Size)
                current_chunk_end := min(current_chunk_start + Loader_Chunk_Size, len(data.texture_infos))

                if current_chunk_start >= len(data.texture_infos) {
                    break
                }

                log.debugf("Creating texture loader for chunk %v of %v", current_chunk_start, len(data.texture_infos))

                load_scene_textures_from_gltf(
                    data.texture_infos[current_chunk_start:current_chunk_end],
                    data.gltf_data,
                    data.scene,
                    data.desc_pool,
                )
            }
        }

        for i := 0; i < Num_Async_Worker_Threads; i += 1 {
            texture_loader_thread := thread.create(texture_loader_thread_proc)
            texture_loader_thread.data = &loader_data
            thread.start(texture_loader_thread)
            append(&worker_threads, texture_loader_thread)
        }

        for worker_thread in worker_threads {
            thread.join(worker_thread)
        }
    }

    scene := upload_scene(gltf_scene, &upload_arena, &bvh_scratch_arena, upload_cmd_buf)
    defer scene_destroy(&scene)

    max_anisotropy := min(16.0, gpu.device_limits().max_anisotropy)
    // Lightmap samplers
    sampler_linear_id  := gpu.desc_pool_alloc_sampler(&desc_pool, gpu.sampler_descriptor({ max_anisotropy = min(16.0, gpu.device_limits().max_anisotropy) }))
    sampler_point_id   := gpu.desc_pool_alloc_sampler(&desc_pool, gpu.sampler_descriptor({ min_filter = .Nearest, mag_filter = .Nearest, mip_filter = .Nearest }))
    sampler_current_id := sampler_linear_id

    bvh_id := gpu.desc_pool_alloc_bvh(&desc_pool, gpu.bvh_descriptor(scene.bvh))

    color_target, depth_target := create_target_textures(u32(window_size_x), u32(window_size_y), &desc_pool)
    defer {
        gpu.texture_free_and_destroy(&color_target)
        gpu.texture_free_and_destroy(&depth_target)
    }

    gpu.cmd_barrier(upload_cmd_buf, .Transfer, .All, {})
    gpu.queue_submit(.Main, {upload_cmd_buf})

    imgui_ctx := init_dear_imgui(window, &desc_pool)
    defer {
        imgui_impl_nogfx.shutdown()
        imgui_impl_sdl3.shutdown()
        imgui.destroy_context(imgui_ctx)
    }

    // lm.bake_begin(4096)

    now_ts := sdl.GetPerformanceCounter()

    pathtrace_gt_counter: u32
    max_gt_accums := u32(1000)
    lm_build_counter: u32
    output_type: Output_Type

    frame_arenas: [Frames_In_Flight]gpu.Arena
    for &frame_arena in frame_arenas do frame_arena = gpu.arena_init()
    defer for &frame_arena in frame_arenas do gpu.arena_destroy(&frame_arena)
    next_frame := u64(1)
    frame_sem := gpu.semaphore_create(0)
    defer gpu.semaphore_destroy(frame_sem)
    for true
    {
        proceed := handle_window_events(window)
        if !proceed do break

        old_window_size_x := window_size_x
        old_window_size_y := window_size_y
        sdl.GetWindowSize(window, &window_size_x, &window_size_y)
        if .MINIMIZED in sdl.GetWindowFlags(window) || window_size_x <= 0 || window_size_y <= 0
        {
            sdl.Delay(16)
            continue
        }

        if next_frame > Frames_In_Flight {
            gpu.semaphore_wait(frame_sem, next_frame - Frames_In_Flight)
        }
        if old_window_size_x != window_size_x || old_window_size_y != window_size_y
        {
            gpu.queue_wait_idle(.Main)
            gpu.swapchain_resize({u32(max(0, window_size_x)), u32(max(0, window_size_y))})

            gpu.texture_free_and_destroy(&color_target)
            gpu.texture_free_and_destroy(&depth_target)
            color_target, depth_target = create_target_textures(u32(window_size_x), u32(window_size_y), &desc_pool)
        }

        if shared.INPUT.pressing_right_click do pathtrace_gt_counter = 0

        imgui_impl_sdl3.new_frame()
        imgui_impl_nogfx.new_frame()
        imgui.new_frame()

        mask_out_dear_imgui_inputs()

        @(static) show_texture_viewer: bool = true
        @(static) show_settings: bool = true
        @(static) show_demo_window: bool = false

        if imgui.begin_main_menu_bar()
        {
            if imgui.begin_menu("Windows")
            {
                imgui.menu_item_bool_ptr("Lightmap Viewer", nil, &show_texture_viewer)
                imgui.menu_item_bool_ptr("Settings", nil, &show_settings)
                imgui.menu_item_bool_ptr("Dear ImGui Demo", nil, &show_demo_window)
                imgui.end_menu()
            }
            imgui.end_main_menu_bar()
        }

        if show_demo_window {
            imgui.show_demo_window(&show_demo_window)
        }

        if show_settings && imgui.begin("Settings", &show_settings)
        {
            // Lightmap sampling
            {
                items := []cstring { "Point", "Bilinear", "Bicubic" }
                @(static) item_selected_idx: int = 1

                combo_preview_value := items[item_selected_idx]
                if imgui.begin_combo("Lightmap Sampling", combo_preview_value, {})
                {
                    for n := 0; n < len(items); n += 1
                    {
                        is_selected := item_selected_idx == n
                        if imgui.selectable(items[n], is_selected) {
                            item_selected_idx = n
                            switch Sampler_Type(n)
                            {
                                case .Point:    sampler_current_id = sampler_point_id
                                case .Bilinear: sampler_current_id = sampler_linear_id
                                case .Bicubic:  sampler_current_id = sampler_current_id
                            }
                        }

                        if is_selected {
                            imgui.set_item_default_focus()
                        }
                    }
                    imgui.end_combo()
                }
            }

            // Output type
            {
                items := []cstring { "Default", "Ground Truth" }
                @(static) item_selected_idx: int = 0
                combo_preview_value := items[item_selected_idx]
                if imgui.begin_combo("Output", combo_preview_value, {})
                {
                    for n := 0; n < len(items); n += 1
                    {
                        is_selected := item_selected_idx == n
                        if imgui.selectable(items[n], is_selected)
                        {
                            item_selected_idx = n
                            output_type = Output_Type(n)
                        }

                        if is_selected {
                            imgui.set_item_default_focus()
                        }
                    }
                    imgui.end_combo()
                }
            }
        }
        imgui.end()

        if show_texture_viewer {
            gui_show_debug_texture_window("Lightmap Viewer", 0, 1024, 1024, &show_texture_viewer)
        }

        imgui.render()

        swapchain := gpu.swapchain_acquire_next() // Blocks CPU until at least one frame is available.

        last_ts := now_ts
        now_ts = sdl.GetPerformanceCounter()
        delta_time := min(max_delta_time, f32(f64((now_ts - last_ts) * 1000) / f64(ts_freq)) / 1000.0)

        world_to_view := shared.first_person_camera_view(delta_time)
        aspect_ratio := f32(window_size_x) / f32(window_size_y)
        view_to_proj := linalg.matrix4_perspective_f32(math.RAD_PER_DEG * 59.0, aspect_ratio, 0.1, 1000.0, false)

        frame_arena := &frame_arenas[next_frame % Frames_In_Flight]
        gpu.arena_free_all(frame_arena)

        cmd_buf := gpu.commands_begin(.Main)

        gpu.cmd_set_desc_pool(cmd_buf, desc_pool)

        switch output_type
        {
            case .Rasterized:
            {
                pathtrace_gt_counter = 0

                gpu.cmd_begin_render_pass(cmd_buf, {
                    color_attachments = {
                        { texture = color_target, clear_color = {0.0, 0.0, 0.0, 1.0} },
                    },
                    depth_attachment = gpu.Render_Attachment { texture = depth_target, clear_color = 1.0 },
                })
                gpu.cmd_set_shaders(cmd_buf, vert_shader_lit, frag_shader_lit)

                // Set texture and sampler heaps
                gpu.cmd_set_desc_pool(cmd_buf, desc_pool)

                gpu.cmd_set_depth_state(cmd_buf, {mode = {.Read, .Write}, compare = .Less})

                for instance in gltf_scene.instances
                {
                    mesh := scene.meshes[instance.mesh_idx]
                    base_color_map := gltf_scene.meshes[instance.mesh_idx].base_color_map
                    metallic_roughness_map := gltf_scene.meshes[instance.mesh_idx].metallic_roughness_map
                    normal_map := gltf_scene.meshes[instance.mesh_idx].normal_map

                    Vert_Data :: struct #all_or_none {
                        positions:             rawptr,
                        normals:               rawptr,
                        uvs:                   rawptr,
                        model_to_world:        [16]f32,
                        model_to_world_normal: [16]f32,
                        world_to_view:         [16]f32,
                        view_to_proj:          [16]f32,
                    }
                    verts_data := gpu.arena_alloc(frame_arena, Vert_Data)
                    verts_data.cpu^ = {
                        positions             = mesh.pos.gpu.ptr,
                        normals               = mesh.normals.gpu.ptr,
                        uvs                   = mesh.uvs.gpu.ptr,
                        model_to_world        = intr.matrix_flatten(instance.transform),
                        model_to_world_normal = intr.matrix_flatten(linalg.transpose(linalg.inverse(instance.transform))),
                        world_to_view         = intr.matrix_flatten(world_to_view),
                        view_to_proj          = intr.matrix_flatten(view_to_proj),
                    }

                    Frag_Data :: struct #all_or_none {
                        base_color_map:                 u32,
                        base_color_map_sampler:         u32,
                        metallic_roughness_map:         u32,
                        metallic_roughness_map_sampler: u32,
                        normal_map:                     u32,
                        normal_map_sampler:             u32,
                    }
                    frag_data := gpu.arena_alloc(frame_arena, Frag_Data)
                    frag_data.cpu^ = {
                        base_color_map                 = base_color_map,
                        base_color_map_sampler         = sampler_current_id,
                        metallic_roughness_map         = metallic_roughness_map,
                        metallic_roughness_map_sampler = 0,
                        normal_map                     = normal_map,
                        normal_map_sampler             = 0,
                    }

                    gpu.cmd_draw_indexed_instanced(cmd_buf, verts_data.gpu, frag_data.gpu, mesh.indices, mesh.idx_count, 1)
                }

                gpu.cmd_end_render_pass(cmd_buf)
                gpu.cmd_barrier(cmd_buf, .Raster_Color_Out, .Fragment_Shader, {})
            }
            case .Pathtraced:
            {
                if pathtrace_gt_counter < max_gt_accums
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

                    camera_to_world := linalg.inverse(world_to_view)

                    compute_data := gpu.arena_alloc(frame_arena, Compute_Data)
                    compute_data.cpu^ = {
                        output_texture_id = COLOR_TARGET_RW_IDX,
                        tlas_id = bvh_id,
                        linear_sampler = sampler_linear_id,
                        scene = { instances = scene.instances.gpu.ptr, meshes = scene.meshes_shader.gpu.ptr },
                        accum_counter = pathtrace_gt_counter,
                        resolution = { f32(window_size_x), f32(window_size_y) },
                        camera_to_world = intr.matrix_flatten(camera_to_world),
                    }

                    gpu.cmd_set_compute_shader(cmd_buf, pathtrace_shader)

                    num_groups_x := (u32(window_size_x) + 8 - 1) / 8
                    num_groups_y := (u32(window_size_y) + 8 - 1) / 8
                    num_groups_z := u32(1)
                    gpu.cmd_dispatch(cmd_buf, compute_data.gpu, num_groups_x, num_groups_y, num_groups_z)

                    gpu.cmd_barrier(cmd_buf, .Compute, .Fragment_Shader, {})

                    pathtrace_gt_counter += 1
                }
            }
        }

        // Tonemap
        {
            gpu.cmd_begin_render_pass(
                cmd_buf,
                {color_attachments = {{texture = swapchain, clear_color = {0.7, 0.7, 0.7, 1.0}}}},
            )
            gpu.cmd_set_shaders(cmd_buf, vert_shader_tonemap, frag_shader_tonemap)

            // Set texture and sampler heaps
            gpu.cmd_set_desc_pool(cmd_buf, desc_pool)

            // Disable depth testing for fullscreen quad
            gpu.cmd_set_depth_state(cmd_buf, { mode = {}, compare = .Always })

            // Vertex data for fullscreen quad
            Vert_Data :: struct #all_or_none {
                verts: rawptr,
            }
            verts_data := gpu.arena_alloc(frame_arena, Vert_Data)
            verts_data.cpu.verts = fsq_verts.gpu.ptr

            // Fragment data with all G-buffer textures and selected texture type
            Frag_Data :: struct #all_or_none {
                texture_id: u32,
                sampler_id: u32,
            }
            frag_data := gpu.arena_alloc(frame_arena, Frag_Data)
            frag_data.cpu^ = {
                texture_id = COLOR_TARGET_IDX,
                sampler_id = sampler_linear_id,
            }

            // Render fullscreen quad
            gpu.cmd_draw_indexed_instanced(cmd_buf, verts_data.gpu, frag_data.gpu, fsq_indices, 6, 1)

            // Render ImGui on top
            draw_data := imgui.get_draw_data()
            if draw_data != nil && draw_data.cmd_lists_count > 0 {
                imgui_impl_nogfx.render_draw_data(draw_data, cmd_buf)
            }

            gpu.cmd_end_render_pass(cmd_buf)
        }

        gpu.cmd_add_signal_semaphore(cmd_buf, frame_sem, next_frame)
        gpu.queue_submit(.Main, {cmd_buf})

        gpu.swapchain_present(.Main, frame_sem, next_frame)
        next_frame += 1
    }

    gpu.wait_idle()
}

Mesh_GPU :: struct
{
    pos: gpu.slice_t([4]f32),
    normals: gpu.slice_t([4]f32),
    uvs: gpu.slice_t([2]f32),
    indices: gpu.slice_t(u32),
    idx_count: u32,
    vert_count: u32,
    bvh: gpu.Owned_BVH,
}

Mesh_Shader :: struct
{
    pos: rawptr,
    normals: rawptr,
    uvs: rawptr,
    indices: rawptr,
}

upload_mesh :: proc(upload_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer, mesh: shared.Mesh) -> Mesh_GPU
{
    assert(len(mesh.pos) == len(mesh.normals))
    assert(len(mesh.pos) == len(mesh.uvs))

    positions_staging := gpu.arena_alloc(upload_arena, [4]f32, len(mesh.pos))
    normals_staging := gpu.arena_alloc(upload_arena, [4]f32, len(mesh.normals))
    uvs_staging := gpu.arena_alloc(upload_arena, [2]f32, len(mesh.uvs))
    indices_staging := gpu.arena_alloc(upload_arena, u32, len(mesh.indices))
    copy(positions_staging.cpu, mesh.pos[:])
    copy(normals_staging.cpu, mesh.normals[:])
    copy(uvs_staging.cpu, mesh.uvs[:])
    copy(indices_staging.cpu, mesh.indices[:])

    res: Mesh_GPU
    res.pos = gpu.mem_alloc([4]f32, len(mesh.pos), mem_type = gpu.Memory.GPU)
    res.normals = gpu.mem_alloc([4]f32, len(mesh.normals), mem_type = gpu.Memory.GPU)
    res.uvs = gpu.mem_alloc([2]f32, len(mesh.uvs), mem_type = gpu.Memory.GPU)
    res.indices = gpu.mem_alloc(u32, len(mesh.indices), mem_type = gpu.Memory.GPU)
    gpu.cmd_mem_copy(cmd_buf, res.pos, positions_staging, len(mesh.pos))
    gpu.cmd_mem_copy(cmd_buf, res.normals, normals_staging, len(mesh.normals))
    gpu.cmd_mem_copy(cmd_buf, res.uvs, uvs_staging, len(mesh.uvs))
    gpu.cmd_mem_copy(cmd_buf, res.indices, indices_staging, len(mesh.indices))

    res.idx_count = u32(len(mesh.indices))
    res.vert_count = u32(len(mesh.pos))
    return res
}

mesh_destroy :: proc(mesh: ^Mesh_GPU)
{
    gpu.bvh_free_and_destroy(&mesh.bvh)
    gpu.mem_free(mesh.pos)
    gpu.mem_free(mesh.normals)
    gpu.mem_free(mesh.uvs)
    gpu.mem_free(mesh.indices)
    mesh^ = {}
}

build_blas :: proc(bvh_scratch_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer, positions: gpu.slice_t([4]f32), indices: gpu.slice_t(u32), idx_count: u32, vert_count: u32) -> gpu.Owned_BVH
{
    assert(idx_count % 3 == 0)

    desc := gpu.BLAS_Desc {
        hint = .Prefer_Fast_Trace,
        shapes = {
            gpu.BVH_Mesh_Desc {
                vertex_stride = 16,
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

upload_bvh_instances :: proc(upload_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer, instances: []shared.Instance, meshes: []Mesh_GPU) -> gpu.slice_t(gpu.BVH_Instance)
{
    instances_staging := gpu.arena_alloc(upload_arena, gpu.BVH_Instance, len(instances))
    for &instance, i in instances_staging.cpu
    {
        instance = {
            transform = shared.transform_to_gpu_transform(instances[i].transform),
            blas_root = gpu.bvh_root_ptr(meshes[instances[i].mesh_idx].bvh),
            disable_culling = true,
            flip_facing = true,
            mask = 1,
        }
    }
    instances_local := gpu.mem_alloc(gpu.BVH_Instance, len(instances))
    gpu.cmd_mem_copy(cmd_buf, instances_local, instances_staging, len(instances_staging.cpu))
    return instances_local
}

Scene_GPU :: struct
{
    bvh: gpu.Owned_BVH,
    meshes: [dynamic]Mesh_GPU,
    instances_bvh: gpu.slice_t(gpu.BVH_Instance),

    // Shader view
    instances: gpu.slice_t(Instance_Shader),
    meshes_shader: gpu.slice_t(Mesh_Shader),
}

Scene_Shader :: struct
{
    instances: rawptr,
    meshes: rawptr,
}

Instance_Shader :: struct
{
    mesh_idx: u32,
    albedo_tex_id: u32,
}

upload_scene :: proc(scene: shared.Scene, upload_arena: ^gpu.Arena, bvh_scratch_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer) -> Scene_GPU
{
    res: Scene_GPU

    // Upload meshes
    for mesh in scene.meshes
    {
        to_add := upload_mesh(upload_arena, cmd_buf, mesh)
        append(&res.meshes, to_add)
    }

    // Construct structures used by the shader
    instances_gpu := gpu.arena_alloc(upload_arena, Instance_Shader, len(scene.instances))
    for &instance, i in instances_gpu.cpu {
        instance = {
            mesh_idx = scene.instances[i].mesh_idx,
            albedo_tex_id = scene.meshes[scene.instances[i].mesh_idx].base_color_map,
        }
    }
    res.instances = gpu.mem_alloc(Instance_Shader, len(scene.instances), gpu.Memory.GPU)
    gpu.cmd_mem_copy(cmd_buf, res.instances, instances_gpu, len(scene.instances))

    meshes_gpu := gpu.arena_alloc(upload_arena, Mesh_Shader, len(scene.meshes))
    for &mesh, i in meshes_gpu.cpu {
        mesh.pos = res.meshes[i].pos.gpu.ptr
        mesh.normals = res.meshes[i].normals.gpu.ptr
        mesh.uvs     = res.meshes[i].uvs.gpu.ptr
        mesh.indices = res.meshes[i].indices.gpu.ptr
    }
    res.meshes_shader = gpu.mem_alloc(Mesh_Shader, len(scene.meshes), gpu.Memory.GPU)
    gpu.cmd_mem_copy(cmd_buf, res.meshes_shader, meshes_gpu, len(scene.meshes))

    // Build BVHs
    gpu.cmd_barrier(cmd_buf, .Transfer, .Build_BVH)
    for &mesh in res.meshes {
        mesh.bvh = build_blas(bvh_scratch_arena, cmd_buf, mesh.pos, mesh.indices, mesh.idx_count, mesh.vert_count)
    }

    res.instances_bvh = upload_bvh_instances(upload_arena, cmd_buf, scene.instances[:], res.meshes[:])
    gpu.cmd_barrier(cmd_buf, .Transfer, .Build_BVH)

    res.bvh = build_tlas(upload_arena, cmd_buf, res.instances_bvh, u32(len(scene.instances)))
    gpu.cmd_barrier(cmd_buf, .Build_BVH, .All)

    return res
}

scene_destroy :: proc(scene: ^Scene_GPU)
{
    gpu.bvh_free_and_destroy(&scene.bvh)
    gpu.mem_free(scene.instances)
    gpu.mem_free(scene.meshes_shader)
    gpu.mem_free(scene.instances_bvh)
    for &mesh in scene.meshes {
        mesh_destroy(&mesh)
    }
    delete(scene.meshes)
    scene^ = {}
}

create_target_textures :: proc(window_size_x: u32, window_size_y: u32, desc_pool: ^gpu.Descriptor_Pool) -> (color_target: gpu.Owned_Texture, depth_target: gpu.Owned_Texture)
{
    // Color
    {
        color_target = gpu.texture_alloc_and_create(gpu.Texture_Desc {
            dimensions   = { u32(window_size_x), u32(window_size_y), 1 },
            format       = .RGBA16_Float,
            mip_count    = 1,
            layer_count  = 1,
            sample_count = 1,
            usage        = { .Color_Attachment, .Sampled, .Storage },
        })
        COLOR_TARGET_IDX = gpu.desc_pool_alloc_texture(desc_pool, gpu.texture_view_descriptor(color_target, {}))
        COLOR_TARGET_RW_IDX = gpu.desc_pool_alloc_texture_rw(desc_pool, gpu.texture_rw_view_descriptor(color_target, {}))
    }

    // Depth
    {
        depth_target = gpu.texture_alloc_and_create(gpu.Texture_Desc {
            dimensions   = { u32(window_size_x), u32(window_size_y), 1 },
            format       = .D32_Float,
            mip_count    = 1,
            layer_count  = 1,
            sample_count = 1,
            usage        = { .Depth_Stencil_Attachment },
        })
    }

    return
}

Fullscreen_Vertex :: struct {
    pos: [3]f32,
    uv:  [2]f32,
}

create_fullscreen_quad :: proc(
    upload_arena: ^gpu.Arena,
    cmd_buf: gpu.Command_Buffer,
) -> (
    gpu.slice_t(Fullscreen_Vertex),
    gpu.slice_t(u32),
) {
    fsq_verts := gpu.arena_alloc(upload_arena, Fullscreen_Vertex, 4)
    fsq_verts.cpu[0].pos = {-1.0, 1.0, 0.0} // Top-left
    fsq_verts.cpu[1].pos = {1.0, -1.0, 0.0} // Bottom-right
    fsq_verts.cpu[2].pos = {1.0, 1.0, 0.0} // Top-right
    fsq_verts.cpu[3].pos = {-1.0, -1.0, 0.0} // Bottom-left
    fsq_verts.cpu[0].uv = {0.0, 1.0}
    fsq_verts.cpu[1].uv = {1.0, 0.0}
    fsq_verts.cpu[2].uv = {1.0, 1.0}
    fsq_verts.cpu[3].uv = {0.0, 0.0}

    fsq_indices := gpu.arena_alloc(upload_arena, u32, 6)
    fsq_indices.cpu[0] = 0
    fsq_indices.cpu[1] = 2
    fsq_indices.cpu[2] = 1
    fsq_indices.cpu[3] = 0
    fsq_indices.cpu[4] = 1
    fsq_indices.cpu[5] = 3

    full_screen_quad_verts_local := gpu.mem_alloc(Fullscreen_Vertex, 4, gpu.Memory.GPU)
    full_screen_quad_indices_local := gpu.mem_alloc(u32, 6, gpu.Memory.GPU)

    gpu.cmd_mem_copy(
        cmd_buf,
        full_screen_quad_verts_local,
        fsq_verts,
        len(fsq_verts.cpu),
    )
    gpu.cmd_mem_copy(
        cmd_buf,
        full_screen_quad_indices_local,
        fsq_indices,
        len(fsq_indices.cpu),
    )

    return full_screen_quad_verts_local, full_screen_quad_indices_local
}

// Load textures from Texture_Info and update mesh texture IDs
load_scene_textures_from_gltf :: proc(
    texture_infos: []shared.Gltf_Texture_Info,
    data: ^gltf2.Data,
    scene: ^shared.Scene,
    desc_pool: ^gpu.Descriptor_Pool,
) {
    upload_arena := gpu.arena_init()
    defer gpu.arena_destroy(&upload_arena)

    for info, i in texture_infos {
        if cancel_loading_textures {
            return
        }

        if info.mesh_id >= u32(len(scene.meshes)) {
            log.error(
                fmt.tprintf(
                    "Invalid mesh_id %v (only %v meshes available)",
                    info.mesh_id,
                    len(scene.meshes),
                ),
            )
            continue
        }

        sync.mutex_lock(&mutex)
        if event, ok := image_uploaded[info.image_index]; ok {
            sync.mutex_unlock(&mutex)
            sync.one_shot_event_wait(event)
        } else {
            event := new(sync.One_Shot_Event)
            image_uploaded[info.image_index] = event
            sync.mutex_unlock(&mutex)

            img := shared.load_texture_from_gltf(
                info.image_index,
                data,
            )
            defer image.destroy(img)

            texture_idx: u32
            texture := upload_texture(img, &upload_arena)

            texture_idx = gpu.desc_pool_alloc_texture(desc_pool, gpu.texture_view_descriptor(texture, {}))
            if sync.guard(&mutex) do image_to_texture[info.image_index] = {texture, texture_idx}

            sync.one_shot_event_signal(event)

            log.infof(
                "Loaded texture for mesh %v, type %v, texture_id %v",
                info.mesh_id,
                info.texture_type,
                texture_idx,
            )
        }
    }

    for info, i in texture_infos {
        sync.mutex_lock(&mutex)
        texture := image_to_texture[info.image_index]
        sync.mutex_unlock(&mutex)

        gpu.semaphore_wait(upload_sem, upload_sem_val)

        sync.guard(&mutex)

        switch info.texture_type {
        case .Base_Color:
            scene.meshes[info.mesh_id].base_color_map = texture.texture_idx
        case .Metallic_Roughness:
            scene.meshes[info.mesh_id].metallic_roughness_map = texture.texture_idx
        case .Normal:
            scene.meshes[info.mesh_id].normal_map = texture.texture_idx
        }
    }
}

upload_texture :: proc(img: ^image.Image, upload_arena: ^gpu.Arena) -> gpu.Owned_Texture
{
    staging := gpu.arena_alloc_raw(upload_arena, len(img.pixels.buf), 1, 16)
    runtime.mem_copy(staging.cpu, raw_data(img.pixels.buf), len(img.pixels.buf))

    sync.guard(&mutex)
    upload_sem_value_old := upload_sem_val
    upload_sem_val += 1

    texture := gpu.texture_alloc_and_create({
        type = .D2,
        dimensions = {u32(img.width), u32(img.height), 1},
        mip_count = u32(math.log2(f32(max(img.width, img.height)))),
        layer_count = 1,
        sample_count = 1,
        format = .RGBA8_SRGB,
        usage = { .Sampled, .Transfer_Src },
    }, .Transfer)
    append(&loaded_textures, texture)

    // Upload and mipmap generation happen on separate queues so they need to be synchronized using timeline semaphores

    {
        // Upload texture to GPU
        upload_cmd_buf := gpu.commands_begin(.Transfer)
        gpu.cmd_copy_to_texture(upload_cmd_buf, texture, staging, texture.mem)
        gpu.cmd_add_signal_semaphore(upload_cmd_buf, upload_sem, upload_sem_value_old + 1)
        gpu.queue_submit(.Transfer, {upload_cmd_buf})
    }

    // Generate mipmaps
    mipmaps_cmd_buf := gpu.commands_begin(.Main)
    gpu.cmd_barrier(mipmaps_cmd_buf, .Transfer, .Transfer, {})
    gpu.cmd_generate_mipmaps(mipmaps_cmd_buf, texture)
    gpu.cmd_add_wait_semaphore(mipmaps_cmd_buf, upload_sem, upload_sem_value_old + 1)
    gpu.queue_submit(.Main, {mipmaps_cmd_buf})
    return texture
}

init_dear_imgui :: proc(window: ^sdl.Window, desc_pool: ^gpu.Descriptor_Pool) -> ^imgui.Context
{
    imgui.CHECKVERSION()
    ctx := imgui.create_context(nil)
    io := imgui.get_io()
    io.config_flags += {.Nav_Enable_Keyboard, .Nav_Enable_Gamepad}
    io.display_size = { 1000, 1000 }

    imgui_impl_sdl3.init_for_vulkan(window)

    result := imgui_impl_nogfx.init({
        frames_in_flight = Frames_In_Flight
    }, desc_pool)
    assert(result, "Failed to initialize imgui vulkan backend")

    dpi_scale := sdl.GetWindowDisplayScale(window)
    set_dear_imgui_font_and_style(dpi_scale)

    imgui_impl_nogfx.create_fonts_texture(desc_pool)

    return ctx
}

handle_window_events :: proc(window: ^sdl.Window) -> (proceed: bool)
{
    // Reset "one-shot" inputs
    for &key in shared.INPUT.keys {
        key.pressed = false
        key.released = false
    }
    shared.INPUT.mouse_dx = 0
    shared.INPUT.mouse_dy = 0
    shared.INPUT.left_click_pressed = false

    event: sdl.Event
    proceed = true
    for sdl.PollEvent(&event) {
        imgui_impl_sdl3.process_event(&event)

        #partial switch event.type {
        case .QUIT:
            proceed = false
        case .WINDOW_CLOSE_REQUESTED:
            {
                if event.window.windowID == sdl.GetWindowID(window) {
                    proceed = false
                }
            }
        // Input events
        case .MOUSE_BUTTON_DOWN, .MOUSE_BUTTON_UP:
            {
                event := event.button
                if event.type == .MOUSE_BUTTON_DOWN {
                    if event.button == sdl.BUTTON_RIGHT {
                        shared.INPUT.pressing_right_click = true
                    } else if event.button == sdl.BUTTON_LEFT {
                        shared.INPUT.left_click_pressed = true
                    }
                } else if event.type == .MOUSE_BUTTON_UP {
                    if event.button == sdl.BUTTON_RIGHT {
                        shared.INPUT.pressing_right_click = false
                    }
                }
            }
        case .KEY_DOWN, .KEY_UP:
            {
                event := event.key
                if event.repeat do break

                if event.type == .KEY_DOWN {
                    shared.INPUT.keys[event.scancode].pressed = true
                    shared.INPUT.keys[event.scancode].pressing = true
                } else {
                    shared.INPUT.keys[event.scancode].pressing = false
                    shared.INPUT.keys[event.scancode].released = true
                }
            }
        case .MOUSE_MOTION:
            {
                event := event.motion
                shared.INPUT.mouse_dx += event.xrel
                shared.INPUT.mouse_dy -= event.yrel // In sdl, up is negative
            }
        }
    }

    return
}

mask_out_dear_imgui_inputs :: proc()
{
    io := imgui.get_io()

    if imgui.is_mouse_clicked(.Right) && !imgui.is_window_hovered({ .Any_Window }) {
        imgui.set_window_focus_str(nil)
    }

    if io.want_capture_mouse {
        shared.INPUT.pressing_right_click = false
        shared.INPUT.left_click_pressed = false
    }
    if io.want_capture_keyboard {
        shared.INPUT.keys = {}
    }
}

Sampler_Type :: enum u32 { Point = 0, Bilinear, Bicubic }
Output_Type :: enum u32 { Rasterized = 0, Pathtraced }

set_dear_imgui_font_and_style :: proc(dpi_scale: f32)
{
    io := imgui.get_io()
    // Setup behavior flags
    io.config_flags |= { .Nav_Enable_Keyboard }
    io.config_flags |= { .Docking_Enable }
    // io.MouseDrawCursor = true

    // Setup style
    {
        colors := &imgui.get_style().colors
        colors[imgui.Col.Text]                   = { 1.00, 1.00, 1.00, 1.00 }
        colors[imgui.Col.Text_Disabled]           = { 0.50, 0.50, 0.50, 1.00 }
        colors[imgui.Col.Window_Bg]               = { 0.10, 0.10, 0.10, 0.96 }
        colors[imgui.Col.Child_Bg]                = { 0.00, 0.00, 0.00, 0.00 }
        colors[imgui.Col.Popup_Bg]                = { 0.19, 0.19, 0.19, 0.92 }
        colors[imgui.Col.Border]                 = { 0.19, 0.19, 0.19, 0.29 }
        colors[imgui.Col.Border_Shadow]           = { 0.00, 0.00, 0.00, 0.24 }
        colors[imgui.Col.Frame_Bg]                = { 0.05, 0.05, 0.05, 0.54 }
        colors[imgui.Col.Frame_Bg_Hovered]         = { 0.19, 0.19, 0.19, 0.54 }
        colors[imgui.Col.Frame_Bg_Active]          = { 0.20, 0.22, 0.23, 1.00 }
        colors[imgui.Col.Title_Bg]                = { 0.08, 0.08, 0.08, 1.00 }
        colors[imgui.Col.Title_Bg_Active]          = { 0.00, 0.00, 0.00, 1.00 }
        colors[imgui.Col.Title_Bg_Collapsed]       = { 0.00, 0.00, 0.00, 1.00 }
        colors[imgui.Col.Menu_Bar_Bg]              = { 0.14, 0.14, 0.14, 1.00 }
        colors[imgui.Col.Scrollbar_Bg]            = { 0.05, 0.05, 0.05, 0.54 }
        colors[imgui.Col.Scrollbar_Grab]          = { 0.34, 0.34, 0.34, 0.54 }
        colors[imgui.Col.Scrollbar_Grab_Hovered]   = { 0.40, 0.40, 0.40, 0.54 }
        colors[imgui.Col.Scrollbar_Grab_Active]    = { 0.56, 0.56, 0.56, 0.54 }
        colors[imgui.Col.Check_Mark]              = { 0.33, 0.67, 0.86, 1.00 }
        colors[imgui.Col.Slider_Grab]             = { 0.34, 0.34, 0.34, 0.54 }
        colors[imgui.Col.Slider_Grab_Active]       = { 0.56, 0.56, 0.56, 0.54 }
        colors[imgui.Col.Button]                 = { 0.05, 0.05, 0.05, 0.54 }
        colors[imgui.Col.Button_Hovered]          = { 0.19, 0.19, 0.19, 0.54 }
        colors[imgui.Col.Button_Active]           = { 0.20, 0.22, 0.23, 1.00 }
        colors[imgui.Col.Header]                 = { 0.03, 0.03, 0.03, 0.52 }
        colors[imgui.Col.Header_Hovered]          = { 0.06, 0.06, 0.06, 0.36 }
        colors[imgui.Col.Header_Active]           = { 0.20, 0.22, 0.23, 0.33 }
        colors[imgui.Col.Separator]              = { 0.28, 0.28, 0.28, 0.29 }
        colors[imgui.Col.Separator_Hovered]       = { 0.44, 0.44, 0.44, 0.29 }
        colors[imgui.Col.Separator_Active]        = { 0.40, 0.44, 0.47, 1.00 }
        colors[imgui.Col.Resize_Grip]             = { 0.28, 0.28, 0.28, 0.29 }
        colors[imgui.Col.Resize_Grip_Hovered]      = { 0.44, 0.44, 0.44, 0.29 }
        colors[imgui.Col.Resize_Grip_Active]       = { 0.40, 0.44, 0.47, 1.00 }
        colors[imgui.Col.Tab]                    = { 0.00, 0.00, 0.00, 0.52 }
        colors[imgui.Col.Tab_Hovered]             = { 0.14, 0.14, 0.14, 1.00 }
        colors[imgui.Col.Tab_Selected]            = { 0.20, 0.20, 0.20, 0.36 }
        colors[imgui.Col.Tab_Dimmed]              = { 0.00, 0.00, 0.00, 0.52 }
        colors[imgui.Col.Tab_Dimmed_Selected]      = { 0.14, 0.14, 0.14, 1.00 }
        colors[imgui.Col.Plot_Lines]              = { 1.00, 0.00, 0.00, 1.00 }
        colors[imgui.Col.Plot_Lines_Hovered]       = { 1.00, 0.00, 0.00, 1.00 }
        colors[imgui.Col.Plot_Histogram]          = { 1.00, 0.00, 0.00, 1.00 }
        colors[imgui.Col.Plot_Histogram_Hovered]   = { 1.00, 0.00, 0.00, 1.00 }
        colors[imgui.Col.Table_Header_Bg]          = { 0.00, 0.00, 0.00, 0.52 }
        colors[imgui.Col.Table_Border_Strong]      = { 0.00, 0.00, 0.00, 0.52 }
        colors[imgui.Col.Table_Border_Light]       = { 0.28, 0.28, 0.28, 0.29 }
        colors[imgui.Col.Table_Row_Bg]             = { 0.00, 0.00, 0.00, 0.00 }
        colors[imgui.Col.Table_Row_Bg_Alt]          = { 1.00, 1.00, 1.00, 0.06 }
        colors[imgui.Col.Text_Selected_Bg]         = { 0.20, 0.22, 0.23, 1.00 }
        colors[imgui.Col.Drag_Drop_Target]         = { 0.33, 0.67, 0.86, 1.00 }
        //colors[imgui.Col.Nav_Highlight]           = { 0.80, 0.30, 0.20, 1.00 }
        colors[imgui.Col.Nav_Windowing_Highlight]  = { 1.00, 0.00, 0.00, 0.70 }
        colors[imgui.Col.Nav_Windowing_Dim_Bg]      = { 1.00, 0.00, 0.00, 0.20 }
        colors[imgui.Col.Modal_Window_Dim_Bg]       = { 1.00, 0.00, 0.00, 0.35 }
        colors[imgui.Col.Docking_Preview]         = { 0.33, 0.67, 0.86, 1.00 }
        colors[imgui.Col.Docking_Empty_Bg]         = { 0.00, 0.00, 0.00, 0.00 }

        style := imgui.get_style()
        style.window_padding                   = { 8.00, 8.00 }
        style.frame_padding                    = { 5.00, 4.00 }
        style.cell_padding                     = { 6.50, 6.50 }
        style.item_spacing                     = { 6.00, 6.00 }
        style.item_inner_spacing                = { 6.00, 6.00 }
        style.touch_extra_padding               = { 0.00, 0.00 }
        style.indent_spacing                   = 25
        style.scrollbar_size                   = 15
        style.grab_min_size                     = 10
        style.window_border_size                = 1
        style.child_border_size                 = 1
        style.popup_border_size                 = 1
        style.frame_border_size                 = 1
        style.tab_border_size                   = 1
        style.window_rounding                  = 7
        style.child_rounding                   = 4
        style.frame_rounding                   = 3
        style.popup_rounding                   = 4
        style.scrollbar_rounding               = 9
        style.grab_rounding                    = 3
        style.log_slider_deadzone               = 4
        style.tab_rounding                     = 4

        imgui.style_scale_all_sizes(style, auto_cast dpi_scale)
    }

    // Setup font
    {
        font_size: f32 = 17.0
        main_font := imgui.font_atlas_add_font_from_file_ttf(io.fonts, "fonts/Roboto-Medium.ttf", auto_cast (font_size * dpi_scale))
        //imgui.font_atlas_build(io.fonts)
        assert(main_font != nil)
    }
}

gui_show_debug_texture_window :: proc(name: cstring, texture_id: u32, tex_width_int: int, tex_height_int: int, show: ^bool = nil)
{
    tex_width   := f32(tex_width_int)
    tex_height  := f32(tex_height_int)

    @(static) zoom    : f32 = 8.0
    @(static) pan     : imgui.Vec2 = {0, 0}
    @(static) selected_px : int = -1
    @(static) selected_py : int = -1

    imgui.begin(name, show)

    // Controls
    if imgui.button("Reset")
    {
        zoom = 8.0
        pan  = {0, 0}
    }

    // Canvas
    canvas_size := imgui.get_content_region_avail()
    canvas_pos  := imgui.get_cursor_screen_pos()
    draw_list   := imgui.get_window_draw_list()

    imgui.draw_list_push_clip_rect(draw_list, canvas_pos,
        canvas_pos + canvas_size, true)

    // Background
    imgui.draw_list_add_rect_filled(draw_list, canvas_pos,
        canvas_pos + canvas_size,
        rgba8_to_u32(30, 30, 30, 255))

    // Checkerboard pattern
    checker_sz := f32(128)
    cols := int(canvas_size.x / checker_sz) + 1
    rows := int(canvas_size.y / checker_sz) + 1
    for row in 0..<rows
    {
        for col in 0..<cols
        {
            if (row + col) % 2 == 0 { continue }

            p0 := canvas_pos + {f32(col) * checker_sz, f32(row) * checker_sz}
            p1 := p0 + {checker_sz, checker_sz}
            imgui.draw_list_add_rect_filled(draw_list, p0, p1,
                rgba8_to_u32(60, 60, 60, 255))
        }
    }

    // Invisible button to capture input over the canvas area
    imgui.invisible_button("canvas", canvas_size, { .Mouse_Button_Left, .Mouse_Button_Right })
    is_hovered := imgui.is_item_hovered()
    is_active  := imgui.is_item_active()

    io := imgui.get_io()

    // Scroll-wheel zoom
    if is_hovered && io.mouse_wheel != 0.0
    {
        old_zoom := zoom
        if io.mouse_wheel > 0 {
            zoom *= f32(1.1)
        } else {
            zoom /= f32(1.1)
        }
        zoom = clamp(zoom, 1.0, 64.0)

        // Mouse position relative to canvas origin
        mouse_canvas := io.mouse_pos - canvas_pos
        // The point on the texture the mouse is over must stay fixed:
        // mouse_canvas = pan + texel_pos * zoom  (before and after)
        // So: pan_new = mouse_canvas - (mouse_canvas - pan_old) * (zoom_new / old_zoom)
        pan.x = mouse_canvas.x - (mouse_canvas.x - pan.x) * (zoom / old_zoom)
        pan.y = mouse_canvas.y - (mouse_canvas.y - pan.y) * (zoom / old_zoom)
    }

    // Drag
    if is_active && imgui.is_mouse_dragging(.Right, 0.0) {
        pan = pan + io.mouse_delta
    }

    // Texture
    img_w := tex_width  * zoom
    img_h := tex_height * zoom
    img_p0 := canvas_pos + pan
    img_p1 := img_p0 + {img_w, img_h}

    // Set sampler to nearest
    imgui.draw_list_add_callback(draw_list, set_nearest_sampler_callback, nil)

    imgui.im_draw_list_add_image(draw_list, imgui.Texture_ID(texture_id),
        img_p0, img_p1,
        {0, 0}, {1, 1},
        rgba8_to_u32(255, 255, 255, 255))

    // Restore sampler
    imgui.draw_list_add_callback(draw_list, set_linear_sampler_callback, nil)

    // Pixel grid (only when zoom >= 4) ---
    if zoom >= 4.0
    {
        grid_col := rgba8_to_u32(0, 0, 0, 120)

        // Vertical lines
        for x in 0..=int(tex_width)
        {
            lx := img_p0.x + f32(x) * zoom
            if lx < canvas_pos.x || lx > canvas_pos.x + canvas_size.x { continue }

            imgui.draw_list_add_line(draw_list,
                {lx, max(img_p0.y, canvas_pos.y)},
                {lx, min(img_p1.y, canvas_pos.y + canvas_size.y)},
                grid_col, 1.0)
        }

        // Horizontal lines
        for y in 0..=int(tex_height)
        {
            ly := img_p0.y + f32(y) * zoom
            if ly < canvas_pos.y || ly > canvas_pos.y + canvas_size.y { continue }

            imgui.draw_list_add_line(draw_list,
                {max(img_p0.x, canvas_pos.x), ly},
                {min(img_p1.x, canvas_pos.x + canvas_size.x), ly},
                grid_col, 1.0)
        }
    }

    // Hovered-pixel outline
    if is_hovered && zoom >= 4.0
    {
        mouse := io.mouse_pos

        // Map mouse from screen-space into texture pixel coords
        rel   := mouse - img_p0
        px    := int(rel.x / zoom)
        py    := int(rel.y / zoom)

        if is_hovered && imgui.is_mouse_clicked(.Left)
        {
            if px >= 0 && px < int(tex_width) && py >= 0 && py < int(tex_height)
            {
                selected_px = px
                selected_py = py
            }
        }

        if px >= 0 && px < int(tex_width) && py >= 0 && py < int(tex_height)
        {
            // Screen-space corners of that pixel
            px0 := img_p0.x + f32(px) * zoom
            py0 := img_p0.y + f32(py) * zoom
            px1 := px0 + zoom
            py1 := py0 + zoom

            outline_col := rgba8_to_u32(255, 255, 255, 220)
            shadow_col  := rgba8_to_u32(0,   0,   0,   160)

            // Shadow for contrast on any background
            imgui.draw_list_add_rect(draw_list, {px0 - 1, py0 - 1}, {px1 + 1, py1 + 1},
                shadow_col, 0.0, {}, 2.0)
            // Bright outline
            imgui.draw_list_add_rect(draw_list, {px0, py0}, {px1, py1},
                outline_col, 0.0, {}, 1.5)

            // Tooltip: pixel coordinates
            imgui.set_tooltip("Pixel (%d, %d)", px, py)
        }

        // Draw selected pixel highlight
        if selected_px >= 0
        {
            spx0 := img_p0.x + f32(selected_px) * zoom
            spy0 := img_p0.y + f32(selected_py) * zoom
            spx1 := spx0 + zoom
            spy1 := spy0 + zoom
            imgui.draw_list_add_rect(draw_list, {spx0, spy0}, {spx1, spy1},
                rgba8_to_u32(255, 80, 0, 220), 0.0, {}, 2.0)
        }
    }

    imgui.draw_list_pop_clip_rect(draw_list)

    imgui.end()
}

set_nearest_sampler_callback :: proc(draw_list: ^imgui.Draw_List, cmd: ^imgui.Draw_Cmd)
{
    render_state := cast(^imgui_impl_nogfx.Render_State) imgui.get_platform_io().renderer_render_state
    render_state.sampler_current_id = render_state.sampler_nearest_id
}

set_linear_sampler_callback :: proc(draw_list: ^imgui.Draw_List, cmd: ^imgui.Draw_Cmd)
{
    render_state := cast(^imgui_impl_nogfx.Render_State) imgui.get_platform_io().renderer_render_state
    render_state.sampler_current_id = render_state.sampler_linear_id
}

rgba8_to_u32 :: proc(r: u8, g: u8, b: u8, a: u8) -> u32
{
    return u32(a) << 24 | u32(b) << 16 | u32(g) << 8 | u32(r)
}
