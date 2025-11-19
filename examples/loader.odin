
package main

import "core:fmt"
import "core:math/linalg"
import "core:math"
import "core:c"
import stbrp "vendor:stb/rect_pack"
import lm "../"

import vku "../vk_utils"
import vk "vendor:vulkan"
import "ufbx"

load_scene_fbx :: proc(using ctx: ^lm.App_Vulkan_Context, cmd_pool: vk.CommandPool, path: cstring,
                       texels_per_world_unit := 100, min_instance_texels := 64, max_instance_texels := 1024) -> lm.Scene
{
    // Load the .fbx file
    opts := ufbx.Load_Opts {
        target_unit_meters = 1,
        target_axes = {
            right = .POSITIVE_X,
            up = .POSITIVE_Y,
            front = .NEGATIVE_Z,
        }
    }
    err: ufbx.Error
    scene := ufbx.load_file(path, &opts, &err)
    defer ufbx.free_scene(scene)
    if scene == nil
    {
        fmt.printfln("%s", err.description.data)
        panic("Failed to load")
    }

    res: lm.Scene

    // Loop through meshes.
    for i in 0..<scene.meshes.count
    {
        fbx_mesh := scene.meshes.data[i]

        mesh: lm.Mesh

        // Indices
        index_count := 3 * fbx_mesh.num_triangles
        indices := make([dynamic]u32, index_count)
        offset := u32(0)
        for j in 0..<fbx_mesh.faces.count
        {
            face := fbx_mesh.faces.data[j]
            num_tris := ufbx.catch_triangulate_face(nil, &indices[offset], uint(index_count), fbx_mesh, face)
            offset += 3 * num_tris
        }

        idx_usage_flags := vk.BufferUsageFlags { .INDEX_BUFFER, .TRANSFER_DST, .SHADER_DEVICE_ADDRESS, .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .STORAGE_BUFFER }
        mesh.indices_gpu = vku.upload_buffer(device, phys_device, queue, cmd_pool, indices[:], idx_usage_flags, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
        mesh.indices_cpu = indices

        // NOTE: uv_set[0] is the same as fbx_mesh.vertex_uv
        // Find the lightmap UVs
        lightmap_uv_idx := -1
        for j in 0..<fbx_mesh.uv_sets.count
        {
            uv_set := fbx_mesh.uv_sets.data[j]
            if uv_set.name.data == "LightMapUV" || uv_set.name.data == "UVMap_Lightmap" {
                lightmap_uv_idx = int(uv_set.index)
            }
        }

        // Verts
        vertex_count := fbx_mesh.num_indices
        pos_buf := make([dynamic][3]f32, vertex_count)
        normals_buf := make([][3]f32, vertex_count, allocator = context.temp_allocator)
        lm_uvs_buf := make([][2]f32, vertex_count, allocator = context.temp_allocator)
        for j in 0..<vertex_count
        {
            assert(j < fbx_mesh.vertex_position.indices.count)
            assert(fbx_mesh.vertex_position.indices.data[j] < u32(fbx_mesh.vertex_position.values.count))

            pos := fbx_mesh.vertex_position.values.data[fbx_mesh.vertex_position.indices.data[j]]
            norm := fbx_mesh.vertex_normal.values.data[fbx_mesh.vertex_normal.indices.data[j]]
            pos_buf[j] = {f32(pos.x), f32(pos.y), f32(pos.z)}
            normals_buf[j] = {f32(norm.x), f32(norm.y), f32(norm.z)}
            if lightmap_uv_idx != -1 {
                uv_set := fbx_mesh.uv_sets.data[lightmap_uv_idx]
                lm_uv := uv_set.vertex_uv.values.data[uv_set.vertex_uv.indices.data[j]]
                lm_uvs_buf[j] = {f32(lm_uv.x), f32(lm_uv.y)}
            }
        }

        v_usage_flags := vk.BufferUsageFlags { .VERTEX_BUFFER, .TRANSFER_DST, .SHADER_DEVICE_ADDRESS, .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .STORAGE_BUFFER }
        mesh.pos = vku.upload_buffer(device, phys_device, queue, cmd_pool, pos_buf[:], v_usage_flags, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
        mesh.normals = vku.upload_buffer(device, phys_device, queue, cmd_pool, normals_buf[:], v_usage_flags, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
        mesh.pos_cpu = pos_buf

        if lightmap_uv_idx == -1 {
            mesh.lm_uvs_present = false
            fmt.println("no lm uvs")
        } else {
            mesh.lm_uvs_present = true
            mesh.lm_uvs = vku.upload_buffer(device, phys_device, queue, cmd_pool, lm_uvs_buf[:], v_usage_flags, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
        }

        mesh.idx_count = u32(index_count)

        mesh.blas = cast(lm.Accel_Structure) create_blas(ctx, cmd_pool, mesh.pos, mesh.indices_gpu, u32(len(pos_buf)), u32(len(indices)))

        append(&res.meshes, mesh)
    }

    // Loop through instances.
    instance_loop: for i in 0..<scene.nodes.count
    {
        node := scene.nodes.data[i]
        if node.is_root || node.mesh == nil do continue

        instance := lm.Instance {
            transform = get_node_world_transform(node),
            mesh_idx = node.mesh.element.typed_id,
        }
        append(&res.instances, instance)
    }

    // Pack lightmap uvs.
    {
        fmt.println("Packing uvs...")
        defer fmt.println("Done packing uvs.")

        num_nodes := LIGHTMAP_SIZE
        tmp_nodes := make([]stbrp.Node, num_nodes, allocator = context.temp_allocator)
        stbrp_ctx: stbrp.Context
        stbrp.init_target(&stbrp_ctx, LIGHTMAP_SIZE, LIGHTMAP_SIZE, raw_data(tmp_nodes), i32(len(tmp_nodes)))

        rects := make([dynamic]stbrp.Rect, len(res.instances))
        defer delete(rects)

        for instance, i in res.instances
        {
            rect_size := get_instance_lm_size(instance, res.meshes[instance.mesh_idx], texels_per_world_unit, min_instance_texels, max_instance_texels)

            rects[i].id = c.int(i)
            rects[i].w = rect_size
            rects[i].h = rect_size
        }

        lm_idx := u32(0)
        for
        {
            all_fit := bool(stbrp.pack_rects(&stbrp_ctx, raw_data(rects), c.int(len(rects))))

            for rect in rects
            {
                if !rect.was_packed { continue }

                assert(rect.w == rect.h)

                res.instances[rect.id].lm_idx = lm_idx
                res.instances[rect.id].lm_offset = { f32(rect.x) / f32(LIGHTMAP_SIZE), f32(rect.y) / f32(LIGHTMAP_SIZE) }
                res.instances[rect.id].lm_scale = f32(rect.w) / f32(LIGHTMAP_SIZE)
            }

            if all_fit { break }

            fmt.println("Did not all fit!")
            assert(false)

            remaining_rects: [dynamic]stbrp.Rect

            for rect in rects
            {
                if !rect.was_packed {
                    append(&remaining_rects, rect)
                }
            }

            delete(rects)
            rects = remaining_rects

            lm_idx += 1
        }
    }

    res.tlas = transmute(lm.Tlas) create_tlas(ctx, cmd_pool, res.instances[:], res.meshes[:])

    return res
}

get_instance_lm_size :: proc(instance: lm.Instance, mesh: lm.Mesh, texels_per_world_unit: int, min_instance_texels: int, max_instance_texels: int) -> stbrp.Coord
{
    res := f32(0.0)
    for i := 0; i < len(mesh.indices_cpu); i += 3
    {
        idx0 := mesh.indices_cpu[i+0]
        idx1 := mesh.indices_cpu[i+1]
        idx2 := mesh.indices_cpu[i+2]

        v0 := (instance.transform * v3_to_v4(mesh.pos_cpu[idx0].xyz, 1.0)).xyz
        v1 := (instance.transform * v3_to_v4(mesh.pos_cpu[idx1].xyz, 1.0)).xyz
        v2 := (instance.transform * v3_to_v4(mesh.pos_cpu[idx2].xyz, 1.0)).xyz

        area := linalg.length(linalg.cross(v1 - v0, v2 - v0)) / 2.0
        res += area
    }

    size := stbrp.Coord(math.ceil(math.sqrt(res) * f32(texels_per_world_unit)))
    size = stbrp.Coord(clamp(int(size), min_instance_texels, max_instance_texels))
    return size
}

get_node_world_transform :: proc(node: ^ufbx.Node) -> matrix[4, 4]f32
{
    if node == nil { return 1 }

    local := xform_to_mat(node.local_transform.translation, transmute(quaternion256) node.local_transform.rotation, node.local_transform.scale)

    if node.is_root { return local }

    return get_node_world_transform(node.parent) * local
}
