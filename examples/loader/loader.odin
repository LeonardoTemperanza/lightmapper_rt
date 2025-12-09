
package loader

import "core:fmt"
import "core:math/linalg"
import "core:math"
import "core:c"
import stbrp "vendor:stb/rect_pack"
import lm "../.."
import "core:strings"
import "core:image"
import "core:image/png"
import "core:image/jpeg"
import "base:runtime"
import "core:os"
import "core:log"

import vku "../../vk_utils"
import vk "vendor:vulkan"
import "../ufbx"
import "../gltf2"

load_scene_fbx :: proc(using ctx: ^lm.App_Vulkan_Context, lm_ctx: ^lm.Context, cmd_pool: vk.CommandPool, path: cstring,
                       lightmap_size: i32, texels_per_world_unit := 100, min_instance_texels := 64, max_instance_texels := 1024) -> [dynamic]lm.Instance
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

    // Loop through meshes.
    meshes: [dynamic]lm.Mesh_Handle
    defer delete(meshes)
    for i in 0..<scene.meshes.count
    {
        fbx_mesh := scene.meshes.data[i]

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

        // NOTE: uv_set[0] is the same as fbx_mesh.vertex_uv
        // Find the lightmap UVs
        lightmap_uv_idx := -1
        for j in 0..<fbx_mesh.uv_sets.count
        {
            uv_set := fbx_mesh.uv_sets.data[j]
            if uv_set.name.data == "LightMapUV" || uv_set.name.data == "UVMap_Lightmap" || uv_set.name.data == "UVSet1" {
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

        append(&meshes, lm.create_mesh(lm_ctx, indices[:], pos_buf[:], normals_buf[:], lm_uvs_buf[:], /*diffuse_uvs_buf[:]*/))
    }

    textures: [dynamic]lm.Texture_Handle
    defer delete(textures)
    for i in 0..<scene.textures.count
    {
        texture := scene.textures.data[i]
        texture_path := strings.string_from_ptr(transmute(^u8) texture.absolute_filename.data,
                                                auto_cast texture.absolute_filename.length)

        fmt.println(texture_path)

        loaded, ok := load_texture(texture_path)
        if !ok do return {}
        append(&textures, loaded)
    }

    // Loop through materials
    materials: [dynamic]lm.Texture_Handle
    defer delete(materials)
    for i in 0..<scene.materials.count
    {
        material := scene.materials.data[i]
        fmt.println(material)
        if material.fbx.diffuse_color.texture != nil {
            append(&materials, textures[material.fbx.diffuse_color.texture.element.typed_id])
        }
    }

    // Loop through instances.
    instances: [dynamic]lm.Instance
    instance_loop: for i in 0..<scene.nodes.count
    {
        node := scene.nodes.data[i]
        if node.is_root || node.mesh == nil do continue

        assert(node.materials.count > 0)
        mat_id := node.materials.data[0].element.typed_id

        instance := lm.Instance {
            transform = get_node_world_transform(node),
            mesh = meshes[node.mesh.element.typed_id],
            diffuse_tex = materials[mat_id]
        }
        append(&instances, instance)
    }

    pack_lightmap_uvs(lm_ctx, instances[:], lightmap_size, texels_per_world_unit, min_instance_texels, max_instance_texels)

    return instances
}

load_scene_gltf :: proc(using ctx: ^lm.App_Vulkan_Context, lm_ctx: ^lm.Context, cmd_pool: vk.CommandPool, path: string,
                       lightmap_size: i32, texels_per_world_unit := 100, min_instance_texels := 64, max_instance_texels := 1024) -> (res: [dynamic]lm.Instance, ok: bool)
{
    data, err_l := gltf2.load_from_file(path)
    switch err in err_l
    {
        case gltf2.JSON_Error: break
        case gltf2.GLTF_Error: break
    }
    defer gltf2.unload(data)

    // Load meshes
    Primitive_ID :: struct
    {
        mesh_idx: int,
        primitive_idx: int,
    }
    meshes: map[Primitive_ID]lm.Mesh_Handle
    defer delete(meshes)
    for mesh, i in data.meshes
    {
        for primitive, j in mesh.primitives
        {
            fmt.println(primitive)

            // ???
            buffer_view := &data.buffer_views[data.accessors[primitive.attributes["POSITION"]].buffer_view.?]
            assert(buffer_view.byte_stride == 12 || buffer_view.byte_stride == nil)
            buffer_view.byte_stride = nil
            buffer_view = &data.buffer_views[data.accessors[primitive.attributes["NORMAL"]].buffer_view.?]
            assert(buffer_view.byte_stride == 12 || buffer_view.byte_stride == nil)
            buffer_view.byte_stride = nil
            buffer_view = &data.buffer_views[data.accessors[primitive.attributes["TEXCOORD_0"]].buffer_view.?]
            assert(buffer_view.byte_stride == 8 || buffer_view.byte_stride == nil)
            buffer_view.byte_stride = nil

            positions := gltf2.buffer_slice(data, primitive.attributes["POSITION"])
            normals := gltf2.buffer_slice(data, primitive.attributes["NORMAL"])
            //diffuse_uvs := gltf2.buffer_slice(data, primitive.attributes["TEXCOORD_0"])
            lm_uvs := gltf2.buffer_slice(data, primitive.attributes["TEXCOORD_0"])
            indices := gltf2.buffer_slice(data, primitive.indices.?)

            indices_u32: [dynamic]u32
            defer delete(indices_u32)
            #partial switch ids in indices
            {
                case []u16:
                    for i in 0..<len(ids) do append(&indices_u32, u32(ids[i]))
                case []u32:
                    for i in 0..<len(ids) do append(&indices_u32, ids[i])
                case: assert(false)
            }

            loaded := lm.create_mesh(lm_ctx, indices_u32[:],
                                             positions.([][3]f32),
                                             normals.([][3]f32),
                                             lm_uvs.([][2]f32),
                                             /*diffuse_uvs_buf[:]*/)
            meshes[{i,j}] = loaded

            fmt.println("loaded primitive")
        }
    }

    // Load textures
    textures: [dynamic]lm.Texture_Handle
    defer delete(textures)
    for image in data.images
    {
        if image.uri != nil
        {
            fmt.println("name:", image.name)

            loaded, ok := load_texture_from_memory(image.uri.([]byte))
            if !ok do return {}, false

            append(&textures, loaded)
            fmt.println("loaded texture")
        }
        else if image.buffer_view != nil
        {
            assert(false, "not implemented")
        }
    }

    Material :: struct
    {
        diffuse_tex: lm.Texture_Handle
    }

    materials: [dynamic]Material
    defer delete(materials)
    for material in data.materials
    {
        base_color_tex := material.metallic_roughness.?.base_color_texture.?
        base_color_tex_id := base_color_tex.index
        append(&materials, Material { textures[base_color_tex_id] })
    }

    // Load instances
    parents := make([]i32, len(data.nodes))  // -1 means no parent
    defer delete(parents)
    for &parent in parents do parent = -1
    for node, i in data.nodes
    {
        for child in node.children {
            parents[child] = i32(i)
        }
    }

    instances: [dynamic]lm.Instance
    for node, i in data.nodes
    {
        mesh := data.meshes[node.mesh.?]
        for primitive, j in mesh.primitives
        {
            mesh_handle := meshes[{int(node.mesh.?),j}]

            mat := materials[primitive.material.?]

            instance := lm.Instance {
                transform = get_node_world_transform_gltf(data, parents, u32(i)),
                mesh = mesh_handle,
                diffuse_tex = mat.diffuse_tex,
                lm_idx = {},
                lm_offset = {},
                lm_scale = {}
            }
            append(&instances, instance)
        }

        fmt.println(data.nodes)
    }

    pack_lightmap_uvs(lm_ctx, instances[:], lightmap_size, texels_per_world_unit, min_instance_texels, max_instance_texels)

    return instances, true
}

pack_lightmap_uvs :: proc(lm_ctx: ^lm.Context, instances: []lm.Instance, lightmap_size: i32, texels_per_world_unit := 100, min_instance_texels := 64, max_instance_texels := 1024)
{
    //fmt.println("Packing uvs...")
    //defer fmt.println("Done packing uvs.")

    num_nodes := lightmap_size
    tmp_nodes := make([]stbrp.Node, num_nodes, allocator = context.allocator)
    defer delete(tmp_nodes)

    stbrp_ctx: stbrp.Context
    stbrp.init_target(&stbrp_ctx, lightmap_size, lightmap_size, raw_data(tmp_nodes), i32(len(tmp_nodes)))

    rects := make([dynamic]stbrp.Rect, len(instances))
    defer delete(rects)

    for instance, i in instances
    {
        rect_size := get_instance_lm_size(instance, lm.get_mesh(lm_ctx, instance.mesh)^, texels_per_world_unit, min_instance_texels, max_instance_texels)

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

            instances[rect.id].lm_idx = lm_idx
            instances[rect.id].lm_offset = { f32(rect.x) / f32(lightmap_size), f32(rect.y) / f32(lightmap_size) }
            instances[rect.id].lm_scale = f32(rect.w) / f32(lightmap_size)
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

load_texture :: proc(path: string) -> (res: lm.Texture_Handle, ok: bool)
{
    file := load_file(path, allocator = context.allocator) or_return
    defer delete(file)
    return load_texture_from_memory(file)
}

load_texture_from_memory :: proc(content: []byte) -> (res: lm.Texture_Handle, ok: bool)
{
    options := image.Options {
        .alpha_add_if_missing,
    }
    tex, err := image.load_from_bytes(content, options, allocator = context.allocator)
    defer image.destroy(tex)
    if err != nil do return {}, false

    // vku.upload_texture()

    return {}, true
}

load_file :: proc(path: string, allocator: runtime.Allocator) -> (data: []byte, ok: bool)
{
    content, ok_r := os.read_entire_file_from_filename(path, allocator)
    if !ok_r
    {
        log.errorf("Failed to read file '%v'.", path)
        return {}, false
    }

    return content, true
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

get_node_world_transform_gltf :: proc(data: ^gltf2.Data, parents: []i32, node: u32) -> matrix[4, 4]f32
{
    local := data.nodes[node].mat
    if parents[node] == -1 do return local
    return get_node_world_transform_gltf(data, parents, u32(parents[node])) * local
}

get_node_world_transform :: proc(node: ^ufbx.Node) -> matrix[4, 4]f32
{
    if node == nil { return 1 }

    local := xform_to_mat(node.local_transform.translation, transmute(quaternion256) node.local_transform.rotation, node.local_transform.scale)

    if node.is_root { return local }

    return get_node_world_transform(node.parent) * local
}

v3_to_v4 :: proc(v: [3]f32, w: Maybe(f32) = nil) -> (res: [4]f32)
{
    res.xyz = v.xyz
    if num, ok := w.?; ok {
        res.w = num
    }
    return
}

xform_to_mat :: proc(pos: [3]f64, rot: quaternion256, scale: [3]f64) -> matrix[4, 4]f32
{
    return cast(matrix[4, 4]f32) (#force_inline linalg.matrix4_translate(pos) *
           #force_inline linalg.matrix4_from_quaternion(rot) *
           #force_inline linalg.matrix4_scale(scale))
}
