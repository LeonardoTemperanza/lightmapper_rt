
/*
MIT License

Copyright (c) 2025 Leonardo Temperanza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

package lightmapper

import "core:fmt"
import "core:math/linalg"
import intr "base:intrinsics"
import "core:slice"
import "core:log"
import "core:sync"
import "base:runtime"
import "core:c"
import thr "core:thread"
import "core:time"
import "core:sort"
import "core:math"

import sdl "vendor:sdl3"
import vk "vendor:vulkan"

import vku "vk_utils"
import oidn "oidn_odin_bindings"

// For renderdoc debugging
SYNCHRONOUS :: #config(LM_SYNCHRONOUS, true)

Context :: struct
{
    using vk_ctx: Lightmapper_Vulkan_Context,
    oidn_device: oidn.Device,
    shaders: Shaders,

    upload_cmd_pool: vk.CommandPool,
    desc_pool: vk.DescriptorPool,

    meshes: [dynamic]Mesh,
    meshes_free: [dynamic]u32,
    textures: [dynamic]Texture,
    textures_free: [dynamic]u32,
}

Buffer :: vku.Buffer
Image :: vku.Image

Texture :: struct
{
    gen: u32,
    using image: Image,
}

// Initialization

// Initializes this library with its own Vulkan context.
// Only really makes sense for projects not using Vulkan.
init_from_scratch :: proc() -> Context
{
    return {}
}

// Initializes this library with an already existing Vulkan context.
// NOTE: This library is asynchronous by default, so use a separate queue here!
// NOTE: This library needs the following extensions:
// - VK_EXT_SHADER_OBJECT
// - VK_KHR_ACCELERATION_STRUCTURE
// - VK_KHR_RAY_TRACING_PIPELINE
// - VK_KHR_DEFERRED_HOST_OPERATIONS
// - VK_EXT_CONSERVATIVE_RASTERIZATION
// - VK_KHR_RAY_TRACING_POSITION_FETCH
// - VK_KHR_EXTERNAL_MEMORY_WIN32 / VK_KHR_EXTERNAL_MEMORY_FD
init_from_vulkan_context :: proc(phys_device: vk.PhysicalDevice, device: vk.Device, queue: vk.Queue, queue_family_idx: u32) -> Context
{
    res: Context
    res.vk_ctx.phys_device = phys_device
    res.vk_ctx.device = device
    res.vk_ctx.queue = queue
    res.vk_ctx.queue_family_idx = queue_family_idx

    res.shaders = create_shaders(&res.vk_ctx)
    res.oidn_device = create_oidn_context(res.vk_ctx.phys_device)

    return res
}

init_test :: proc(vk_ctx: Lightmapper_Vulkan_Context) -> Context
{
    res: Context
    res.vk_ctx = vk_ctx

    res.shaders = create_shaders(&res.vk_ctx)
    res.oidn_device = create_oidn_context(vk_ctx.phys_device)

    upload_cmd_pool_ci := vk.CommandPoolCreateInfo {
        sType = .COMMAND_POOL_CREATE_INFO,
        queueFamilyIndex = res.queue_family_idx,
        flags = { .TRANSIENT }
    }
    upload_cmd_pool: vk.CommandPool
    vk_check(vk.CreateCommandPool(res.device, &upload_cmd_pool_ci, nil, &res.upload_cmd_pool))

    desc_pool_ci := vk.DescriptorPoolCreateInfo {
        sType = .DESCRIPTOR_POOL_CREATE_INFO,
        flags = { .FREE_DESCRIPTOR_SET },
        maxSets = 100000,
        poolSizeCount = 5,
        pPoolSizes = raw_data([]vk.DescriptorPoolSize {
            {
                type = .ACCELERATION_STRUCTURE_KHR,
                descriptorCount = 5,
            },
            {
                type = .STORAGE_IMAGE,
                descriptorCount = 10,
            },
            {
                type = .SAMPLED_IMAGE,
                descriptorCount = 10,
            },
            {
                type = .COMBINED_IMAGE_SAMPLER,
                descriptorCount = 10,
            },
            {
                type = .STORAGE_BUFFER,
                descriptorCount = 100000,
            },
        })
    }
    vk_check(vk.CreateDescriptorPool(vk_ctx.device, &desc_pool_ci, nil, &res.desc_pool))

    return res
}

// Scene description

Dir_Light :: struct
{
    dir: [3]f32,
    emission: [3]f32,
    angle: f32,
}

Handle :: struct
{
    idx: u32,
    gen: u32,
}

Mesh_Handle :: distinct Handle
Texture_Handle :: distinct Handle

create_mesh :: proc(using ctx: ^Context, indices: []u32, positions: [][3]f32, normals: [][3]f32, lm_uvs: [][2]f32 /*, diffuse_uvs: [][2]f32 */) -> Mesh_Handle
{
    assert(len(indices) > 0 && len(positions) > 0 && len(normals) > 0 && len(lm_uvs) > 0 /*&& len(diffuse_uvs) > 0*/)

    mesh: Mesh

    mesh.indices_cpu = slice.clone_to_dynamic(indices)
    mesh.pos_cpu = slice.clone_to_dynamic(positions)
    mesh.normals_cpu = slice.clone_to_dynamic(normals)
    mesh.geom_normals_cpu = compute_geom_normals(indices, positions)
    mesh.lm_uvs_cpu = slice.clone_to_dynamic(lm_uvs)
    // TODO diffuse uvs

    seams := find_seams(indices, positions, normals, lm_uvs)
    defer delete(seams)
    seams_gpu := make([dynamic]Seam_GPU, len(seams), len(seams))
    defer delete(seams_gpu)
    for seam, i in seams
    {
        seams_gpu[i] = {
            lines = {
                Line_GPU { { lm_uvs[seam.edge_a[0]], lm_uvs[seam.edge_a[1]] } },
                Line_GPU { { lm_uvs[seam.edge_b[0]], lm_uvs[seam.edge_b[1]] } },
            }
        }
    }

    idx_usage_flags := vk.BufferUsageFlags { .INDEX_BUFFER, .TRANSFER_DST, .SHADER_DEVICE_ADDRESS, .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .STORAGE_BUFFER }
    v_usage_flags := vk.BufferUsageFlags { .VERTEX_BUFFER, .TRANSFER_DST, .SHADER_DEVICE_ADDRESS, .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .STORAGE_BUFFER }
    seams_usage_flags := vk.BufferUsageFlags { .TRANSFER_DST, .SHADER_DEVICE_ADDRESS, .STORAGE_BUFFER }

    mesh.indices = vku.upload_buffer(device, phys_device, queue, upload_cmd_pool, indices[:], idx_usage_flags, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    mesh.pos = vku.upload_buffer(device, phys_device, queue, upload_cmd_pool, positions[:], v_usage_flags, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    mesh.normals = vku.upload_buffer(device, phys_device, queue, upload_cmd_pool, normals[:], v_usage_flags, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    mesh.geom_normals = vku.upload_buffer(device, phys_device, queue, upload_cmd_pool, mesh.geom_normals_cpu[:], v_usage_flags, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    mesh.lm_uvs = vku.upload_buffer(device, phys_device, queue, upload_cmd_pool, lm_uvs[:], v_usage_flags, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    // TODO diffuse uvs
    mesh.seams = vku.upload_buffer(device, phys_device, queue, upload_cmd_pool, seams_gpu[:], seams_usage_flags, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    mesh.num_seams = u32(len(seams))
    desc_set_ai := vk.DescriptorSetAllocateInfo {
        sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool = desc_pool,
        descriptorSetCount = 1,
        pSetLayouts = raw_data([]vk.DescriptorSetLayout { shaders.seams_desc_set_layout })
    }
    vk_check(vk.AllocateDescriptorSets(device, &desc_set_ai, &mesh.seams_desc_set))
    writes := []vk.WriteDescriptorSet {
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = mesh.seams_desc_set,
            dstBinding = 0,
            descriptorCount = 1,
            descriptorType = .STORAGE_BUFFER,
            pBufferInfo = raw_data([]vk.DescriptorBufferInfo {
                {
                    buffer = mesh.seams.handle,
                    offset = vk.DeviceSize(0),
                    range = vk.DeviceSize(vk.WHOLE_SIZE),
                }
            })
        }
    }
    vk.UpdateDescriptorSets(device, u32(len(writes)), raw_data(writes), 0, nil)

    mesh.blas = vku.create_blas(device, phys_device, queue, upload_cmd_pool, mesh.pos, mesh.indices, u32(len(positions)), u32(len(indices)))

    free_slot := u32(0)
    if len(meshes_free) > 0
    {
        free_slot = pop(&meshes_free)
        mesh.gen = meshes[free_slot].gen
        meshes[free_slot] = mesh
    }
    else
    {
        mesh.gen = 0
        append(&meshes, mesh)
        free_slot = u32(len(meshes)) - 1
    }
    return Mesh_Handle { idx = free_slot, gen = mesh.gen }
}

/*
register_mesh :: proc(indices: []u32, positions: [][3]f32, lm_uvs: [][2]f32, indices_buf: Buffer, positions_buf: Buffer, lm_uvs_buf: Buffer) -> Mesh_Handle
{
    return {}
}
*/

get_mesh :: proc(using ctx: ^Context, handle: Mesh_Handle) -> ^Mesh
{
    if u32(len(meshes)) <= handle.idx do return nil
    if meshes[handle.idx].gen != handle.gen do return nil
    return &meshes[handle.idx]
}

release_mesh :: proc(using ctx: ^Context, handle: Mesh_Handle)
{
    if u32(len(meshes)) <= handle.idx do return
    if meshes[handle.idx].gen != handle.gen do return
    meshes[handle.idx].gen += 1
    if u32(len(meshes)) - 1 == handle.idx {
        pop(&meshes)
    } else {
        append(&meshes_free, handle.idx)
    }
}

create_texture :: proc(using ctx: ^Context) -> Texture_Handle
{
    texture: Texture

    free_slot := u32(0)
    if len(textures_free) > 0
    {
        free_slot = pop(&textures_free)
        texture.gen = textures[free_slot].gen
        textures[free_slot] = texture
    }
    else
    {
        texture.gen = 0
        append(&textures, texture)
        free_slot = u32(len(textures)) - 1
    }
    return Texture_Handle { idx = free_slot, gen = texture.gen }
}

register_texture :: proc(using ctx: ^Context, img: Texture) -> Texture_Handle
{
    return {}
}

get_texture :: proc(using ctx: ^Context, handle: Texture_Handle) -> ^Texture
{
    if u32(len(textures)) <= handle.idx do return nil
    if textures[handle.idx].gen != handle.gen do return nil
    return &textures[handle.idx]
}

release_texture :: proc(using ctx: ^Context, handle: Texture_Handle)
{
    if u32(len(textures)) <= handle.idx do return
    if textures[handle.idx].gen != handle.gen do return
    textures[handle.idx].gen += 1
    if u32(len(textures)) - 1 == handle.idx {
        pop(&textures)
    } else {
        append(&textures_free, handle.idx)
    }
}

// This procedure is immediate mode, and can be called
// every frame with little performance impact if no changes
// occur. (State is diffed across calls). Will update the TLAS
// if things change across calls
update_scene :: proc(bake: ^Bake, instances: []Instance)
{

}

Instance :: struct
{
    transform: matrix[4, 4]f32,
    mesh: Mesh_Handle,
    lm_idx: u32,
    lm_offset: [2]f32,
    lm_scale: f32,
}

Mesh :: struct
{
    gen: u32,

    indices_cpu: [dynamic]u32,
    pos_cpu: [dynamic][3]f32,
    normals_cpu: [dynamic][3]f32,
    geom_normals_cpu: [dynamic][3]f32,
    lm_uvs_cpu: [dynamic][2]f32,

    seams: Buffer,
    seams_desc_set: vk.DescriptorSet,
    num_seams: u32,

    indices: Buffer,
    pos: Buffer,
    normals: Buffer,
    geom_normals: Buffer,
    lm_uvs: Buffer,
    blas: vku.Accel_Structure,
}

Seam_GPU :: struct
{
    lines: [2]Line_GPU
}

Line_GPU :: struct
{
    p: [2][2]f32,
}

// update_scene_structures

// Baking

Bake :: struct
{
    using ctx: ^Context,
    thread: ^thr.Thread,

    // Resources
    mutex: ^sync.Mutex,
    sem: vk.Semaphore,
    lightmap_backbuffer: Image,

    instances: [dynamic]Instance,
    tlas: Tlas,

    // OIDN Resources
    oidn_buf: oidn.Buffer,
    vk_external_buf: External_Buf,
    filter: oidn.Filter,

    // Synchronization counters
    submission_counter: u64,
    last_pathtrace_value: u64,
    last_usage_value: u64,

    // Settings
    num_accums: u32,
    lightmap_size: u32,
    use_dir_light: bool,
    dir_light: Dir_Light,

    // For renderdoc debugging
    debug_mutex0: ^sync.Mutex,
    debug_mutex1: ^sync.Mutex,
}

// Starts a baking process in a separate thread.
//
// Inputs:
// - lightmap_size: Size in pixels of the lightmap to be built.
// - num_accums: Number of accumulations for pathtracing.
// - num_samples_per_pixel: Number of pathtrace samples per pixel done on each accumulation.
start_bake :: proc(using ctx: ^Context, instances: []Instance, use_dir_light: bool, dir_light: Dir_Light,
                   lightmap_size: u32 = 4096, num_accums: u32 = 400, num_samples_per_pixel: u32 = 1,
                   ) -> ^Bake
{
    bake := new(Bake)
    bake.ctx = ctx

    bake.num_accums = num_accums
    bake.lightmap_size = lightmap_size
    bake.use_dir_light = true
    bake.dir_light = dir_light

    bake.thread = thr.create(bake_thread)
    bake.thread.init_context = context
    bake.thread.user_index = 0
    bake.thread.data = bake
    bake.instances = slice.clone_to_dynamic(instances)

    bake.mutex = new(sync.Mutex)
    bake.debug_mutex0 = new(sync.Mutex)
    bake.debug_mutex1 = new(sync.Mutex)
    sync.mutex_lock(bake.debug_mutex1)

    next: rawptr
    next = &vk.SemaphoreTypeCreateInfo {
        sType = .SEMAPHORE_TYPE_CREATE_INFO,
        pNext = next,
        semaphoreType = .TIMELINE,
        initialValue = 0,
    }
    sem_ci := vk.SemaphoreCreateInfo {
        sType = .SEMAPHORE_CREATE_INFO,
        pNext = next
    }
    vk_check(vk.CreateSemaphore(device, &sem_ci, nil, &bake.sem))

    // Tlas
    bake.tlas = create_tlas(device, phys_device, queue, upload_cmd_pool, instances[:], meshes[:])

    // Lightmap backbuffer
    cmd_buf := vku.begin_tmp_cmd_buf(device, upload_cmd_pool)

    bake.lightmap_backbuffer = vku.create_image(device, phys_device, cmd_buf, {
        sType = .IMAGE_CREATE_INFO,
        flags = {},
        imageType = .D2,
        format = .R16G16B16A16_SFLOAT,
        extent = {
            width = lightmap_size,
            height = lightmap_size,
            depth = 1,
        },
        mipLevels = 1,
        arrayLayers = 1,
        samples = { ._1 },
        usage = { .STORAGE, .SAMPLED, .TRANSFER_DST, .TRANSFER_SRC, .COLOR_ATTACHMENT },
        sharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &vk_ctx.queue_family_idx,
        initialLayout = .UNDEFINED,
    })

    vku.end_tmp_cmd_buf(device, upload_cmd_pool, queue, cmd_buf)

    thr.start(bake.thread)
    return bake
}

// Reports the process, in percentage, in [0, 1].
progress :: proc(using bake: ^Bake) -> f32
{
    return 0.0
}

// Stops the baking process before it is complete.
stop_bake :: proc(using bake: ^Bake)
{

}

pause_bake :: proc(using bake: ^Bake)
{

}

// Stops the current thread until the entire bake is finished.
wait_end_of_bake :: proc(using bake: ^Bake)
{
    thr.join(bake.thread)
}

// Cleans up all temporary resources linked to the
// lightmap baking process.
// Must be called after each start_bake!
cleanup_bake :: proc(using bake: ^Bake)
{
    free(mutex)
    free(bake)

    // Free vulkan resources
    // Free OIDN resources
}

// Semaphore is owned by this library and shouldn't be destroyed.
View_Info :: struct
{
    sem: vk.Semaphore,
    wait_value: u64,
    signal_value: u64,
}
acquire_next_lightmap_view_vk :: proc(using bake: ^Bake) -> View_Info
{
    res: View_Info

    if sync.mutex_guard(mutex)
    {
        submission_counter += 1
        res.signal_value = submission_counter
        res.wait_value = last_pathtrace_value
        last_usage_value = res.signal_value
    }

    res.sem = sem
    return res
}

App_Vulkan_Context :: struct
{
    instance: vk.Instance,
    debug_messenger: vk.DebugUtilsMessengerEXT,
    surface: vk.SurfaceKHR,

    phys_device: vk.PhysicalDevice,
    device: vk.Device,
    queue: vk.Queue,
    lm_queue: vk.Queue,
    queue_family_idx: u32,
}

// Utility function, could be used to initialize Vulkan
// with all the appropriate extensions for this library.
init_vk_context :: proc(window: ^sdl.Window, debug_callback: vk.ProcDebugUtilsMessengerCallbackEXT) -> App_Vulkan_Context
{
    res: App_Vulkan_Context

    // Create instance
    {
        when ODIN_DEBUG
        {
            layers := []cstring {
                "VK_LAYER_KHRONOS_validation",
            }
        }
        else
        {
            layers := []cstring {}
        }

        count: u32
        instance_extensions := sdl.Vulkan_GetInstanceExtensions(&count)
        extensions := slice.concatenate([][]cstring {
            instance_extensions[:count],

            {
                vk.EXT_DEBUG_UTILS_EXTENSION_NAME,
                vk.KHR_WIN32_SURFACE_EXTENSION_NAME,
            }
        }, context.temp_allocator)

        debug_messenger_ci := vk.DebugUtilsMessengerCreateInfoEXT {
            sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            messageSeverity = { .WARNING, .ERROR },
            messageType = { .VALIDATION, .PERFORMANCE },
            pfnUserCallback = debug_callback
        }

        validation_features := []vk.ValidationFeatureEnableEXT {
            //.GPU_ASSISTED,
            //.GPU_ASSISTED_RESERVE_BINDING_SLOT,
            .SYNCHRONIZATION_VALIDATION,
        }

        next: rawptr
        next = &debug_messenger_ci
        next = &vk.ValidationFeaturesEXT {
            sType = .VALIDATION_FEATURES_EXT,
            pNext = next,
            enabledValidationFeatureCount = u32(len(validation_features)),
            pEnabledValidationFeatures = raw_data(validation_features),
        }

        vk_check(vk.CreateInstance(&{
            sType = .INSTANCE_CREATE_INFO,
            pApplicationInfo = &{
                sType = .APPLICATION_INFO,
                apiVersion = vk.API_VERSION_1_3,
            },
            enabledLayerCount = u32(len(layers)),
            ppEnabledLayerNames = raw_data(layers),
            enabledExtensionCount = u32(len(extensions)),
            ppEnabledExtensionNames = raw_data(extensions),
            pNext = next,
        }, nil, &res.instance))

        vk.load_proc_addresses_instance(res.instance)
        assert(vk.DestroyInstance != nil, "Failed to load Vulkan instance API")

        vk_check(vk.CreateDebugUtilsMessengerEXT(res.instance, &debug_messenger_ci, nil, &res.debug_messenger))
    }

    // Create surface
    {
        ok_s := sdl.Vulkan_CreateSurface(window, res.instance, nil, &res.surface)
        if !ok_s do fatal_error("Could not create vulkan surface.")
    }

    // Physical device
    phys_device_count: u32
    vk_check(vk.EnumeratePhysicalDevices(res.instance, &phys_device_count, nil))
    if phys_device_count == 0 do fatal_error("Did not find any GPUs!")
    phys_devices := make([]vk.PhysicalDevice, phys_device_count, context.temp_allocator)
    vk_check(vk.EnumeratePhysicalDevices(res.instance, &phys_device_count, raw_data(phys_devices)))

    chosen_phys_device: vk.PhysicalDevice
    queue_family_idx: u32
    found := false
    device_loop: for candidate in phys_devices
    {
        queue_family_count: u32
        vk.GetPhysicalDeviceQueueFamilyProperties(candidate, &queue_family_count, nil)
        queue_families := make([]vk.QueueFamilyProperties, queue_family_count, context.temp_allocator)
        vk.GetPhysicalDeviceQueueFamilyProperties(candidate, &queue_family_count, raw_data(queue_families))

        for family, i in queue_families
        {
            supports_graphics := .GRAPHICS in family.queueFlags
            supports_present: b32
            vk_check(vk.GetPhysicalDeviceSurfaceSupportKHR(candidate, u32(i), res.surface, &supports_present))

            if supports_graphics && supports_present
            {
                chosen_phys_device = candidate
                queue_family_idx = u32(i)
                found = true
                break device_loop
            }
        }
    }

    if !found do fatal_error("Could not find suitable GPU.")

    res.phys_device = chosen_phys_device

    queue_priority := f32(1)
    queue_create_infos := []vk.DeviceQueueCreateInfo {
        {
            sType = .DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex = queue_family_idx,
            queueCount = 2,
            pQueuePriorities = &queue_priority,
        },
    }

    // Device
    device_extensions := []cstring {
        vk.KHR_SWAPCHAIN_EXTENSION_NAME,
        vk.EXT_SHADER_OBJECT_EXTENSION_NAME,
        vk.KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        vk.KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        vk.KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        vk.EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME,
        vk.KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME,
        vk.KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    }

    next: rawptr
    /*
    next = &vk.PhysicalDeviceMaintenance6Features {
        sType = .PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES,
        pNext = next,
        maintenance6 = true,
    }
    */
    next = &vk.PhysicalDeviceVulkan13Features {
        sType = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        pNext = next,
        dynamicRendering = true,
        synchronization2 = true,
    }
    next = &vk.PhysicalDeviceShaderObjectFeaturesEXT {
        sType = .PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
        pNext = next,
        shaderObject = true,
    }
    next = &vk.PhysicalDeviceDepthClipEnableFeaturesEXT {
        sType = .PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT,
        pNext = next,
        depthClipEnable = true,
    }
    next = &vk.PhysicalDeviceAccelerationStructureFeaturesKHR {
        sType = .PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        pNext = next,
        accelerationStructure = true,
    }
    next = &vk.PhysicalDeviceRayTracingPipelineFeaturesKHR {
        sType = .PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        pNext = next,
        rayTracingPipeline = true,
    }
    next = &vk.PhysicalDeviceBufferDeviceAddressFeatures {
        sType = .PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
        pNext = next,
        bufferDeviceAddress = true,
    }
    next = &vk.PhysicalDeviceTimelineSemaphoreFeatures {
        sType = .PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        pNext = next,
        timelineSemaphore = true,
    }
    next = &vk.PhysicalDeviceFeatures2 {
        sType = .PHYSICAL_DEVICE_FEATURES_2,
        pNext = next,
        features = {
            geometryShader = true,  // For the tri_idx gbuffer.
        }
    }
    next = &vk.PhysicalDeviceRayTracingPositionFetchFeaturesKHR {
        sType = .PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
        pNext = next,
        rayTracingPositionFetch = true,
    }

    device_ci := vk.DeviceCreateInfo {
        sType = .DEVICE_CREATE_INFO,
        pNext = next,
        queueCreateInfoCount = u32(len(queue_create_infos)),
        pQueueCreateInfos = raw_data(queue_create_infos),
        enabledExtensionCount = u32(len(device_extensions)),
        ppEnabledExtensionNames = raw_data(device_extensions),
    }
    vk_check(vk.CreateDevice(chosen_phys_device, &device_ci, nil, &res.device))

    vk.load_proc_addresses_device(res.device)
    if vk.BeginCommandBuffer == nil do fatal_error("Failed to load Vulkan device API")

    vk.GetDeviceQueue(res.device, queue_family_idx, 0, &res.queue)
    vk.GetDeviceQueue(res.device, queue_family_idx, 1, &res.lm_queue)
    return res
}

destroy_vk_context :: proc(using vk_ctx: ^App_Vulkan_Context)
{
    sdl.Vulkan_DestroySurface(instance, surface, nil)
    if debug_messenger != 0 do vk.DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nil)

    vk.DestroyInstance(instance, nil)
}

///////////////////////////////////
// Internals

Lightmapper_Vulkan_Context :: struct
{
    device: vk.Device,
    phys_device: vk.PhysicalDevice,
    queue: vk.Queue,  // NOTE: Use a separate queue from the main render queue!
    queue_family_idx: u32
}

Geometry :: struct
{
    normals: vk.DeviceAddress,
    indices: vk.DeviceAddress,
}

bake_thread :: proc(t: ^thr.Thread)
{
    ctx := cast(^Bake) t.data
    bake_main(ctx)
}

bake_main :: proc(using bake: ^Bake)
{
    // Setup

    // Create queue and command buffers
    cmd_pool_ci := vk.CommandPoolCreateInfo {
        sType = .COMMAND_POOL_CREATE_INFO,
        queueFamilyIndex = queue_family_idx,
        flags = { .RESET_COMMAND_BUFFER }
    }
    cmd_pool: vk.CommandPool
    vk_check(vk.CreateCommandPool(device, &cmd_pool_ci, nil, &cmd_pool))
    defer vk.DestroyCommandPool(device, cmd_pool, nil)

    fence_ci := vk.FenceCreateInfo {
        sType = .FENCE_CREATE_INFO,
        flags = { .SIGNALED },
    }
    fence: vk.Fence
    vk_check(vk.CreateFence(device, &fence_ci, nil, &fence))

    cmd_buf_ai := vk.CommandBufferAllocateInfo {
        sType = .COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool = cmd_pool,
        level = .PRIMARY,
        commandBufferCount = 1,
    }
    cmd_buf: vk.CommandBuffer
    vk_check(vk.AllocateCommandBuffers(device, &cmd_buf_ai, &cmd_buf))
    defer vk.FreeCommandBuffers(device, cmd_pool, 1, &cmd_buf)

    vk_check(vk.BeginCommandBuffer(cmd_buf, &{
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        flags = { .ONE_TIME_SUBMIT },
    }))

    gbufs := create_gbuffers(bake, cmd_buf)

    lightmap := vku.create_image(device, phys_device, cmd_buf, {
        sType = .IMAGE_CREATE_INFO,
        flags = {},
        imageType = .D2,
        format = .R16G16B16A16_SFLOAT,
        extent = {
            width = lightmap_size,
            height = lightmap_size,
            depth = 1,
        },
        mipLevels = 1,
        arrayLayers = 1,
        samples = { ._1 },
        usage = { .STORAGE, .SAMPLED, .TRANSFER_SRC, .COLOR_ATTACHMENT, .TRANSFER_DST },
        sharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &vk_ctx.queue_family_idx,
        initialLayout = .UNDEFINED,
    })

    // Create geometries buffer
    geoms_buf: Buffer
    defer vku.destroy_buffer(device, &geoms_buf)
    {
        geoms := make([]Geometry, len(meshes))
        defer delete(geoms)

        for &geom, i in geoms
        {
            if meshes[i].pos.handle == vk.Buffer(0) do continue  // Guard for invalid/released meshes

            geom.normals = vku.get_buffer_device_address(device, meshes[i].normals)
            geom.indices = vku.get_buffer_device_address(device, meshes[i].indices)
        }

        usage := vk.BufferUsageFlags { .TRANSFER_DST, .SHADER_DEVICE_ADDRESS, .STORAGE_BUFFER }
        properties := vk.MemoryPropertyFlags { .DEVICE_LOCAL }
        allocate_flags := vk.MemoryAllocateFlags { .DEVICE_ADDRESS }
        geoms_buf = vku.upload_buffer(device, phys_device, queue, upload_cmd_pool, geoms, usage, properties, allocate_flags)
    }

    // Create SBT buffers
    pathtrace_sbt    := vku.create_sbt_buffer(device, phys_device, queue, cmd_pool, shaders.rt_pipeline, 3)
    defer vku.destroy_buffer(device, &pathtrace_sbt)
    push_samples_sbt := vku.create_sbt_buffer(device, phys_device, queue, cmd_pool, shaders.push_samples_pipeline, 3)
    defer vku.destroy_buffer(device, &push_samples_sbt)

    desc_set_ai := vk.DescriptorSetAllocateInfo {
        sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool = desc_pool,
        descriptorSetCount = 1,
        pSetLayouts = raw_data([]vk.DescriptorSetLayout { shaders.rt_desc_set_layout })
    }
    rt_desc_set: vk.DescriptorSet
    vk_check(vk.AllocateDescriptorSets(device, &desc_set_ai, &rt_desc_set))
    //defer vk.FreeDescriptorSets(device, 1, &rt_desc_set, nil)
    update_rt_desc_set(device, rt_desc_set, tlas.as.handle, lightmap, gbufs, geoms_buf)

    // Dummy sampler
    dummy_sampler_ci := vk.SamplerCreateInfo {
        sType = .SAMPLER_CREATE_INFO,
        magFilter = .LINEAR,
        minFilter = .LINEAR,
        mipmapMode = .LINEAR,
        addressModeU = .REPEAT,
        addressModeV = .REPEAT,
        addressModeW = .REPEAT,
    }
    dummy_sampler: vk.Sampler
    vk_check(vk.CreateSampler(vk_ctx.device, &dummy_sampler_ci, nil, &dummy_sampler))
    defer vk.DestroySampler(device, dummy_sampler, nil)

    // Dilate desc set
    dilate_desc_set_ai := vk.DescriptorSetAllocateInfo {
        sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool = desc_pool,
        descriptorSetCount = 1,
        pSetLayouts = raw_data([]vk.DescriptorSetLayout { shaders.dilate_desc_set_layout })
    }
    dilate_desc_set: [2]vk.DescriptorSet  // Front to back, back to front.
    vk_check(vk.AllocateDescriptorSets(device, &dilate_desc_set_ai, &dilate_desc_set[0]))
    vk_check(vk.AllocateDescriptorSets(device, &dilate_desc_set_ai, &dilate_desc_set[1]))
    update_dilate_desc_set(device, dilate_desc_set[0], dummy_sampler, lightmap, lightmap_backbuffer)
    update_dilate_desc_set(device, dilate_desc_set[1], dummy_sampler, lightmap_backbuffer, lightmap)

    when SYNCHRONOUS {
        sync.mutex_lock(debug_mutex0)
    }

    render_gbuffers(bake, cmd_buf, &gbufs, push_samples_sbt, rt_desc_set)

    vk_check(vk.EndCommandBuffer(cmd_buf))

    {
        wait_stage_flags := vk.PipelineStageFlags { .COLOR_ATTACHMENT_OUTPUT }
        submit_info := vk.SubmitInfo {
            sType = .SUBMIT_INFO,
            pWaitDstStageMask = &wait_stage_flags,
            commandBufferCount = 1,
            pCommandBuffers = &cmd_buf,
        }
        vk_check(vk.QueueSubmit(vk_ctx.queue, 1, &submit_info, {}))
        vk_check(vk.QueueWaitIdle(vk_ctx.queue))
    }

    vk_external_buf = create_vk_external_buffer_for_oidn(&vk_ctx, lightmap_size * lightmap_size * 2 * 4)
    oidn_buf = oidn_shared_buffer_from_vk_buffer(oidn_device, vk_external_buf)
    filter = oidn.NewFilter(oidn_device, "RTLightmap")
    oidn.SetFilterImage(filter, "color", oidn_buf, .HALF3, auto_cast lightmap_size, auto_cast lightmap_size, pixelByteStride = 2 * 4)
    oidn.SetFilterImage(filter, "output", oidn_buf, .HALF3, auto_cast lightmap_size, auto_cast lightmap_size, pixelByteStride = 2 * 4)
    oidn.CommitFilter(filter)
    oidn_check(oidn_device)

    vk_check(vk.WaitForFences(vk_ctx.device, 1, &fence, true, max(u64)))
    vk_check(vk.ResetFences(vk_ctx.device, 1, &fence))
    vk_check(vk.ResetCommandPool(vk_ctx.device, cmd_pool, {}))

    // Pathtrace loop
    for iter in 0..<num_accums
    {
        when SYNCHRONOUS
        {
            if iter > 0 do sync.mutex_lock(debug_mutex0)
            defer { sync.mutex_unlock(debug_mutex1) }
        }

        // fmt.println("pathtrace iter")

        vk_check(vk.BeginCommandBuffer(cmd_buf, &{
            sType = .COMMAND_BUFFER_BEGIN_INFO,
            flags = { .ONE_TIME_SUBMIT },
        }))

        pathtrace_iter(bake, cmd_buf, pathtrace_sbt, rt_desc_set, iter)

        vku.image_barrier_safe_slow(&lightmap, cmd_buf, .GENERAL)

        // Blit to backbuffer, dilating the image in the process
        images := [2]^Image { &lightmap, &lightmap_backbuffer }
        src_idx := 0
        dst_idx := 1
        for _ in 0..<2
        {
            src_image := images[src_idx]
            dst_image := images[dst_idx]
            desc_set  := dilate_desc_set[src_idx]  // front to back if 0, back to front if 1.
            defer
            {
                tmp := src_idx
                src_idx = dst_idx
                dst_idx = tmp
            }

            vku.image_barrier_safe_slow(dst_image, cmd_buf, .GENERAL)

            shader_stages := vk.ShaderStageFlags { .COMPUTE }
            vk.CmdBindShadersEXT(cmd_buf, 1, &shader_stages, &shaders.dilate_shader)
            vk.CmdBindDescriptorSets(cmd_buf, .COMPUTE, shaders.dilate_pipeline_layout, 0, 1, &desc_set, 0, nil)

            // NOTE: Coupled to shader code.
            GROUP_SIZE :: u32(8)
            vk.CmdDispatch(cmd_buf, lightmap_size / GROUP_SIZE, lightmap_size / GROUP_SIZE, 1)

            vku.image_barrier_safe_slow(dst_image, cmd_buf, .GENERAL)
            vku.image_barrier_safe_slow(src_image, cmd_buf, .GENERAL)
        }

        vk_check(vk.EndCommandBuffer(cmd_buf))

        wait_value: u64
        signal_value: u64
        if sync.mutex_guard(mutex)
        {
            submission_counter += 1
            wait_value = last_usage_value
            signal_value = submission_counter
            last_pathtrace_value = signal_value
        }

        // fmt.println("pathtrace waiting on", wait_value, "and signaling", signal_value)

        wait_stage_flags := vk.PipelineStageFlags { .COLOR_ATTACHMENT_OUTPUT }
        next: rawptr
        next = &vk.TimelineSemaphoreSubmitInfo {
            sType = .TIMELINE_SEMAPHORE_SUBMIT_INFO,
            pNext = next,
            waitSemaphoreValueCount = 1,
            pWaitSemaphoreValues = &wait_value,
            signalSemaphoreValueCount = 1,
            pSignalSemaphoreValues = &signal_value,
        }
        submit_info := vk.SubmitInfo {
            sType = .SUBMIT_INFO,
            pNext = next,
            pWaitDstStageMask = &wait_stage_flags,
            commandBufferCount = 1,
            pCommandBuffers = &cmd_buf,
            waitSemaphoreCount = 1,
            pWaitSemaphores = &sem,
            signalSemaphoreCount = 1,
            pSignalSemaphores = &sem,
        }
        vk_check(vk.QueueSubmit(vk_ctx.queue, 1, &submit_info, fence))

        vk_check(vk.WaitForFences(vk_ctx.device, 1, &fence, true, max(u64)))
        vk_check(vk.ResetFences(vk_ctx.device, 1, &fence))
        vk_check(vk.ResetCommandPool(vk_ctx.device, cmd_pool, {}))
    }

    // Denoise
    {
        when SYNCHRONOUS
        {
            sync.mutex_lock(debug_mutex0)
            defer sync.mutex_unlock(debug_mutex1)
        }

        vk_check(vk.BeginCommandBuffer(cmd_buf, &{
            sType = .COMMAND_BUFFER_BEGIN_INFO,
            flags = { .ONE_TIME_SUBMIT },
        }))

        vk.CmdCopyImageToBuffer2(cmd_buf, &vk.CopyImageToBufferInfo2 {
            sType = .COPY_IMAGE_TO_BUFFER_INFO_2,
            pNext = nil,
            srcImage = lightmap_backbuffer.img,
            srcImageLayout = .GENERAL,
            dstBuffer = vk_external_buf.buf.handle,
            regionCount = 1,
            pRegions = &vk.BufferImageCopy2 {
                sType = .BUFFER_IMAGE_COPY_2,
                bufferRowLength = 0,
                bufferImageHeight = 0,
                imageSubresource = vk.ImageSubresourceLayers {
                    aspectMask = { .COLOR },
                    layerCount = 1,
                },
                imageExtent = vk.Extent3D {
                    width = lightmap_size,
                    height = lightmap_size,
                    depth = 1,
                },
            },
        })

        vk_check(vk.EndCommandBuffer(cmd_buf))

        wait_stage_flags := vk.PipelineStageFlags { .ALL_COMMANDS }
        submit_info := vk.SubmitInfo {
            sType = .SUBMIT_INFO,
            pWaitDstStageMask = &wait_stage_flags,
            commandBufferCount = 1,
            pCommandBuffers = &cmd_buf,
        }
        vk_check(vk.QueueSubmit(vk_ctx.queue, 1, &submit_info, {}))
        vk_check(vk.QueueWaitIdle(vk_ctx.queue))

        oidn_run_lightmap_filter(oidn_device, filter)
        oidn.SyncDevice(oidn_device)

        vk_check(vk.BeginCommandBuffer(cmd_buf, &{
            sType = .COMMAND_BUFFER_BEGIN_INFO,
            flags = { .ONE_TIME_SUBMIT },
        }))

        vk.CmdCopyBufferToImage2(cmd_buf, &vk.CopyBufferToImageInfo2 {
            sType = .COPY_BUFFER_TO_IMAGE_INFO_2,
            pNext = nil,
            srcBuffer = vk_external_buf.buf.handle,
            dstImage = lightmap_backbuffer.img,
            dstImageLayout = .GENERAL,
            regionCount = 1,
            pRegions = &vk.BufferImageCopy2 {
                sType = .BUFFER_IMAGE_COPY_2,
                bufferRowLength = 0,
                bufferImageHeight = 0,
                imageSubresource = vk.ImageSubresourceLayers {
                    aspectMask = { .COLOR },
                    layerCount = 1,
                },
                imageExtent = vk.Extent3D {
                    width = lightmap_size,
                    height = lightmap_size,
                    depth = 1,
                },
            },
        })

        vk_check(vk.EndCommandBuffer(cmd_buf))

        wait_stage_flags = vk.PipelineStageFlags { .ALL_COMMANDS }
        submit_info = vk.SubmitInfo {
            sType = .SUBMIT_INFO,
            pWaitDstStageMask = &wait_stage_flags,
            commandBufferCount = 1,
            pCommandBuffers = &cmd_buf,
        }
        vk_check(vk.QueueSubmit(vk_ctx.queue, 1, &submit_info, {}))
        vk_check(vk.QueueWaitIdle(vk_ctx.queue))
    }

    // Smooth seams
    {
        smooth_seams(bake, cmd_buf, shaders.seams_pipeline_layout, &lightmap, &lightmap_backbuffer)
    }

    when SYNCHRONOUS
    {
        for
        {
            sync.mutex_lock(debug_mutex0)
            sync.mutex_unlock(debug_mutex1)
        }
    }
}

render_gbuffers :: proc(using bake: ^Bake, cmd_buf: vk.CommandBuffer, gbuffers: ^GBuffers, push_samples_sbt: Buffer, rt_desc_set: vk.DescriptorSet)
{
    vert_input_bindings := []vk.VertexInputBindingDescription2EXT {
        {  // Positions
            sType = .VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            binding = 0,
            stride = size_of([3]f32),
            inputRate = .VERTEX,
            divisor = 1,
        },
        {  // Normals
            sType = .VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            binding = 1,
            stride = size_of([3]f32),
            inputRate = .VERTEX,
            divisor = 1,
        },
        {  // Lightmap UVs
            sType = .VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            binding = 2,
            stride = size_of([2]f32),
            inputRate = .VERTEX,
            divisor = 1,
        },
    }
    vert_attributes := []vk.VertexInputAttributeDescription2EXT {
        {
            sType = .VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            location = 0,
            binding = 0,
            format = .R32G32B32_SFLOAT,
            offset = 0
        },
        {
            sType = .VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            location = 1,
            binding = 1,
            format = .R32G32B32_SFLOAT,
            offset = 0
        },
        {
            sType = .VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            location = 2,
            binding = 2,
            format = .R32G32_SFLOAT,
            offset = 0
        },
    }
    vk.CmdSetVertexInputEXT(cmd_buf, u32(len(vert_input_bindings)), raw_data(vert_input_bindings), u32(len(vert_attributes)), raw_data(vert_attributes))
    vk.CmdSetPrimitiveTopology(cmd_buf, .TRIANGLE_LIST)
    vk.CmdSetPrimitiveRestartEnable(cmd_buf, false)

    // World pos
    {
        color_attachment := vk.RenderingAttachmentInfo {
            sType = .RENDERING_ATTACHMENT_INFO,
            imageView = gbuffers.world_pos.view,
            imageLayout = gbuffers.world_normals.layout,
            loadOp = .CLEAR,
            storeOp = .STORE,
            clearValue = {
                color = { float32 = { 0.0, 0.0, 0.0, 0.0 } }
            }
        }
        rendering_info := vk.RenderingInfo {
            sType = .RENDERING_INFO,
            renderArea = {
                offset = { 0, 0 },
                extent = { gbuffers.world_pos.width, gbuffers.world_pos.height }
            },
            layerCount = 1,
            colorAttachmentCount = 1,
            pColorAttachments = &color_attachment,
            pDepthAttachment = nil,
        }

        vk.CmdBeginRendering(cmd_buf, &rendering_info)
        defer vk.CmdEndRendering(cmd_buf)

        shader_stages := []vk.ShaderStageFlags { { .VERTEX }, { .GEOMETRY }, { .FRAGMENT } }
        to_bind := []vk.ShaderEXT { shaders.uv_space, vk.ShaderEXT(0), shaders.gbuffer_world_pos }
        assert(len(shader_stages) == len(to_bind))
        vk.CmdBindShadersEXT(cmd_buf, u32(len(shader_stages)), raw_data(shader_stages), raw_data(to_bind))

        draw_gbuffer(bake, cmd_buf, shaders.pipeline_layout)
    }

    // World normal
    {
        color_attachment := vk.RenderingAttachmentInfo {
            sType = .RENDERING_ATTACHMENT_INFO,
            imageView = gbuffers.world_normals.view,
            imageLayout = gbuffers.world_normals.layout,
            loadOp = .CLEAR,
            storeOp = .STORE,
            clearValue = {
                color = { float32 = { 0.0, 0.0, 0.0, 0.0 } }
            }
        }
        rendering_info := vk.RenderingInfo {
            sType = .RENDERING_INFO,
            renderArea = {
                offset = { 0, 0 },
                extent = { gbuffers.world_normals.width, gbuffers.world_normals.height }
            },
            layerCount = 1,
            colorAttachmentCount = 1,
            pColorAttachments = &color_attachment,
            pDepthAttachment = nil,
        }

        vk.CmdBeginRendering(cmd_buf, &rendering_info)
        defer vk.CmdEndRendering(cmd_buf)

        shader_stages := []vk.ShaderStageFlags { { .VERTEX }, { .GEOMETRY }, { .FRAGMENT } }
        to_bind := []vk.ShaderEXT { shaders.uv_space, vk.ShaderEXT(0), shaders.gbuffer_world_normals }
        assert(len(shader_stages) == len(to_bind))
        vk.CmdBindShadersEXT(cmd_buf, u32(len(shader_stages)), raw_data(shader_stages), raw_data(to_bind))

        draw_gbuffer(bake, cmd_buf, shaders.pipeline_layout)
    }

    // World geometric normal
    {
        color_attachment := vk.RenderingAttachmentInfo {
            sType = .RENDERING_ATTACHMENT_INFO,
            imageView = gbuffers.world_geom_normals.view,
            imageLayout = gbuffers.world_geom_normals.layout,
            loadOp = .CLEAR,
            storeOp = .STORE,
            clearValue = {
                color = { float32 = { 0.0, 0.0, 0.0, 0.0 } }
            }
        }
        rendering_info := vk.RenderingInfo {
            sType = .RENDERING_INFO,
            renderArea = {
                offset = { 0, 0 },
                extent = { gbuffers.world_geom_normals.width, gbuffers.world_geom_normals.height }
            },
            layerCount = 1,
            colorAttachmentCount = 1,
            pColorAttachments = &color_attachment,
            pDepthAttachment = nil,
        }

        vk.CmdBeginRendering(cmd_buf, &rendering_info)
        defer vk.CmdEndRendering(cmd_buf)

        shader_stages := []vk.ShaderStageFlags { { .VERTEX }, { .GEOMETRY }, { .FRAGMENT } }
        to_bind := []vk.ShaderEXT { shaders.uv_space, vk.ShaderEXT(0), shaders.gbuffer_world_normals }
        assert(len(shader_stages) == len(to_bind))
        vk.CmdBindShadersEXT(cmd_buf, u32(len(shader_stages)), raw_data(shader_stages), raw_data(to_bind))

        draw_gbuffer(bake, cmd_buf, shaders.pipeline_layout, use_geom_normals = true)
    }

    // Tri idx
    {
        color_attachment := vk.RenderingAttachmentInfo {
            sType = .RENDERING_ATTACHMENT_INFO,
            imageView = gbuffers.tri_idx.view,
            imageLayout = gbuffers.tri_idx.layout,
            loadOp = .CLEAR,
            storeOp = .STORE,
            clearValue = {
                color = { uint32 = { max(u32), max(u32), max(u32), max(u32) } }
            }
        }
        rendering_info := vk.RenderingInfo {
            sType = .RENDERING_INFO,
            renderArea = {
                offset = { 0, 0 },
                extent = { gbuffers.world_pos.width, gbuffers.world_pos.height }
            },
            layerCount = 1,
            colorAttachmentCount = 1,
            pColorAttachments = &color_attachment,
            pDepthAttachment = nil,
        }

        vk.CmdBeginRendering(cmd_buf, &rendering_info)
        defer vk.CmdEndRendering(cmd_buf)

        shader_stages := []vk.ShaderStageFlags { { .VERTEX }, { .GEOMETRY }, { .FRAGMENT } }
        to_bind := []vk.ShaderEXT { shaders.uv_space, vk.ShaderEXT(0), shaders.gbuffer_tri_idx }
        assert(len(shader_stages) == len(to_bind))
        vk.CmdBindShadersEXT(cmd_buf, u32(len(shader_stages)), raw_data(shader_stages), raw_data(to_bind))

        draw_gbuffer(bake, cmd_buf, shaders.pipeline_layout)
    }

    gbuffers_barrier(gbuffers, cmd_buf)

    push_samples_outside_geometry(bake, cmd_buf, gbuffers, push_samples_sbt, rt_desc_set)
}

draw_gbuffer :: proc(using bake: ^Bake, cmd_buf: vk.CommandBuffer, pipeline_layout: vk.PipelineLayout, use_geom_normals := false)
{
    sample_mask := vk.SampleMask(1)
    vk.CmdSetSampleMaskEXT(cmd_buf, { ._1 }, &sample_mask)
    vk.CmdSetAlphaToCoverageEnableEXT(cmd_buf, false)

    vk.CmdSetPolygonModeEXT(cmd_buf, .FILL)
    vk.CmdSetCullMode(cmd_buf, {})
    vk.CmdSetFrontFace(cmd_buf, .COUNTER_CLOCKWISE)

    vk.CmdSetDepthCompareOp(cmd_buf, .LESS)
    vk.CmdSetDepthTestEnable(cmd_buf, false)
    vk.CmdSetDepthWriteEnable(cmd_buf, false)
    vk.CmdSetDepthBiasEnable(cmd_buf, false)
    vk.CmdSetDepthClipEnableEXT(cmd_buf, true)

    vk.CmdSetStencilTestEnable(cmd_buf, false)
    b32_false := b32(false)
    vk.CmdSetColorBlendEnableEXT(cmd_buf, 0, 1, &b32_false)

    color_mask := vk.ColorComponentFlags { .R, .G, .B, .A }
    vk.CmdSetColorWriteMaskEXT(cmd_buf, 0, 1, &color_mask)

    // Perform 2 passes, the first with conservative rasterization and
    // the second one without, which cleans up edges on contiguous triangles.
    for use_conservative_rasterization in ([?]bool{ true, false })
    {
        vk.CmdSetExtraPrimitiveOverestimationSizeEXT(cmd_buf, 0.0)
        vk.CmdSetConservativeRasterizationModeEXT(cmd_buf, .OVERESTIMATE if use_conservative_rasterization else nil)
        vk.CmdSetRasterizationSamplesEXT(cmd_buf, { ._1 })

        for instance in instances
        {
            mesh := get_mesh(ctx, instance.mesh)

            // if !mesh.lm_uvs_present { continue }

            viewport := vk.Viewport {
                x = f32(lightmap_size) * instance.lm_offset.x,
                y = f32(lightmap_size) * instance.lm_offset.y,
                width = f32(lightmap_size) * instance.lm_scale,
                height = f32(lightmap_size) * instance.lm_scale,
                minDepth = 0.0,
                maxDepth = 1.0,
            }
            vk.CmdSetViewportWithCount(cmd_buf, 1, &viewport)
            scissor := vk.Rect2D {
                offset = {
                    x = i32(f32(lightmap_size) * instance.lm_offset.x),
                    y = i32(f32(lightmap_size) * instance.lm_offset.y),
                },
                extent = {
                    width = u32(f32(lightmap_size) * instance.lm_scale),
                    height = u32(f32(lightmap_size) * instance.lm_scale),
                }
            }
            vk.CmdSetScissorWithCount(cmd_buf, 1, &scissor)
            vk.CmdSetRasterizerDiscardEnable(cmd_buf, false)

            offset := vk.DeviceSize(0)
            vk.CmdBindVertexBuffers(cmd_buf, 0, 1, &mesh.pos.handle, &offset)
            if use_geom_normals {
                vk.CmdBindVertexBuffers(cmd_buf, 1, 1, &mesh.geom_normals.handle, &offset)
            } else {
                vk.CmdBindVertexBuffers(cmd_buf, 1, 1, &mesh.normals.handle, &offset)
            }
            // if mesh.lm_uvs_present {
                vk.CmdBindVertexBuffers(cmd_buf, 2, 1, &mesh.lm_uvs.handle, &offset)
            // }
            vk.CmdBindIndexBuffer(cmd_buf, mesh.indices.handle, 0, .UINT32)

            Push :: struct {
                model_to_world: matrix[4, 4]f32,
                normal_mat: matrix[4, 4]f32,
                lm_uv_offset: [2]f32,
                lm_uv_scale: f32
            }
            push := Push {
                model_to_world = instance.transform,
                normal_mat = linalg.transpose(linalg.inverse(instance.transform)),
                lm_uv_offset = instance.lm_offset,
                lm_uv_scale = instance.lm_scale,
            }
            vk.CmdPushConstants(cmd_buf, pipeline_layout, { .VERTEX, .FRAGMENT }, 0, size_of(push), &push)

            vk.CmdDrawIndexed(cmd_buf, u32(len(mesh.indices_cpu)), 1, 0, 0, 0)
        }
    }
}

push_samples_outside_geometry :: proc(using bake: ^Bake, cmd_buf: vk.CommandBuffer, gbuffers: ^GBuffers, sbt: Buffer, rt_desc_set: vk.DescriptorSet)
{
    rt_info := vku.get_rt_info(vk_ctx.phys_device)

    vk.CmdBindPipeline(cmd_buf, .RAY_TRACING_KHR, shaders.push_samples_pipeline)

    tmp := rt_desc_set
    vk.CmdBindDescriptorSets(cmd_buf, .RAY_TRACING_KHR, shaders.rt_pipeline_layout, 0, 1, &tmp, 0, nil)

    sbt_addr := u64(vku.get_buffer_device_address(device, sbt))

    raygen_size   := align_up(rt_info.handle_size, rt_info.handle_align);
    rayhit_size   := align_up(rt_info.handle_size, rt_info.handle_align);
    raymiss_size  := align_up(rt_info.handle_size, rt_info.handle_align);
    callable_size := u32(0)
    raygen_offset := u32(0)
    rayhit_offset := align_up(raygen_offset + raygen_size, rt_info.base_align)
    raymiss_offset := align_up(rayhit_offset + rayhit_size, rt_info.base_align)
    callable_offset := align_up(raymiss_offset + raymiss_size, rt_info.base_align)

    raygen_region := vk.StridedDeviceAddressRegionKHR {
        deviceAddress = vk.DeviceAddress(sbt_addr + u64(raygen_offset)),
        stride = vk.DeviceSize(raygen_size),
        size = vk.DeviceSize(raygen_size),
    }
    raymiss_region := vk.StridedDeviceAddressRegionKHR {
        deviceAddress = vk.DeviceAddress(sbt_addr + u64(raymiss_offset)),
        stride = vk.DeviceSize(raymiss_size),
        size = vk.DeviceSize(raymiss_size),
    }
    rayhit_region := vk.StridedDeviceAddressRegionKHR {
        deviceAddress = vk.DeviceAddress(sbt_addr + u64(rayhit_offset)),
        stride = vk.DeviceSize(rayhit_size),
        size = vk.DeviceSize(rayhit_size),
    }
    callable_region := vk.StridedDeviceAddressRegionKHR {}

    vk.CmdTraceRaysKHR(cmd_buf, &raygen_region, &raymiss_region, &rayhit_region, &callable_region, lightmap_size, lightmap_size, 1)

    vku.image_barrier_safe_slow(&gbuffers.world_pos, cmd_buf, .GENERAL)
}

pathtrace_iter :: proc(using bake: ^Bake, cmd_buf: vk.CommandBuffer, sbt: Buffer, rt_desc_set: vk.DescriptorSet, accum_counter: u32)
{
    rt_info := vku.get_rt_info(vk_ctx.phys_device)

    vk.CmdBindPipeline(cmd_buf, .RAY_TRACING_KHR, shaders.rt_pipeline)

    tmp := rt_desc_set
    vk.CmdBindDescriptorSets(cmd_buf, .RAY_TRACING_KHR, shaders.rt_pipeline_layout, 0, 1, &tmp, 0, nil)

    sbt_addr := u64(vku.get_buffer_device_address(device, sbt))

    data_size := u32(0)
    count := u32(1)
    raygen_size    := align_up(rt_info.handle_size, rt_info.handle_align);
    rayhit_size    := align_up(rt_info.handle_size, rt_info.handle_align);
    raymiss_size   := align_up(rt_info.handle_size, rt_info.handle_align);
    callable_size  := u32(0)

    raygen_offset := u32(0)
    rayhit_offset := align_up(raygen_offset + raygen_size, rt_info.base_align)
    raymiss_offset := align_up(rayhit_offset + rayhit_size, rt_info.base_align)
    callable_offset := align_up(raymiss_offset + raymiss_size, rt_info.base_align)

    raygen_region := vk.StridedDeviceAddressRegionKHR {
        deviceAddress = vk.DeviceAddress(sbt_addr + u64(raygen_offset)),
        stride = vk.DeviceSize(raygen_size),
        size = vk.DeviceSize(raygen_size),
    }
    raymiss_region := vk.StridedDeviceAddressRegionKHR {
        deviceAddress = vk.DeviceAddress(sbt_addr + u64(raymiss_offset)),
        stride = vk.DeviceSize(raymiss_size),
        size = vk.DeviceSize(raymiss_size),
    }
    rayhit_region := vk.StridedDeviceAddressRegionKHR {
        deviceAddress = vk.DeviceAddress(sbt_addr + u64(rayhit_offset)),
        stride = vk.DeviceSize(rayhit_size),
        size = vk.DeviceSize(rayhit_size),
    }
    callable_region := vk.StridedDeviceAddressRegionKHR {}

    Push :: struct {
        accum_counter: u32,
        seed: u32,
        use_dir_light: b32,
        dir_light_angle: f32,
        dir_light: [3]f32,
        padding: u32,
        dir_light_emission: [3]f32,
    }
    push := Push {
        accum_counter = accum_counter,
        seed = 0,
        use_dir_light = b32(use_dir_light),
        dir_light_angle = dir_light.angle,
        dir_light = dir_light.dir,
        dir_light_emission = dir_light.emission,
    }
    vk.CmdPushConstants(cmd_buf, shaders.rt_pipeline_layout, { .RAYGEN_KHR, .MISS_KHR }, 0, size_of(push), &push)

    vk.CmdTraceRaysKHR(cmd_buf, &raygen_region, &raymiss_region, &rayhit_region, &callable_region, lightmap_size, lightmap_size, 1)
}

smooth_seams :: proc(using bake: ^Bake, cmd_buf: vk.CommandBuffer, pipeline_layout: vk.PipelineLayout, lm: ^vku.Image, lm_backbuffer: ^vku.Image)
{
    vk_check(vk.BeginCommandBuffer(cmd_buf, &{
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        flags = { .ONE_TIME_SUBMIT },
    }))

    blit := vk.ImageBlit {
        srcSubresource = {
            aspectMask = { .COLOR },
            mipLevel = 0,
            baseArrayLayer = 0,
            layerCount = 1,
        },
        srcOffsets = { { 0, 0, 0 }, { i32(lm_backbuffer.width), i32(lm_backbuffer.height), 1 } },
        dstSubresource = {
            aspectMask = { .COLOR },
            mipLevel = 0,
            baseArrayLayer = 0,
            layerCount = 1,
        },
        dstOffsets = { { 0, 0, 0 }, { i32(lm.width), i32(lm.height), 1 } },
    }
    vk.CmdBlitImage(cmd_buf, lm_backbuffer.img, lm_backbuffer.layout, lm.img, lm.layout, 1, &blit, .NEAREST)

    src_desc_set_ai := vk.DescriptorSetAllocateInfo {
        sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool = desc_pool,
        descriptorSetCount = 2,
        pSetLayouts = raw_data([]vk.DescriptorSetLayout {
            shaders.seams_src_desc_set_layout,
            shaders.seams_src_desc_set_layout,
        })
    }
    src_desc_sets: [2]vk.DescriptorSet
    vk_check(vk.AllocateDescriptorSets(device, &src_desc_set_ai, &src_desc_sets[0]))
    defer vk_check(vk.FreeDescriptorSets(device, desc_pool, 1, &src_desc_sets[0]))
    defer vk_check(vk.FreeDescriptorSets(device, desc_pool, 1, &src_desc_sets[1]))

    linear_sampler_ci := vk.SamplerCreateInfo {
        sType = .SAMPLER_CREATE_INFO,
        magFilter = .NEAREST,
        minFilter = .NEAREST,
        mipmapMode = .NEAREST,
        addressModeU = .REPEAT,
        addressModeV = .REPEAT,
        addressModeW = .REPEAT,
    }
    linear_sampler: vk.Sampler
    vk_check(vk.CreateSampler(vk_ctx.device, &linear_sampler_ci, nil, &linear_sampler))
    defer vk.DestroySampler(device, linear_sampler, nil)

    writes := []vk.WriteDescriptorSet {
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = src_desc_sets[0],
            dstBinding = 0,
            descriptorCount = 1,
            descriptorType = .COMBINED_IMAGE_SAMPLER,
            pImageInfo = raw_data([]vk.DescriptorImageInfo {
                {
                    imageView = lm_backbuffer.view,
                    imageLayout = .READ_ONLY_OPTIMAL,
                    sampler = linear_sampler,
                }
            })
        },
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = src_desc_sets[1],
            dstBinding = 0,
            descriptorCount = 1,
            descriptorType = .COMBINED_IMAGE_SAMPLER,
            pImageInfo = raw_data([]vk.DescriptorImageInfo {
                {
                    imageView = lm.view,
                    imageLayout = .READ_ONLY_OPTIMAL,
                    sampler = linear_sampler,
                }
            })
        },
    }
    vk.UpdateDescriptorSets(device, u32(len(writes)), raw_data(writes), 0, nil)

    // src = images[0], dst = images[1]
    images := [2]^vku.Image { lm_backbuffer, lm }

    vku.image_barrier_safe_slow(images[0], cmd_buf, .READ_ONLY_OPTIMAL)
    vku.image_barrier_safe_slow(images[1], cmd_buf, .COLOR_ATTACHMENT_OPTIMAL)

    NUM_SEAMS_ACCUMULATION :: 10000
    for iter in 0..<NUM_SEAMS_ACCUMULATION
    {
        color_attachment := vk.RenderingAttachmentInfo {
            sType = .RENDERING_ATTACHMENT_INFO,
            imageView = images[1].view,
            imageLayout = images[1].layout,
            loadOp = .LOAD,
            storeOp = .STORE,
        }
        rendering_info := vk.RenderingInfo {
            sType = .RENDERING_INFO,
            renderArea = {
                offset = { 0, 0 },
                extent = { images[1].width, images[1].height }
            },
            layerCount = 1,
            colorAttachmentCount = 1,
            pColorAttachments = &color_attachment,
            pDepthAttachment = nil,
        }

        {
            vk.CmdBeginRendering(cmd_buf, &rendering_info)
            defer vk.CmdEndRendering(cmd_buf)

            shader_stages := []vk.ShaderStageFlags { { .VERTEX }, { .GEOMETRY }, { .FRAGMENT } }
            to_bind := []vk.ShaderEXT { shaders.seams_vert, vk.ShaderEXT(0), shaders.seams_frag }
            assert(len(shader_stages) == len(to_bind))
            vk.CmdBindShadersEXT(cmd_buf, u32(len(shader_stages)), raw_data(shader_stages), raw_data(to_bind))

            src_desc_set := src_desc_sets[iter % 2]
            draw_seams(bake, cmd_buf, pipeline_layout, src_desc_set, linear_sampler, iter, images[0]^, images[1]^)
        }

        images[0], images[1] = images[1], images[0]
        vku.image_barrier_safe_slow(images[0], cmd_buf, .READ_ONLY_OPTIMAL)
        vku.image_barrier_safe_slow(images[1], cmd_buf, .COLOR_ATTACHMENT_OPTIMAL)
    }

    vku.image_barrier_safe_slow(lm, cmd_buf, .GENERAL)
    vku.image_barrier_safe_slow(lm_backbuffer, cmd_buf, .GENERAL)

    vk_check(vk.EndCommandBuffer(cmd_buf))

    wait_stage_flags := vk.PipelineStageFlags { .ALL_COMMANDS }
    cmd_bufs := []vk.CommandBuffer { cmd_buf }
    submit_info := vk.SubmitInfo {
        sType = .SUBMIT_INFO,
        pWaitDstStageMask = &wait_stage_flags,
        commandBufferCount = u32(len(cmd_bufs)),
        pCommandBuffers = raw_data(cmd_bufs),
    }
    vk_check(vk.QueueSubmit(vk_ctx.queue, 1, &submit_info, {}))
    vk_check(vk.QueueWaitIdle(vk_ctx.queue))
}

draw_seams :: proc(using bake: ^Bake, cmd_buf: vk.CommandBuffer, pipeline_layout: vk.PipelineLayout, src_desc_set: vk.DescriptorSet, sampler: vk.Sampler, iter: int, src, dst: vku.Image)
{
    sample_mask := vk.SampleMask(1)
    vk.CmdSetSampleMaskEXT(cmd_buf, { ._1 }, &sample_mask)
    vk.CmdSetAlphaToCoverageEnableEXT(cmd_buf, false)

    vk.CmdSetPolygonModeEXT(cmd_buf, .FILL)
    vk.CmdSetCullMode(cmd_buf, {})
    vk.CmdSetFrontFace(cmd_buf, .COUNTER_CLOCKWISE)

    vk.CmdSetDepthCompareOp(cmd_buf, .LESS)
    vk.CmdSetDepthTestEnable(cmd_buf, false)
    vk.CmdSetDepthWriteEnable(cmd_buf, false)
    vk.CmdSetDepthBiasEnable(cmd_buf, false)
    vk.CmdSetDepthClipEnableEXT(cmd_buf, true)

    vk.CmdSetStencilTestEnable(cmd_buf, false)
    b32_true := b32(true)
    vk.CmdSetColorBlendEnableEXT(cmd_buf, 0, 1, &b32_true)
    blend_equation := vk.ColorBlendEquationEXT {
        srcColorBlendFactor = .SRC_ALPHA,
        dstColorBlendFactor = .ONE_MINUS_SRC_ALPHA,
        colorBlendOp = .ADD,
        srcAlphaBlendFactor = .ONE,
        dstAlphaBlendFactor = .ONE_MINUS_SRC_ALPHA,
        alphaBlendOp = .ADD,
    }
    vk.CmdSetColorBlendEquationEXT(cmd_buf, 0, 1, &blend_equation)

    color_mask := vk.ColorComponentFlags { .R, .G, .B, .A }
    vk.CmdSetColorWriteMaskEXT(cmd_buf, 0, 1, &color_mask)

    vk.CmdSetPrimitiveTopology(cmd_buf, .LINE_LIST)
    vk.CmdSetLineWidth(cmd_buf, 1.0)

    vk.CmdSetPrimitiveRestartEnable(cmd_buf, false)

    vert_input_bindings := []vk.VertexInputBindingDescription2EXT {
        {  // Positions
            sType = .VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            binding = 0,
            stride = size_of([3]f32),
            inputRate = .VERTEX,
            divisor = 1,
        },
        {  // Normals
            sType = .VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            binding = 1,
            stride = size_of([3]f32),
            inputRate = .VERTEX,
            divisor = 1,
        },
        {  // Lightmap UVs
            sType = .VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            binding = 2,
            stride = size_of([2]f32),
            inputRate = .VERTEX,
            divisor = 1,
        },
    }
    vert_attributes := []vk.VertexInputAttributeDescription2EXT {
        {
            sType = .VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            location = 0,
            binding = 0,
            format = .R32G32B32_SFLOAT,
            offset = 0
        },
        {
            sType = .VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            location = 1,
            binding = 1,
            format = .R32G32B32_SFLOAT,
            offset = 0
        },
        {
            sType = .VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            location = 2,
            binding = 2,
            format = .R32G32_SFLOAT,
            offset = 0
        },
    }
    vk.CmdSetVertexInputEXT(cmd_buf, u32(len(vert_input_bindings)), raw_data(vert_input_bindings), u32(len(vert_attributes)), raw_data(vert_attributes))

    vk.CmdSetExtraPrimitiveOverestimationSizeEXT(cmd_buf, 0.0)
    //vk.CmdSetConservativeRasterizationModeEXT(cmd_buf, .OVERESTIMATE)
    vk.CmdSetConservativeRasterizationModeEXT(cmd_buf, nil)
    vk.CmdSetRasterizationSamplesEXT(cmd_buf, { ._1 })
    for instance, i in instances
    {
        mesh := get_mesh(ctx, instance.mesh)

        vk.CmdBindDescriptorSets(cmd_buf, .GRAPHICS, shaders.seams_pipeline_layout, 0, 1, &mesh.seams_desc_set, 0, nil)
        tmp := src_desc_set
        vk.CmdBindDescriptorSets(cmd_buf, .GRAPHICS, shaders.seams_pipeline_layout, 1, 1, &tmp, 0, nil)

        viewport := vk.Viewport {
            x = f32(lightmap_size) * instance.lm_offset.x,
            y = f32(lightmap_size) * instance.lm_offset.y,
            width = f32(lightmap_size) * instance.lm_scale,
            height = f32(lightmap_size) * instance.lm_scale,
            minDepth = 0.0,
            maxDepth = 1.0,
        }
        vk.CmdSetViewportWithCount(cmd_buf, 1, &viewport)
        scissor := vk.Rect2D {
            offset = {
                x = i32(f32(lightmap_size) * instance.lm_offset.x),
                y = i32(f32(lightmap_size) * instance.lm_offset.y),
            },
            extent = {
                width = u32(f32(lightmap_size) * instance.lm_scale),
                height = u32(f32(lightmap_size) * instance.lm_scale),
            }
        }
        vk.CmdSetScissorWithCount(cmd_buf, 1, &scissor)
        vk.CmdSetRasterizerDiscardEnable(cmd_buf, false)

        offset := vk.DeviceSize(0)

        Push :: struct {
            render_to_line0: b32,
            target_size: f32,
        }
        push := Push {
            render_to_line0 = iter % 2 == 0,
            target_size = f32(lightmap_size)
        }
        vk.CmdPushConstants(cmd_buf, pipeline_layout, { .VERTEX, .FRAGMENT }, 0, size_of(push), &push)

        vk.CmdDraw(cmd_buf, mesh.num_seams * 2, 1, 0, 0)
    }
}

Shaders :: struct
{
    // GBuffer generation
    lm_desc_set_layout: vk.DescriptorSetLayout,
    pipeline_layout: vk.PipelineLayout,
    uv_space: vk.ShaderEXT,
    gbuffer_world_pos: vk.ShaderEXT,
    gbuffer_world_normals: vk.ShaderEXT,
    gbuffer_tri_idx: vk.ShaderEXT,

    // Lightmap dilation
    dilate_desc_set_layout: vk.DescriptorSetLayout,
    dilate_shader: vk.ShaderEXT,
    dilate_pipeline_layout: vk.PipelineLayout,

    // Sample pushing
    push_samples_pipeline: vk.Pipeline,

    // Seam smoothing
    seams_src_desc_set_layout: vk.DescriptorSetLayout,
    seams_desc_set_layout: vk.DescriptorSetLayout,
    seams_pipeline_layout: vk.PipelineLayout,
    seams_vert: vk.ShaderEXT,
    seams_frag: vk.ShaderEXT,

    // RT
    rt_desc_set_layout: vk.DescriptorSetLayout,
    rt_pipeline_layout: vk.PipelineLayout,
    rt_pipeline: vk.Pipeline,
}

create_shaders :: proc(using ctx: ^Lightmapper_Vulkan_Context) -> Shaders
{
    res: Shaders

    push_constant_ranges := []vk.PushConstantRange {
        {
            stageFlags = { .VERTEX, .FRAGMENT },
            size = 256,
        }
    }

    // Desc set layouts
    {
        lm_desc_set_layout_ci := vk.DescriptorSetLayoutCreateInfo {
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            flags = {},
            bindingCount = 1,
            pBindings = raw_data([]vk.DescriptorSetLayoutBinding {
                {
                    binding = 0,
                    descriptorType = .COMBINED_IMAGE_SAMPLER,
                    descriptorCount = 1,
                    stageFlags = { .FRAGMENT },
                },
            })
        }
        vk_check(vk.CreateDescriptorSetLayout(device, &lm_desc_set_layout_ci, nil, &res.lm_desc_set_layout))

        dilate_desc_set_layout_ci := vk.DescriptorSetLayoutCreateInfo {
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            flags = {},
            bindingCount = 2,
            pBindings = raw_data([]vk.DescriptorSetLayoutBinding {
                {
                    binding = 0,
                    descriptorType = .COMBINED_IMAGE_SAMPLER,
                    descriptorCount = 1,
                    stageFlags = { .COMPUTE },
                },
                {
                    binding = 1,
                    descriptorType = .STORAGE_IMAGE,
                    descriptorCount = 1,
                    stageFlags = { .COMPUTE },
                }
            })
        }
        vk_check(vk.CreateDescriptorSetLayout(device, &dilate_desc_set_layout_ci, nil, &res.dilate_desc_set_layout))

        rt_desc_set_layout_bindings := []vk.DescriptorSetLayoutBinding {
            {
                binding = 0,
                descriptorType = .ACCELERATION_STRUCTURE_KHR,
                descriptorCount = 1,
                stageFlags = { .RAYGEN_KHR },
            },
            {
                binding = 1,
                descriptorType = .STORAGE_IMAGE,
                descriptorCount = 1,
                stageFlags = { .RAYGEN_KHR },
            },
            {
                binding = 2,
                descriptorType = .STORAGE_IMAGE,
                descriptorCount = 1,
                stageFlags = { .RAYGEN_KHR },
            },
            {
                binding = 3,
                descriptorType = .STORAGE_IMAGE,
                descriptorCount = 1,
                stageFlags = { .RAYGEN_KHR },
            },
            {
                binding = 4,
                descriptorType = .STORAGE_IMAGE,
                descriptorCount = 1,
                stageFlags = { .RAYGEN_KHR },
            },
            {
                binding = 5,
                descriptorType = .STORAGE_BUFFER,
                descriptorCount = 1,
                stageFlags = { .CLOSEST_HIT_KHR },
            }
        }

        rt_desc_set_layout_ci := vk.DescriptorSetLayoutCreateInfo {
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            flags = {},
            bindingCount = u32(len(rt_desc_set_layout_bindings)),
            pBindings = raw_data(rt_desc_set_layout_bindings)
        }
        vk_check(vk.CreateDescriptorSetLayout(device, &rt_desc_set_layout_ci, nil, &res.rt_desc_set_layout))

        seams_desc_set_layout_ci := vk.DescriptorSetLayoutCreateInfo {
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            flags = {},
            bindingCount = 1,
            pBindings = raw_data([]vk.DescriptorSetLayoutBinding {
                {
                    binding = 0,
                    descriptorType = .STORAGE_BUFFER,
                    descriptorCount = 1,
                    stageFlags = { .VERTEX },
                },
            })
        }
        seams_src_desc_set_layout_ci := vk.DescriptorSetLayoutCreateInfo {
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            flags = {},
            bindingCount = 1,
            pBindings = raw_data([]vk.DescriptorSetLayoutBinding {
                {
                    binding = 0,
                    descriptorType = .COMBINED_IMAGE_SAMPLER,
                    descriptorCount = 1,
                    stageFlags = { .FRAGMENT },
                }
            })
        }
        vk_check(vk.CreateDescriptorSetLayout(device, &seams_desc_set_layout_ci, nil, &res.seams_desc_set_layout))
        vk_check(vk.CreateDescriptorSetLayout(device, &seams_src_desc_set_layout_ci, nil, &res.seams_src_desc_set_layout))
    }

    // Pipeline layouts
    {
        pipeline_layout_ci := vk.PipelineLayoutCreateInfo {
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            pushConstantRangeCount = u32(len(push_constant_ranges)),
            pPushConstantRanges = raw_data(push_constant_ranges),
            setLayoutCount = 1,
            pSetLayouts = &res.lm_desc_set_layout,
        }
        vk_check(vk.CreatePipelineLayout(device, &pipeline_layout_ci, nil, &res.pipeline_layout))

        dilate_pipeline_layout_ci := vk.PipelineLayoutCreateInfo {
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            pushConstantRangeCount = u32(len(push_constant_ranges)),
            pPushConstantRanges = raw_data(push_constant_ranges),
            setLayoutCount = 1,
            pSetLayouts = &res.dilate_desc_set_layout,
        }
        vk_check(vk.CreatePipelineLayout(device, &dilate_pipeline_layout_ci, nil, &res.dilate_pipeline_layout))

        rt_pipeline_layout_ci := vk.PipelineLayoutCreateInfo {
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            flags = {},
            pushConstantRangeCount = u32(1),
            pPushConstantRanges = raw_data([]vk.PushConstantRange {
                {
                    stageFlags = { .RAYGEN_KHR, .MISS_KHR },
                    size = 256,
                }
            }),
            setLayoutCount = 1,
            pSetLayouts = &res.rt_desc_set_layout
        }
        vk_check(vk.CreatePipelineLayout(device, &rt_pipeline_layout_ci, nil, &res.rt_pipeline_layout))

        seams_pipeline_layout_ci := vk.PipelineLayoutCreateInfo {
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            flags = {},
            pushConstantRangeCount = u32(1),
            pPushConstantRanges = raw_data([]vk.PushConstantRange {
                {
                    stageFlags = { .VERTEX, .FRAGMENT },
                    size = 256,
                }
            }),
            setLayoutCount = 2,
            pSetLayouts = raw_data([]vk.DescriptorSetLayout {
                res.seams_desc_set_layout,
                res.seams_src_desc_set_layout,
            })
        }
        vk_check(vk.CreatePipelineLayout(device, &seams_pipeline_layout_ci, nil, &res.seams_pipeline_layout))
    }

    uv_space_vert_code              := #load("shaders/uv_space.vert.spv", []u32)
    gbuffer_world_pos_frag_code     := #load("shaders/gbuffer_world_pos.frag.spv", []u32)
    gbuffer_world_normals_frag_code := #load("shaders/gbuffer_world_normals.frag.spv", []u32)
    gbuffer_tri_idx_frag_code       := #load("shaders/gbuffer_tri_idx.frag.spv", []u32)
    push_samples_raygen_code        := #load("shaders/push_samples.rgen.spv", []u32)
    push_samples_raymiss_code       := #load("shaders/push_samples.rmiss.spv", []u32)
    push_samples_rayhit_code        := #load("shaders/push_samples.rchit.spv", []u32)
    raygen_code                     := #load("shaders/pathtrace.rgen.spv", []u32)
    raymiss_code                    := #load("shaders/pathtrace.rmiss.spv", []u32)
    rayhit_code                     := #load("shaders/pathtrace.rchit.spv", []u32)
    dilate_code                     := #load("shaders/dilate.comp.spv", []u32)
    seams_vert_code                 := #load("shaders/seams.vert.spv", []u32)
    seams_frag_code                 := #load("shaders/seams.frag.spv", []u32)

    // NOTE: #load(<string-path>, <type>) used to produce unaligned reads and writes (https://github.com/odin-lang/Odin/issues/5771)
    ensure(uintptr(raw_data(uv_space_vert_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(gbuffer_world_pos_frag_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(gbuffer_world_normals_frag_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(gbuffer_tri_idx_frag_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(push_samples_raygen_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(push_samples_raymiss_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(push_samples_rayhit_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(raygen_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(raymiss_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(rayhit_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(dilate_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(seams_vert_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")
    ensure(uintptr(raw_data(seams_frag_code)) % 4 == 0, "#load directive is producing unaligned accesses! Are you on an old Odin version? Update to >= dev-2025-11!")

    // Create shader objects.
    {
        shader_cis := [?]vk.ShaderCreateInfoEXT {
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(uv_space_vert_code) * size_of(uv_space_vert_code[0]),
                pCode = raw_data(uv_space_vert_code),
                pName = "main",
                stage = { .VERTEX },
                nextStage = { .FRAGMENT },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges)
            },
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(gbuffer_world_pos_frag_code) * size_of(gbuffer_world_pos_frag_code[0]),
                pCode = raw_data(gbuffer_world_pos_frag_code),
                pName = "main",
                stage = { .FRAGMENT },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
            },
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(gbuffer_world_normals_frag_code) * size_of(gbuffer_world_normals_frag_code[0]),
                pCode = raw_data(gbuffer_world_normals_frag_code),
                pName = "main",
                stage = { .FRAGMENT },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
            },
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(gbuffer_tri_idx_frag_code) * size_of(gbuffer_tri_idx_frag_code[0]),
                pCode = raw_data(gbuffer_tri_idx_frag_code),
                pName = "main",
                stage = { .FRAGMENT },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
            },
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(dilate_code) * size_of(dilate_code[0]),
                pCode = raw_data(dilate_code),
                pName = "main",
                stage = { .COMPUTE },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
                setLayoutCount = 1,
                pSetLayouts = &res.dilate_desc_set_layout,
            },
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(seams_vert_code) * size_of(seams_vert_code[0]),
                pCode = raw_data(seams_vert_code),
                pName = "main",
                stage = { .VERTEX },
                nextStage = { .FRAGMENT },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
                setLayoutCount = 2,
                pSetLayouts = raw_data([]vk.DescriptorSetLayout {
                    res.seams_desc_set_layout,
                    res.seams_src_desc_set_layout,
                })
            },
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(seams_frag_code) * size_of(seams_frag_code[0]),
                pCode = raw_data(seams_frag_code),
                pName = "main",
                stage = { .FRAGMENT },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
                setLayoutCount = 2,
                pSetLayouts = raw_data([]vk.DescriptorSetLayout {
                    res.seams_desc_set_layout,
                    res.seams_src_desc_set_layout,
                })
            },
        }
        shaders: [len(shader_cis)]vk.ShaderEXT
        vk_check(vk.CreateShadersEXT(device, len(shaders), raw_data(&shader_cis), nil, raw_data(&shaders)))
        res.uv_space = shaders[0]
        res.gbuffer_world_pos = shaders[1]
        res.gbuffer_world_normals = shaders[2]
        res.gbuffer_tri_idx = shaders[3]
        res.dilate_shader = shaders[4]
        res.seams_vert = shaders[5]
        res.seams_frag = shaders[6]
    }

    // RT
    // NOTE: The constant is wrong, this is now fixed but not in the latest release.
    #assert(vk.SHADER_UNUSED_KHR == ~u32(0), "Some vk constants are wrong! Are you on an old Odin version? Update to >= dev-2025-11!")

    push_samples_raygen_shader: vk.ShaderModule
    push_samples_raymiss_shader: vk.ShaderModule
    push_samples_rayhit_shader: vk.ShaderModule

    raygen_shader: vk.ShaderModule
    raymiss_shader: vk.ShaderModule
    rayhit_shader: vk.ShaderModule

    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(push_samples_raygen_code) * size_of(push_samples_raygen_code[0]),
            pCode = auto_cast raw_data(push_samples_raygen_code),
        }
        vk_check(vk.CreateShaderModule(device, &shader_module_ci, nil, &push_samples_raygen_shader))
    }
    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(push_samples_raymiss_code) * size_of(push_samples_raymiss_code[0]),
            pCode = auto_cast raw_data(push_samples_raymiss_code),
        }
        vk_check(vk.CreateShaderModule(device, &shader_module_ci, nil, &push_samples_raymiss_shader))
    }
    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(push_samples_rayhit_code) * size_of(push_samples_rayhit_code[0]),
            pCode = auto_cast raw_data(push_samples_rayhit_code),
        }
        vk_check(vk.CreateShaderModule(device, &shader_module_ci, nil, &push_samples_rayhit_shader))
    }

    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(raygen_code) * size_of(raygen_code[0]),
            pCode = auto_cast raw_data(raygen_code),
        }
        vk_check(vk.CreateShaderModule(device, &shader_module_ci, nil, &raygen_shader))
    }
    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(raymiss_code) * size_of(raymiss_code[0]),
            pCode = auto_cast raw_data(raymiss_code),
        }
        vk_check(vk.CreateShaderModule(device, &shader_module_ci, nil, &raymiss_shader))
    }
    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(rayhit_code) * size_of(rayhit_code[0]),
            pCode = auto_cast raw_data(rayhit_code),
        }
        vk_check(vk.CreateShaderModule(device, &shader_module_ci, nil, &rayhit_shader))
    }

    push_samples_pipeline_ci := vk.RayTracingPipelineCreateInfoKHR {
        sType = .RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        flags = {},
        stageCount = 3,
        pStages = raw_data([]vk.PipelineShaderStageCreateInfo {
            {
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                flags = {},
                stage = { .RAYGEN_KHR },
                module = push_samples_raygen_shader,
                pName = "main"
            },
            {
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                flags = {},
                stage = { .MISS_KHR },
                module = push_samples_raymiss_shader,
                pName = "main"
            },
            {
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                flags = {},
                stage = { .CLOSEST_HIT_KHR },
                module = push_samples_rayhit_shader,
                pName = "main"
            },
        }),
        groupCount = 3,
        pGroups = raw_data([]vk.RayTracingShaderGroupCreateInfoKHR {
            {  // Raygen group
                sType = .RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                type = .GENERAL,
                generalShader = 0,
                closestHitShader = vk.SHADER_UNUSED_KHR,
                anyHitShader = vk.SHADER_UNUSED_KHR,
                intersectionShader = vk.SHADER_UNUSED_KHR,
            },
            {  // Miss group
                sType = .RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                type = .GENERAL,
                generalShader = 1,
                closestHitShader = vk.SHADER_UNUSED_KHR,
                anyHitShader = vk.SHADER_UNUSED_KHR,
                intersectionShader = vk.SHADER_UNUSED_KHR,
            },
            {  // Triangles hit group
                sType = .RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                type = .TRIANGLES_HIT_GROUP,
                generalShader = vk.SHADER_UNUSED_KHR,
                closestHitShader = 2,
                anyHitShader = vk.SHADER_UNUSED_KHR,
                intersectionShader = vk.SHADER_UNUSED_KHR,
            }
        }),
        maxPipelineRayRecursionDepth = 1,
        pLibraryInfo = nil,
        pLibraryInterface = nil,
        pDynamicState = nil,
        layout = res.rt_pipeline_layout,
        basePipelineHandle = cast(vk.Pipeline) 0,
        basePipelineIndex = 0,
    }
    vk_check(vk.CreateRayTracingPipelinesKHR(device, {}, {}, 1, &push_samples_pipeline_ci, nil, &res.push_samples_pipeline))

    rt_pipeline_ci := vk.RayTracingPipelineCreateInfoKHR {
        sType = .RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        flags = {},
        stageCount = 3,
        pStages = raw_data([]vk.PipelineShaderStageCreateInfo {
            {
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                flags = {},
                stage = { .RAYGEN_KHR },
                module = raygen_shader,
                pName = "main"
            },
            {
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                flags = {},
                stage = { .MISS_KHR },
                module = raymiss_shader,
                pName = "main"
            },
            {
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                flags = {},
                stage = { .CLOSEST_HIT_KHR },
                module = rayhit_shader,
                pName = "main"
            },
        }),
        groupCount = 3,
        pGroups = raw_data([]vk.RayTracingShaderGroupCreateInfoKHR {
            {  // Raygen group
                sType = .RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                type = .GENERAL,
                generalShader = 0,
                closestHitShader = vk.SHADER_UNUSED_KHR,
                anyHitShader = vk.SHADER_UNUSED_KHR,
                intersectionShader = vk.SHADER_UNUSED_KHR,
            },
            {  // Miss group
                sType = .RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                type = .GENERAL,
                generalShader = 1,
                closestHitShader = vk.SHADER_UNUSED_KHR,
                anyHitShader = vk.SHADER_UNUSED_KHR,
                intersectionShader = vk.SHADER_UNUSED_KHR,
            },
            {  // Triangles hit group
                sType = .RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                type = .TRIANGLES_HIT_GROUP,
                generalShader = vk.SHADER_UNUSED_KHR,
                closestHitShader = 2,
                anyHitShader = vk.SHADER_UNUSED_KHR,
                intersectionShader = vk.SHADER_UNUSED_KHR,
            }
        }),
        maxPipelineRayRecursionDepth = 1,
        pLibraryInfo = nil,
        pLibraryInterface = nil,
        pDynamicState = nil,
        layout = res.rt_pipeline_layout,
        basePipelineHandle = cast(vk.Pipeline) 0,
        basePipelineIndex = 0,
    }
    vk_check(vk.CreateRayTracingPipelinesKHR(device, {}, {}, 1, &rt_pipeline_ci, nil, &res.rt_pipeline))

    return res
}

destroy_shaders :: proc(using ctx: ^Lightmapper_Vulkan_Context, shaders: ^Shaders)
{
    vk.DestroyPipelineLayout(device, shaders.pipeline_layout, nil)
    vk.DestroyShaderEXT(device, shaders.uv_space, nil)
    vk.DestroyShaderEXT(device, shaders.gbuffer_world_pos, nil)
    vk.DestroyShaderEXT(device, shaders.gbuffer_world_normals, nil)

    vk.DestroyDescriptorSetLayout(device, shaders.rt_desc_set_layout, nil)
    vk.DestroyPipelineLayout(device, shaders.rt_pipeline_layout, nil)
    vk.DestroyPipeline(device, shaders.rt_pipeline, nil)
}

GBuffers :: struct
{
    world_pos: Image,
    world_normals: Image,
    world_geom_normals: Image,
    tri_idx: Image,
}

create_gbuffers :: proc(using bake: ^Bake, cmd_buf: vk.CommandBuffer) -> GBuffers
{
    world_pos := vku.create_image(device, phys_device, cmd_buf, {
        sType = .IMAGE_CREATE_INFO,
        flags = {},
        imageType = .D2,
        format = .R32G32B32A32_SFLOAT,
        extent = {
            width = lightmap_size,
            height = lightmap_size,
            depth = 1,
        },
        mipLevels = 1,
        arrayLayers = 1,
        samples = { ._1 },
        usage = { .COLOR_ATTACHMENT, .STORAGE },
        sharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &queue_family_idx,
        initialLayout = .UNDEFINED,
    })

    world_normals := vku.create_image(device, phys_device, cmd_buf, {
        sType = .IMAGE_CREATE_INFO,
        flags = {},
        imageType = .D2,
        format = .R8G8B8A8_UNORM,
        extent = {
            width = lightmap_size,
            height = lightmap_size,
            depth = 1,
        },
        mipLevels = 1,
        arrayLayers = 1,
        samples = { ._1 },
        usage = { .COLOR_ATTACHMENT, .STORAGE },
        sharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &queue_family_idx,
        initialLayout = .UNDEFINED,
    })

    world_geom_normals := vku.create_image(device, phys_device, cmd_buf, {
        sType = .IMAGE_CREATE_INFO,
        flags = {},
        imageType = .D2,
        format = .R8G8B8A8_UNORM,
        extent = {
            width = lightmap_size,
            height = lightmap_size,
            depth = 1,
        },
        mipLevels = 1,
        arrayLayers = 1,
        samples = { ._1 },
        usage = { .COLOR_ATTACHMENT, .STORAGE },
        sharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &queue_family_idx,
        initialLayout = .UNDEFINED,
    })

    tri_idx := vku.create_image(device, phys_device, cmd_buf, {
        sType = .IMAGE_CREATE_INFO,
        flags = {},
        imageType = .D2,
        format = .R32_UINT,
        extent = {
            width = lightmap_size,
            height = lightmap_size,
            depth = 1,
        },
        mipLevels = 1,
        arrayLayers = 1,
        samples = { ._1 },
        usage = { .COLOR_ATTACHMENT, .STORAGE },
        sharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &queue_family_idx,
        initialLayout = .UNDEFINED,
    })

    return {
        world_pos,
        world_normals,
        world_geom_normals,
        tri_idx,
    }
}

gbuffers_barrier :: proc(gbufs: ^GBuffers, cmd_buf: vk.CommandBuffer)
{
    vku.image_barrier_safe_slow(&gbufs.world_pos, cmd_buf, .GENERAL)
    vku.image_barrier_safe_slow(&gbufs.world_normals, cmd_buf, .GENERAL)
    vku.image_barrier_safe_slow(&gbufs.world_geom_normals, cmd_buf, .GENERAL)
    vku.image_barrier_safe_slow(&gbufs.tri_idx, cmd_buf, .GENERAL)
}

// TODO
gbuffers_destroy :: proc(gbufs: ^GBuffers)
{

}

update_rt_desc_set :: proc(device: vk.Device, to_update: vk.DescriptorSet, tlas: vk.AccelerationStructureKHR, lightmap: Image, gbuffers: GBuffers, geometries: Buffer)
{
    as_info := vk.WriteDescriptorSetAccelerationStructureKHR {
        sType = .WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        accelerationStructureCount = 1,
        pAccelerationStructures = raw_data([]vk.AccelerationStructureKHR { tlas })
    }

    writes := []vk.WriteDescriptorSet {
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = to_update,
            dstBinding = 0,
            descriptorCount = 1,
            descriptorType = .ACCELERATION_STRUCTURE_KHR,
            pNext = &as_info,
        },
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = to_update,
            dstBinding = 1,
            descriptorCount = 1,
            descriptorType = .STORAGE_IMAGE,
            pImageInfo = raw_data([]vk.DescriptorImageInfo {
                {
                    imageView = lightmap.view,
                    imageLayout = lightmap.layout,
                }
            })
        },
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = to_update,
            dstBinding = 2,
            descriptorCount = 1,
            descriptorType = .STORAGE_IMAGE,
            pImageInfo = raw_data([]vk.DescriptorImageInfo {
                {
                    imageView = gbuffers.world_pos.view,
                    imageLayout = gbuffers.world_pos.layout,
                }
            })
        },
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = to_update,
            dstBinding = 3,
            descriptorCount = 1,
            descriptorType = .STORAGE_IMAGE,
            pImageInfo = raw_data([]vk.DescriptorImageInfo {
                {
                    imageView = gbuffers.world_normals.view,
                    imageLayout = gbuffers.world_normals.layout,
                }
            })
        },
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = to_update,
            dstBinding = 4,
            descriptorCount = 1,
            descriptorType = .STORAGE_IMAGE,
            pImageInfo = raw_data([]vk.DescriptorImageInfo {
                {
                    imageView = gbuffers.world_geom_normals.view,
                    imageLayout = gbuffers.world_geom_normals.layout,
                }
            })
        },
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = to_update,
            dstBinding = 5,
            descriptorCount = 1,
            descriptorType = .STORAGE_BUFFER,
            pBufferInfo = raw_data([]vk.DescriptorBufferInfo {
                {
                    buffer = geometries.handle,
                    offset = vk.DeviceSize(0),
                    range = vk.DeviceSize(vk.WHOLE_SIZE),
                }
            })
        }
    }
    vk.UpdateDescriptorSets(device, u32(len(writes)), raw_data(writes), 0, nil)
}

update_dilate_desc_set :: proc(device: vk.Device, to_update: vk.DescriptorSet, dummy_sampler: vk.Sampler, src_image: Image, dst_image: Image)
{
    writes := []vk.WriteDescriptorSet {
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = to_update,
            dstBinding = 0,
            descriptorCount = 1,
            descriptorType = .COMBINED_IMAGE_SAMPLER,
            pImageInfo = raw_data([]vk.DescriptorImageInfo {
                {
                    imageView = src_image.view,
                    imageLayout = src_image.layout,
                    sampler = dummy_sampler,
                }
            })
        },
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = to_update,
            dstBinding = 1,
            descriptorCount = 1,
            descriptorType = .STORAGE_IMAGE,
            pImageInfo = raw_data([]vk.DescriptorImageInfo {
                {
                    imageView = dst_image.view,
                    imageLayout = dst_image.layout,
                }
            })
        },
    }
    vk.UpdateDescriptorSets(device, u32(len(writes)), raw_data(writes), 0, nil)
}

fatal_error :: proc(fmt: string, args: ..any, location := #caller_location)
{
    when ODIN_DEBUG {
        log.fatal(fmt, args, location = location)
        runtime.panic("")
    } else {
        log.panicf(fmt, args, location = location)
    }
}

// OIDN

create_oidn_context :: proc(phys_device: vk.PhysicalDevice) -> oidn.Device
{
    id_props := vk.PhysicalDeviceIDProperties {
        sType = .PHYSICAL_DEVICE_ID_PROPERTIES
    }

    props := vk.PhysicalDeviceProperties2 {
        sType = .PHYSICAL_DEVICE_PROPERTIES_2,
        pNext = &id_props,
    }

    vk.GetPhysicalDeviceProperties2(phys_device, &props)

    device: oidn.Device
    if device == nil && id_props.deviceLUIDValid {
        device = oidn.NewDeviceByLUID(&id_props.deviceLUID[0])
    }
    if device == nil {
        device = oidn.NewDeviceByUUID(&id_props.deviceUUID[0])
    }

    oidn.SetDeviceErrorFunction(device, oidn_error_callback, nil)
    oidn.CommitDevice(device)

    oidn_check(device)

    return device
}

oidn_error_callback :: proc "c"(userPtr: rawptr, code: oidn.Error, message: cstring)
{
    context = runtime.default_context()
    log.error(message)
}

oidn_shared_buffer_from_vk_buffer :: proc(device: oidn.Device, buf: External_Buf) -> oidn.Buffer
{
    when ODIN_OS == .Windows
    {
        return oidn.NewSharedBufferFromWin32Handle(device, { .OPAQUE_WIN32 }, buf.win_handle, nil, c.size_t(buf.buf.size))
    }
    else when ODIN_OS == .Linux
    {
        return oidn.NewSharedBufferFromFD(device, { .OPAQUE_FD }, buf.linux_handle, buf.buf.size)
    }
    else do #panic("Unsupported OS.")
}

oidn_run_lightmap_filter :: proc(device: oidn.Device, filter: oidn.Filter)
{
    oidn.ExecuteFilter(filter)
    oidn.SyncDevice(device)
    oidn_check(device)
}

oidn_check :: proc(device: oidn.Device)
{
    msg: cstring
    if oidn.GetDeviceError(device, &msg) != .NONE
    {
        log.error(msg)
        panic("")
    }
}

External_Buf :: struct
{
    linux_handle: c.int,
    win_handle: vk.HANDLE,
    buf: Buffer,
}

create_vk_external_buffer_for_oidn :: proc(using vk_ctx: ^Lightmapper_Vulkan_Context, size: u32) -> External_Buf
{
    res: External_Buf
    res.buf.size = vk.DeviceSize(size)

    next: rawptr
    when ODIN_OS == .Windows
    {
        next = &vk.ExternalMemoryBufferCreateInfo {
            sType = .EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            handleTypes = { .OPAQUE_WIN32 },
        }
    }
    else when ODIN_OS == .Linux
    {
        next = &vk.ExternalMemoryBufferCreateInfo {
            sType = .EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            handleTypes = { .OPAQUE_FD },
        }
    }
    else do #panic("Unsupported OS.")

    buf_ci := vk.BufferCreateInfo {
        sType = .BUFFER_CREATE_INFO,
        pNext = next,
        size = vk.DeviceSize(size),
        usage = { .TRANSFER_DST, .TRANSFER_SRC, .STORAGE_BUFFER },
        sharingMode = .EXCLUSIVE,
    }
    vk.CreateBuffer(device, &buf_ci, nil, &res.buf.handle)

    mem_reqs: vk.MemoryRequirements
    vk.GetBufferMemoryRequirements(device, res.buf.handle, &mem_reqs)

    next = nil
    when ODIN_OS == .Windows
    {
        next = &vk.ExportMemoryAllocateInfo {
            sType = .EXPORT_MEMORY_ALLOCATE_INFO,
            pNext = next,
            handleTypes = { .OPAQUE_WIN32 },
        }
    }
    else when ODIN_OS == .Linux
    {
        next = &vk.ExportMemoryAllocateInfo {
            sType = .EXPORT_MEMORY_ALLOCATE_INFO,
            pNext = next,
            handleTypes = { .OPAQUE_FD },
        }
    }
    else do #panic("Unsupported OS.")

    allocInfo := vk.MemoryAllocateInfo {
        sType = .MEMORY_ALLOCATE_INFO,
        pNext = next,
        allocationSize = mem_reqs.size,
        memoryTypeIndex = vku.find_mem_type(phys_device, mem_reqs.memoryTypeBits, { .DEVICE_LOCAL }),
    }
    vk.AllocateMemory(device, &allocInfo, nil, &res.buf.mem);

    vk.BindBufferMemory(device, res.buf.handle, res.buf.mem, 0)

    when ODIN_OS == .Windows
    {
        get_fd_info := vk.MemoryGetWin32HandleInfoKHR {
            sType = .MEMORY_GET_WIN32_HANDLE_INFO_KHR,
            memory = res.buf.mem,
            handleType = { .OPAQUE_WIN32 },
        }
        vk_check(vk.GetMemoryWin32HandleKHR(device, &get_fd_info, &res.win_handle))
    }
    else when ODIN_OS == .Linux
    {
        get_fd_info := vk.MemoryGetFdInfoKHR {
            sType = .GET_FD_INFO_KHR,
            memory = res.buf.mem,
            handleType = { .OPAQUE_FD },
        }
        vk_check(vk.GetMemoryFdKHR(device, &get_fd_info, &res.linux_handle))
    }
    else do #panic("Unsupported OS.")

    return res
}

vk_check :: proc(result: vk.Result, location := #caller_location)
{
    if result != .SUCCESS {
        fatal_error("Vulkan failure: %", result, location = location)
    }
}

align_up :: proc(x, align: u32) -> (aligned: u32)
{
    assert(0 == (align & (align - 1)), "must align to a power of two")
    return (x + (align - 1)) &~ (align - 1)
}

Tlas :: struct
{
    as: vku.Accel_Structure,
    instances_buf: Buffer,
}

create_tlas :: proc(device: vk.Device, phys_device: vk.PhysicalDevice, queue: vk.Queue, cmd_pool: vk.CommandPool, instances: []Instance, meshes: []Mesh) -> Tlas
{
    as: vku.Accel_Structure

    vk_instances := make([]vk.AccelerationStructureInstanceKHR, len(instances), allocator = context.temp_allocator)
    for &vk_instance, i in vk_instances
    {
        instance := instances[i]
        transform := instance.transform

        vk_transform := vk.TransformMatrixKHR {
            mat = {
                { transform[0, 0], transform[0, 1], transform[0, 2], transform[0, 3] },
                { transform[1, 0], transform[1, 1], transform[1, 2], transform[1, 3] },
                { transform[2, 0], transform[2, 1], transform[2, 2], transform[2, 3] },
            }
        }

        vk_instance = {
            transform = vk_transform,
            instanceCustomIndex = u32(instance.mesh.idx),
            mask = 0xFF,
            instanceShaderBindingTableRecordOffset = 0,
            // NOTE: Unintuitive bindings! This cast is necessary!
            flags = auto_cast(vk.GeometryInstanceFlagsKHR { .TRIANGLE_FACING_CULL_DISABLE }),
            accelerationStructureReference = u64(meshes[instance.mesh.idx].blas.addr)
        }
    }

    instances_buf := vku.upload_buffer(device, phys_device, queue, cmd_pool, vk_instances, { .ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS, .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .TRANSFER_DST }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })

    instances_data := vk.AccelerationStructureGeometryInstancesDataKHR {
        sType = .ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        arrayOfPointers = false,
        data = {
            deviceAddress = vku.get_buffer_device_address(device, instances_buf)
        }
    }

    geometry := vk.AccelerationStructureGeometryKHR {
        sType = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        geometryType = .INSTANCES,
        geometry = {
            instances = instances_data
        }
    }

    build_info := vk.AccelerationStructureBuildGeometryInfoKHR {
        sType = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        flags = { .PREFER_FAST_TRACE },
        geometryCount = 1,
        pGeometries = &geometry,
        type = .TOP_LEVEL,
    }

    primitive_count := u32(len(vk_instances))
    size_info := vk.AccelerationStructureBuildSizesInfoKHR {
        sType = .ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
    }
    vk.GetAccelerationStructureBuildSizesKHR(device, .DEVICE, &build_info, &primitive_count, &size_info)

    as.buf = vku.create_buffer(device, phys_device, auto_cast size_info.accelerationStructureSize, { .ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })

    // Create the scratch buffer for tlas building
    scratch_buf := vku.create_buffer(device, phys_device, auto_cast size_info.buildScratchSize, { .ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS, .STORAGE_BUFFER }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })

    // Build acceleration structure
    blas_ci := vk.AccelerationStructureCreateInfoKHR {
        sType = .ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        buffer = as.buf.handle,
        size = size_info.accelerationStructureSize,
        type = .TOP_LEVEL,
    }
    vk_check(vk.CreateAccelerationStructureKHR(device, &blas_ci, nil, &as.handle))

    {
        cmd_buf := vku.begin_tmp_cmd_buf(device, cmd_pool)
        defer vku.end_tmp_cmd_buf(device, cmd_pool, queue, cmd_buf)

        range_info := vk.AccelerationStructureBuildRangeInfoKHR {
            primitiveCount = u32(len(instances)),
            primitiveOffset = 0,
            firstVertex = 0,
            transformOffset = 0,
        }
        range_info_ptrs := []^vk.AccelerationStructureBuildRangeInfoKHR {
            &range_info
        }

        build_info.dstAccelerationStructure = as.handle
        build_info.scratchData.deviceAddress = vku.get_buffer_device_address(device, scratch_buf)
        vk.CmdBuildAccelerationStructuresKHR(cmd_buf, 1, &build_info, auto_cast raw_data(range_info_ptrs))
    }

    // Get device address
    addr_info := vk.AccelerationStructureDeviceAddressInfoKHR {
        sType = .ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        accelerationStructure = as.handle,
    }
    as.addr = vk.GetAccelerationStructureDeviceAddressKHR(device, &addr_info)

    return {
        as = as,
        instances_buf = instances_buf
    }
}

compute_geom_normals :: proc(indices: []u32, positions: [][3]f32) -> [dynamic][3]f32
{
    res := make_dynamic_array_len_cap([dynamic][3]f32, 0, len(indices))
    for i := 0; i < len(indices); i += 3
    {
        idx0 := indices[i + 0]
        idx1 := indices[i + 1]
        idx2 := indices[i + 2]
        pos0 := positions[idx0]
        pos1 := positions[idx1]
        pos2 := positions[idx2]

        geom_normal := linalg.cross(pos1 - pos0, pos2 - pos0)

        append(&res, geom_normal)
        append(&res, geom_normal)
        append(&res, geom_normal)
    }

    return res
}

Seam :: struct
{
    edge_a: [2]u32,
    edge_b: [2]u32,
}

find_seams :: proc(indices: []u32, positions: [][3]f32, normals: [][3]f32, uvs: [][2]f32) -> [dynamic]Seam
{
    // Collect edges
    Edge :: [2]u32
    edges: [dynamic]Edge
    defer delete(edges)

    for i := 0; i < len(indices); i += 3
    {
        append(&edges, Edge { indices[i + 0], indices[i + 1] })
        append(&edges, Edge { indices[i + 1], indices[i + 2] })
        append(&edges, Edge { indices[i + 2], indices[i + 0] })
    }

    // Sort edges (for faster comparisons)
    for &edge in edges
    {
        if edge[0] > edge[1] {
            edge[0], edge[1] = edge[1], edge[0]
        }
    }

    // Build acceleration struture for nearest neighbor searches
    {
        Collection :: struct
        {
            edges: []Edge,
            positions: [][3]f32,
        }

        collection := Collection { edges[:], positions }

        interface := sort.Interface {
            collection = rawptr(&collection),
            len = proc(it: sort.Interface) -> int {
                c := (^Collection)(it.collection)
                return len(c.edges)
            },
            less = proc(it: sort.Interface, i, j: int) -> bool {
                c := (^Collection)(it.collection)
                pos0_x := min(c.positions[c.edges[i][0]].x, c.positions[c.edges[i][1]].x)
                pos1_x := min(c.positions[c.edges[j][0]].x, c.positions[c.edges[j][1]].x)
                return pos0_x < pos1_x
            },
            swap = proc(it: sort.Interface, i, j: int) {
                c := (^Collection)(it.collection)
                c.edges[i], c.edges[j] = c.edges[j], c.edges[i]
            },
        }

        sort.sort(interface)
    }

    res: [dynamic]Seam
    EPSILON :: 0.00001
    for i in 0..<len(edges)
    {
        pos0_x := min(positions[edges[i][0]].x, positions[edges[i][1]].x)

        for j := i-1; j >= 0; j -= 1
        {
            pos1_x := min(positions[edges[j][0]].x, positions[edges[j][1]].x)
            if abs(pos1_x - pos0_x) > EPSILON do break

            // Check first vertex
            same_pos := linalg.length(positions[edges[i][0]] - positions[edges[j][0]]) < EPSILON
            if !same_pos do continue
            same_normal := linalg.length(normals[edges[i][0]] - normals[edges[j][0]]) < EPSILON
            if !same_normal do continue
            same_uv := linalg.length(uvs[edges[i][0]] - uvs[edges[j][0]]) < EPSILON
            if same_uv do continue

            // Check second vertex
            same_pos = linalg.length(positions[edges[i][1]] - positions[edges[j][1]]) < EPSILON
            if !same_pos do continue
            same_normal = linalg.length(normals[edges[i][1]] - normals[edges[j][1]]) < EPSILON
            if !same_normal do continue
            same_uv = linalg.length(uvs[edges[i][1]] - uvs[edges[j][1]]) < EPSILON
            if same_uv do continue

            // Edges could be aligned and share a segment even though uv verts are not the same
            if edges_share_segment(uvs, edges[i], edges[j], EPSILON) do continue

            // Found a seam
            append(&res, Seam { edges[i], edges[j] })
        }

        for j in i+1..<len(edges)
        {
            pos1_x := min(positions[edges[j][0]].x, positions[edges[j][1]].x)
            if abs(pos1_x - pos0_x) > EPSILON do break

            // Check first vertex
            same_pos := linalg.length(positions[edges[i][0]] - positions[edges[j][0]]) < EPSILON
            if !same_pos do continue
            same_normal := linalg.length(normals[edges[i][0]] - normals[edges[j][0]]) < EPSILON
            if !same_normal do continue
            same_uv := linalg.length(uvs[edges[i][0]] - uvs[edges[j][0]]) < EPSILON
            if same_uv do continue

            // Check second vertex
            same_pos = linalg.length(positions[edges[i][1]] - positions[edges[j][1]]) < EPSILON
            if !same_pos do continue
            same_normal = linalg.length(normals[edges[i][1]] - normals[edges[j][1]]) < EPSILON
            if !same_normal do continue
            same_uv = linalg.length(uvs[edges[i][1]] - uvs[edges[j][1]]) < EPSILON
            if same_uv do continue

            // Edges could be aligned and share a segment even though uv verts are not the same
            if edges_share_segment(uvs, edges[i], edges[j], EPSILON) do continue

            // Found a seam
            append(&res, Seam { edges[i], edges[j] })
        }
    }

    return res

    edges_share_segment :: proc(uvs: [][2]f32, edge0: Edge, edge1: Edge, eps: f32) -> bool
    {
        a := uvs[edge0[0]]
        b := uvs[edge0[1]]
        c := uvs[edge1[0]]
        d := uvs[edge1[1]]

        ab_dir := linalg.normalize(b - a)
        ac_dir := linalg.normalize(c - a)
        ad_dir := linalg.normalize(d - a)

        // Check if aligned
        if abs(linalg.dot(ab_dir, ac_dir) - 1) > eps ||
           abs(linalg.dot(ab_dir, ad_dir) - 1) > eps {
            return false
        }

        // Project verts to ab_dir
        a_p := linalg.dot(ab_dir, a)
        b_p := linalg.dot(ab_dir, b)
        c_p := linalg.dot(ab_dir, c)
        d_p := linalg.dot(ab_dir, d)

        // Sort verts
        if a_p > b_p do a_p, b_p = b_p, a_p
        if c_p > d_p do c_p, d_p = d_p, c_p

        // Check interval overlap
        if c_p > a_p && d_p < b_p do return true
        if a_p > c_p && b_p < d_p do return true
        if c_p > a_p && c_p < b_p do return true
        if d_p > a_p && d_p < b_p do return true

        return false
    }
}
