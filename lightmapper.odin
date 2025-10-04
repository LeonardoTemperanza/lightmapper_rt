
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
import "core:math"
import "core:math/rand"
import la "core:math/linalg"
import intr "base:intrinsics"
import "core:slice"
import "core:mem"
import "core:log"
import "base:runtime"

import sdl "vendor:sdl3"
import vk "vendor:vulkan"

Context :: struct
{
    using vk_ctx: Vulkan_Context,
    shaders: Shaders,
}

Vulkan_Context :: struct
{
    phys_device: vk.PhysicalDevice,
    device: vk.Device,
    queue: vk.Queue,
    queue_family_idx: u32,
    rt_handle_alignment: u32,
    rt_handle_size: u32,
    rt_base_align: u32,
}

// Initialization

init_from_scratch :: proc() -> Context
{
    return {}
}

init_from_vulkan_instance :: proc(vk_ctx: Vulkan_Context) -> Context
{
    res: Context
    res.vk_ctx = vk_ctx

    return res
}

// Scene description

Scene :: struct
{

}

// Baking

@(private="file")
Bake :: struct
{
    using ctx: ^Context,
}

start_bake :: proc(using ctx: ^Context, lightmap_size: u32, scene: Scene) -> Bake
{
    return {}
}

// If it's 1 it's done.
progress :: proc(using bake: ^Bake) -> f32
{
    return 0.0
}

// Cleans up resources if called after progress(bake) == 1.0 and
// forcibly aborts the baking process if called before progress(bake) == 1.0
end_bake :: proc(using bake: ^Bake)
{

}

wait_end_of_bake :: proc()
{

}

get_current_lightmap_view_vk :: proc() -> vk.ImageView
{
    return vk.ImageView(0)
}

get_current_lightmap_view_cpu_buf :: proc() -> []byte
{
    return {}
}

// Internals

Shaders :: struct
{
    // GBuffer generation
    pipeline_layout: vk.PipelineLayout,
    uv_space: vk.ShaderEXT,
    gbuffer_world_pos: vk.ShaderEXT,
    gbuffer_world_normals: vk.ShaderEXT,

    // RT
    rt_desc_set_layout: vk.DescriptorSetLayout,
    rt_pipeline_layout: vk.PipelineLayout,
    rt_pipeline: vk.Pipeline,
    sbt_buf: Buffer,
}

Buffer :: struct
{
    handle: vk.Buffer,
    mem: vk.DeviceMemory,
}

uv_space_vert_code              := #load("shaders/uv_space.vert.spv")
gbuffer_world_pos_frag_code     := #load("shaders/gbuffer_world_pos.frag.spv")
gbuffer_world_normals_frag_code := #load("shaders/gbuffer_world_normals.frag.spv")
raygen_code                     := #load("shaders/raygen.rgen.spv")
raymiss_code                    := #load("shaders/raymiss.rmiss.spv")
rayhit_code                     := #load("shaders/rayhit.rchit.spv")

create_shaders :: proc(using ctx: ^Vulkan_Context) -> Shaders
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
        rt_desc_set_layout_ci := vk.DescriptorSetLayoutCreateInfo {
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            flags = {},
            bindingCount = 4,
            pBindings = raw_data([]vk.DescriptorSetLayoutBinding {
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
                }
            })
        }
        vk_check(vk.CreateDescriptorSetLayout(device, &rt_desc_set_layout_ci, nil, &res.rt_desc_set_layout))
    }

    // Pipeline layouts
    {
        rt_pipeline_layout_ci := vk.PipelineLayoutCreateInfo {
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            flags = {},
            pushConstantRangeCount = u32(1),
            pPushConstantRanges = raw_data([]vk.PushConstantRange {
                {
                    stageFlags = { .RAYGEN_KHR },
                    size = 256,
                }
            }),
            setLayoutCount = 1,
            pSetLayouts = &res.rt_desc_set_layout
        }
        vk_check(vk.CreatePipelineLayout(device, &rt_pipeline_layout_ci, nil, &res.rt_pipeline_layout))
    }

    // Create shader objects.
    {
        shader_cis := [?]vk.ShaderCreateInfoEXT {
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(uv_space_vert_code),
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
                codeSize = len(gbuffer_world_pos_frag_code),
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
                codeSize = len(gbuffer_world_normals_frag_code),
                pCode = raw_data(gbuffer_world_normals_frag_code),
                pName = "main",
                stage = { .FRAGMENT },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
            },
        }
        shaders: [len(shader_cis)]vk.ShaderEXT
        vk_check(vk.CreateShadersEXT(device, len(shaders), raw_data(&shader_cis), nil, raw_data(&shaders)))
        res.uv_space = shaders[0]
        res.gbuffer_world_pos = shaders[1]
        res.gbuffer_world_normals = shaders[2]
    }

    // RT
    SHADER_UNUSED :: ~u32(0)

    raygen_shader: vk.ShaderModule
    raymiss_shader: vk.ShaderModule
    rayhit_shader: vk.ShaderModule

    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(raygen_code),
            pCode = auto_cast raw_data(raygen_code),
        }
        vk_check(vk.CreateShaderModule(device, &shader_module_ci, nil, &raygen_shader))
    }
    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(raymiss_code),
            pCode = auto_cast raw_data(raymiss_code),
        }
        vk_check(vk.CreateShaderModule(device, &shader_module_ci, nil, &raymiss_shader))
    }
    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(rayhit_code),
            pCode = auto_cast raw_data(rayhit_code),
        }
        vk_check(vk.CreateShaderModule(device, &shader_module_ci, nil, &rayhit_shader))
    }

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
                closestHitShader = SHADER_UNUSED,
                anyHitShader = SHADER_UNUSED,
                intersectionShader = SHADER_UNUSED,
            },
            {  // Miss group
                sType = .RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                type = .GENERAL,
                generalShader = 1,
                closestHitShader = SHADER_UNUSED,
                anyHitShader = SHADER_UNUSED,
                intersectionShader = SHADER_UNUSED,
            },
            {  // Triangles hit group
                sType = .RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                type = .TRIANGLES_HIT_GROUP,
                generalShader = SHADER_UNUSED,
                closestHitShader = 2,
                anyHitShader = SHADER_UNUSED,
                intersectionShader = SHADER_UNUSED,
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

    // Raytracing resources
    {
        // Shader Binding Table
        group_count := u32(3)
        size := group_count * rt_handle_size
        shader_handle_storage := make([]byte, size, allocator = context.temp_allocator)
        vk_check(vk.GetRayTracingShaderGroupHandlesKHR(device, res.rt_pipeline, 0, group_count, int(size), raw_data(shader_handle_storage)))
        res.sbt_buf = create_sbt_buffer(ctx, shader_handle_storage, group_count)
    }

    return res
}

destroy_shaders :: proc(using ctx: ^Vulkan_Context, shaders: ^Shaders)
{
    vk.DestroyPipelineLayout(device, shaders.pipeline_layout, nil)
    vk.DestroyShaderEXT(device, shaders.uv_space, nil)
    vk.DestroyShaderEXT(device, shaders.gbuffer_world_pos, nil)
    vk.DestroyShaderEXT(device, shaders.gbuffer_world_normals, nil)

    vk.DestroyDescriptorSetLayout(device, shaders.rt_desc_set_layout, nil)
    vk.DestroyPipelineLayout(device, shaders.rt_pipeline_layout, nil)
    vk.DestroyPipeline(device, shaders.rt_pipeline, nil)
    destroy_buffer(ctx, &shaders.sbt_buf)
}

create_buffer :: proc(using ctx: ^Vulkan_Context, el_size: int, count: int, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, allocate_flags: vk.MemoryAllocateFlags) -> Buffer
{
    res: Buffer

    buf_ci := vk.BufferCreateInfo {
        sType = .BUFFER_CREATE_INFO,
        size = cast(vk.DeviceSize) (el_size * count),
        usage = usage,
        sharingMode = .EXCLUSIVE,
    }
    vk_check(vk.CreateBuffer(device, &buf_ci, nil, &res.handle))

    mem_requirements: vk.MemoryRequirements
    vk.GetBufferMemoryRequirements(device, res.handle, &mem_requirements)

    alloc_info := vk.MemoryAllocateFlagsInfo {
        sType = .MEMORY_ALLOCATE_FLAGS_INFO,
        flags = allocate_flags,
    }

    next: rawptr
    if allocate_flags != {} {
        next = &alloc_info
    }

    memory_ai := vk.MemoryAllocateInfo {
        sType = .MEMORY_ALLOCATE_INFO,
        pNext = next,
        allocationSize = mem_requirements.size,
        memoryTypeIndex = find_mem_type(ctx, mem_requirements.memoryTypeBits, properties)
    }
    vk_check(vk.AllocateMemory(device, &memory_ai, nil, &res.mem))

    vk.BindBufferMemory(device, res.handle, res.mem, 0)

    return res
}

copy_buffer :: proc(using ctx: ^Vulkan_Context, src: Buffer, dst: Buffer, size: vk.DeviceSize)
{
    // TEMPORARY CMD_BUF!!!
    cmd_pool_ci := vk.CommandPoolCreateInfo {
        sType = .COMMAND_POOL_CREATE_INFO,
        queueFamilyIndex = queue_family_idx,
        flags = { .TRANSIENT }
    }
    cmd_pool: vk.CommandPool
    vk_check(vk.CreateCommandPool(device, &cmd_pool_ci, nil, &cmd_pool))
    defer vk.DestroyCommandPool(device, cmd_pool, nil)

    cmd_buf_ai := vk.CommandBufferAllocateInfo {
        sType = .COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool = cmd_pool,
        level = .PRIMARY,
        commandBufferCount = 1,
    }
    cmd_buf: vk.CommandBuffer
    vk_check(vk.AllocateCommandBuffers(device, &cmd_buf_ai, &cmd_buf))
    defer vk.FreeCommandBuffers(device, cmd_pool, 1, &cmd_buf)

    cmd_buf_bi := vk.CommandBufferBeginInfo {
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        flags = { .ONE_TIME_SUBMIT },
    }
    vk_check(vk.BeginCommandBuffer(cmd_buf, &cmd_buf_bi))

    copy_region := vk.BufferCopy {
        srcOffset = 0,
        dstOffset = 0,
        size = size,
    }
    vk.CmdCopyBuffer(cmd_buf, src.handle, dst.handle, 1, &copy_region)

    vk_check(vk.EndCommandBuffer(cmd_buf))

    submit_info := vk.SubmitInfo {
        sType = .SUBMIT_INFO,
        commandBufferCount = 1,
        pCommandBuffers = &cmd_buf,
    }
    vk_check(vk.QueueSubmit(queue, 1, &submit_info, {}))
    vk_check(vk.QueueWaitIdle(queue))
}

destroy_buffer :: proc(using ctx: ^Vulkan_Context, buf: ^Buffer)
{
    vk.FreeMemory(device, buf.mem, nil)
    vk.DestroyBuffer(device, buf.handle, nil)

    buf^ = {}
}

create_sbt_buffer :: proc(using ctx: ^Vulkan_Context, shader_handle_storage: []byte, num_groups: u32) -> Buffer
{
    assert(auto_cast len(shader_handle_storage) == rt_handle_size * num_groups)

    size := num_groups * rt_base_align
    staging_buf := create_buffer(ctx, 1, int(size), { .TRANSFER_SRC }, { .HOST_VISIBLE, .HOST_COHERENT }, {})
    defer destroy_buffer(ctx, &staging_buf)

    data: rawptr
    vk.MapMemory(device, staging_buf.mem, 0, vk.DeviceSize(size), {}, &data)
    for group_idx in 0..<num_groups
    {
        mem.copy(rawptr(uintptr(data) + uintptr(group_idx * rt_base_align)),
                 &shader_handle_storage[group_idx * rt_handle_size],
                 int(rt_handle_size))
    }
    data_tmp := cast([^]byte) data
    data_slice := data_tmp[:size]
    fmt.printfln("%x", data_slice)
    vk.UnmapMemory(device, staging_buf.mem)

    res := create_buffer(ctx, 1, int(size), { .SHADER_BINDING_TABLE_KHR, .TRANSFER_DST, .SHADER_DEVICE_ADDRESS }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    copy_buffer(ctx, staging_buf, res, vk.DeviceSize(size))

    // TEMPORARY CMD_BUF!!!
    cmd_pool_ci := vk.CommandPoolCreateInfo {
        sType = .COMMAND_POOL_CREATE_INFO,
        queueFamilyIndex = queue_family_idx,
        flags = { .TRANSIENT }
    }
    cmd_pool: vk.CommandPool
    vk_check(vk.CreateCommandPool(device, &cmd_pool_ci, nil, &cmd_pool))
    defer vk.DestroyCommandPool(device, cmd_pool, nil)

    cmd_buf_ai := vk.CommandBufferAllocateInfo {
        sType = .COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool = cmd_pool,
        level = .PRIMARY,
        commandBufferCount = 1,
    }
    cmd_buf: vk.CommandBuffer
    vk_check(vk.AllocateCommandBuffers(device, &cmd_buf_ai, &cmd_buf))
    defer vk.FreeCommandBuffers(device, cmd_pool, 1, &cmd_buf)

    cmd_buf_bi := vk.CommandBufferBeginInfo {
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        flags = { .ONE_TIME_SUBMIT },
    }
    vk_check(vk.BeginCommandBuffer(cmd_buf, &cmd_buf_bi))

    barrier := vk.MemoryBarrier2 {
        sType = .MEMORY_BARRIER_2,
        srcStageMask = { .TRANSFER },
        srcAccessMask = { .TRANSFER_WRITE },
        dstStageMask = { .RAY_TRACING_SHADER_KHR },
        dstAccessMask = { .SHADER_READ },
    }
    vk.CmdPipelineBarrier2(cmd_buf, &{
        sType = .DEPENDENCY_INFO,
        memoryBarrierCount = 1,
        pMemoryBarriers = &barrier,
    })

    vk_check(vk.EndCommandBuffer(cmd_buf))

    submit_info := vk.SubmitInfo {
        sType = .SUBMIT_INFO,
        commandBufferCount = 1,
        pCommandBuffers = &cmd_buf,
    }
    vk_check(vk.QueueSubmit(queue, 1, &submit_info, {}))
    vk_check(vk.QueueWaitIdle(queue))

    return res
}

find_mem_type :: proc(using ctx: ^Vulkan_Context, type_filter: u32, properties: vk.MemoryPropertyFlags) -> u32
{
    mem_properties: vk.PhysicalDeviceMemoryProperties
    vk.GetPhysicalDeviceMemoryProperties(phys_device, &mem_properties)
    for i in 0..<mem_properties.memoryTypeCount
    {
        if (type_filter & (1 << i) != 0) &&
           (mem_properties.memoryTypes[i].propertyFlags & properties) == properties {
            return i
        }
    }

    panic("Vulkan Error: Could not find suitable memory type!")
}

create_instance :: proc() -> (vk.Instance, vk.DebugUtilsMessengerEXT)
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
        pfnUserCallback = vk_debug_callback
    }

    next: rawptr
    next = &debug_messenger_ci

    instance: vk.Instance
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
    }, nil, &instance))

    vk.load_proc_addresses_instance(instance)
    assert(vk.DestroyInstance != nil, "Failed to load Vulkan instance API")

    debug_messenger: vk.DebugUtilsMessengerEXT
    vk_check(vk.CreateDebugUtilsMessengerEXT(instance, &debug_messenger_ci, nil, &debug_messenger))

    return instance, debug_messenger
}

create_ctx :: proc(instance: vk.Instance, surface: vk.SurfaceKHR) -> Vulkan_Context
{
    phys_device_count: u32
    vk_check(vk.EnumeratePhysicalDevices(instance, &phys_device_count, nil))
    if phys_device_count == 0 do fatal_error("Did not find any GPUs!")
    phys_devices := make([]vk.PhysicalDevice, phys_device_count, context.temp_allocator)
    vk_check(vk.EnumeratePhysicalDevices(instance, &phys_device_count, raw_data(phys_devices)))

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
            vk_check(vk.GetPhysicalDeviceSurfaceSupportKHR(candidate, u32(i), surface, &supports_present))

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

    queue_priority := f32(1)
    queue_create_infos := []vk.DeviceQueueCreateInfo {
        {
            sType = .DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex = queue_family_idx,
            queueCount = 1,
            pQueuePriorities = &queue_priority,
        },
    }

    device_extensions := []cstring {
        vk.KHR_SWAPCHAIN_EXTENSION_NAME,
        vk.EXT_SHADER_OBJECT_EXTENSION_NAME,
        vk.KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        vk.KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        vk.KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        vk.EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME,
    }

    next: rawptr
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

    device_ci := vk.DeviceCreateInfo {
        sType = .DEVICE_CREATE_INFO,
        pNext = next,
        queueCreateInfoCount = u32(len(queue_create_infos)),
        pQueueCreateInfos = raw_data(queue_create_infos),
        enabledExtensionCount = u32(len(device_extensions)),
        ppEnabledExtensionNames = raw_data(device_extensions),
    }
    device: vk.Device
    vk_check(vk.CreateDevice(chosen_phys_device, &device_ci, nil, &device))

    vk.load_proc_addresses_device(device)
    if vk.BeginCommandBuffer == nil do fatal_error("Failed to load Vulkan device API")

    queue: vk.Queue
    vk.GetDeviceQueue(device, queue_family_idx, 0, &queue)

    // Useful constants
    rt_handle_alignment: u32
    rt_base_align: u32
    rt_handle_size: u32
    {
        rt_properties := vk.PhysicalDeviceRayTracingPipelinePropertiesKHR {
            sType = .PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR
        }
        properties := vk.PhysicalDeviceProperties2 {
            sType = .PHYSICAL_DEVICE_PROPERTIES_2,
            pNext = &rt_properties
        }

        vk.GetPhysicalDeviceProperties2(chosen_phys_device, &properties)

        rt_handle_alignment = rt_properties.shaderGroupHandleAlignment
        rt_base_align       = rt_properties.shaderGroupBaseAlignment
        rt_handle_size      = rt_properties.shaderGroupHandleSize
    }

    return {
        phys_device = chosen_phys_device,
        device = device,
        queue = queue,
        queue_family_idx = queue_family_idx,
        rt_handle_alignment = rt_handle_alignment,
        rt_base_align = rt_base_align,
        rt_handle_size = rt_handle_size,
    }
}

vk_check :: proc(result: vk.Result, location := #caller_location)
{
    if result != .SUCCESS
    {
        fatal_error("Vulkan failure: %", result, location = location)
    }
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



vk_debug_callback :: proc "system" (severity: vk.DebugUtilsMessageSeverityFlagsEXT,
                                    types: vk.DebugUtilsMessageTypeFlagsEXT,
                                    callback_data: ^vk.DebugUtilsMessengerCallbackDataEXT,
                                    user_data: rawptr) -> b32
{
    context = runtime.default_context()
    // TODO
    //context.logger = vk_logger

    level: log.Level
    if .ERROR in severity        do level = .Error
    else if .WARNING in severity do level = .Warning
    else if .INFO in severity    do level = .Info
    else                         do level = .Debug
    log.log(level, callback_data.pMessage)

    return false
}
