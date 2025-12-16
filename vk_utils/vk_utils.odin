
package vk_utils

import vk "vendor:vulkan"
import "core:mem"
import "base:runtime"
import "core:log"
import intr "base:intrinsics"

// Utilities to make using Vulkan a little more bearable. A lot of stuff here
// is not really optimal, mostly for prototyping or for things that don't
// need good performance.

// Buffers
Buffer :: struct
{
    handle: vk.Buffer,
    mem: vk.DeviceMemory,
    size: vk.DeviceSize,
}

create_buffer :: proc(device: vk.Device, phys_device: vk.PhysicalDevice, byte_size: u32, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, allocate_flags: vk.MemoryAllocateFlags) -> Buffer
{
    res: Buffer
    res.size = vk.DeviceSize(byte_size)

    if byte_size <= 0 do return res

    buf_ci := vk.BufferCreateInfo {
        sType = .BUFFER_CREATE_INFO,
        size = cast(vk.DeviceSize) (byte_size),
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
        memoryTypeIndex = find_mem_type(phys_device, mem_requirements.memoryTypeBits, properties)
    }
    vk_check(vk.AllocateMemory(device, &memory_ai, nil, &res.mem))

    vk.BindBufferMemory(device, res.handle, res.mem, 0)

    return res
}

create_sbt_buffer :: proc(device: vk.Device, phys_device: vk.PhysicalDevice, queue: vk.Queue, cmd_pool: vk.CommandPool, pipeline: vk.Pipeline, group_count: u32) -> Buffer
{
    align_up :: proc(x, align: u32) -> (aligned: u32) {
        assert(0 == (align & (align - 1)), "must align to a power of two")
        return (x + (align - 1)) &~ (align - 1)
    }

    rt_info := get_rt_info(phys_device)

    data_size := rt_info.handle_size * group_count
    shader_handle_storage := make([]byte, data_size)
    defer delete(shader_handle_storage)
    vk_check(vk.GetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, group_count, int(data_size), raw_data(shader_handle_storage)))

    raygen_size    := align_up(rt_info.handle_size, rt_info.handle_align);
    rayhit_size    := align_up(rt_info.handle_size, rt_info.handle_align);
    raymiss_size   := align_up(rt_info.handle_size, rt_info.handle_align);
    callable_size  := u32(0)

    raygen_offset := u32(0)
    rayhit_offset := align_up(raygen_offset + raygen_size, rt_info.base_align)
    raymiss_offset := align_up(rayhit_offset + rayhit_size, rt_info.base_align)
    callable_offset := align_up(raymiss_offset + raymiss_size, rt_info.base_align)

    buf_size := callable_offset + callable_size

    staging_usage := vk.BufferUsageFlags { .TRANSFER_SRC }
    staging_properties := vk.MemoryPropertyFlags { .HOST_VISIBLE, .HOST_COHERENT }
    staging_buf := create_buffer(device, phys_device, buf_size, staging_usage, staging_properties, {})
    defer destroy_buffer(device, &staging_buf)

    data: rawptr
    vk.MapMemory(device, staging_buf.mem, 0, vk.DeviceSize(buf_size), {}, &data)
    intr.mem_copy(rawptr(uintptr(data) + uintptr(raygen_offset)),  &shader_handle_storage[0 * rt_info.handle_size], rt_info.handle_size)
    intr.mem_copy(rawptr(uintptr(data) + uintptr(raymiss_offset)), &shader_handle_storage[1 * rt_info.handle_size], rt_info.handle_size)
    intr.mem_copy(rawptr(uintptr(data) + uintptr(rayhit_offset)),  &shader_handle_storage[2 * rt_info.handle_size], rt_info.handle_size)
    vk.UnmapMemory(device, staging_buf.mem)

    cmd_buf := begin_tmp_cmd_buf(device, cmd_pool)
    res := create_buffer(device, phys_device, buf_size, { .SHADER_BINDING_TABLE_KHR, .TRANSFER_DST, .SHADER_DEVICE_ADDRESS }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    copy_buffer(cmd_buf, staging_buf, res, vk.DeviceSize(buf_size))
    end_tmp_cmd_buf(device, cmd_pool, queue, cmd_buf)

    return res
}

copy_buffer :: proc(cmd_buf: vk.CommandBuffer, src, dst: Buffer, size: vk.DeviceSize)
{
    copy_regions := []vk.BufferCopy {
        {
            srcOffset = 0,
            dstOffset = 0,
            size = size,
        }
    }
    vk.CmdCopyBuffer(cmd_buf, src.handle, dst.handle, u32(len(copy_regions)), raw_data(copy_regions))
}

upload_buffer :: proc(device: vk.Device, phys_device: vk.PhysicalDevice, queue: vk.Queue, cmd_pool: vk.CommandPool, buf: []$T, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, allocate_flags: vk.MemoryAllocateFlags) -> Buffer
{
    byte_slice := mem.byte_slice(raw_data(buf), len(buf) * size_of(T))
    return upload_buffer_raw(device, phys_device, queue, cmd_pool, byte_slice, usage, properties, allocate_flags)
}

upload_buffer_raw :: proc(device: vk.Device, phys_device: vk.PhysicalDevice, queue: vk.Queue, cmd_pool: vk.CommandPool, buf: []byte, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, allocate_flags: vk.MemoryAllocateFlags) -> Buffer
{
    if len(buf) <= 0 do return {}

    staging_buf_usage := vk.BufferUsageFlags { .TRANSFER_SRC }
    staging_buf_properties := vk.MemoryPropertyFlags { .HOST_VISIBLE, .HOST_COHERENT }
    staging_buf := create_buffer(device, phys_device, auto_cast len(buf), staging_buf_usage, staging_buf_properties, {})
    defer destroy_buffer(device, &staging_buf)

    data: rawptr
    vk.MapMemory(device, staging_buf.mem, 0, vk.DeviceSize(len(buf)), {}, &data)
    mem.copy(data, raw_data(buf), len(buf))
    vk.UnmapMemory(device, staging_buf.mem)

    dst_usage := usage | { .TRANSFER_DST }
    res := create_buffer(device, phys_device, auto_cast len(buf), dst_usage, properties, allocate_flags)

    cmd_buf := begin_tmp_cmd_buf(device, cmd_pool)
    copy_buffer(cmd_buf, staging_buf, res, vk.DeviceSize(len(buf)))
    end_tmp_cmd_buf(device, cmd_pool, queue, cmd_buf)
    return res
}

destroy_buffer :: proc(device: vk.Device, buf: ^Buffer)
{
    vk.FreeMemory(device, buf.mem, nil)
    vk.DestroyBuffer(device, buf.handle, nil)

    buf^ = {}
}

Shared_Buffer :: struct($T: typeid)
{
    buf: []T,
    buf_shared: Buffer,
    buf_gpu: Buffer
}

create_shared_buffer :: proc(device: vk.Device, phys_device: vk.PhysicalDevice, $T: typeid, num_elems: u32, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, allocate_flags: vk.MemoryAllocateFlags) -> Shared_Buffer(T)
{
    res: Persistently_Mapped_Buffer(T)

    staging_usage := vk.BufferUsageFlags { .TRANSFER_SRC }
    staging_properties := vk.MemoryPropertyFlags { .HOST_VISIBLE, .HOST_COHERENT }
    res.buf_shared = create_buffer(device, phys_device, size_of(T) * num_elems, staging_usage, staging_properties, {})
    res.buf_gpu = create_buffer(device, phys_device, size_of(T) * num_elems, usage, properties, allocate_flags)

    data: rawptr
    vk.MapMemory(device, res.buf_shared.mem, 0, vk.DeviceSize(res.buf_shared.size), {}, &data)
    res.buf = slice.from_ptr(cast(^T) data, num_elems)
    return res
}

shared_buffer_submit :: proc(using buf: ^Shared_Buffer, cmd_buf: vk.CommandBuffer)
{
    copy_buffer(cmd_buf, buf.buf_shared, buf.buf_gpu, vk.DeviceSize(buf.buf_shared.size))
}

destroy_shared_buffer :: proc(device: vk.Device, buf: ^Shared_Buffer)
{
    vk.UnmapMemory(device, buf.mem)
    destroy_buffer(device, buf.buf_shared)
    destroy_buffer(device, buf.buf_gpu)
}

// Images

Image :: struct
{
    handle: vk.Image,
    mem: vk.DeviceMemory,
    view: vk.ImageView,
    width: u32,
    height: u32,
    layout: vk.ImageLayout,
}

create_image :: proc(device: vk.Device, phys_device: vk.PhysicalDevice, cmd_buf: vk.CommandBuffer, ci: vk.ImageCreateInfo) -> Image
{
    res: Image

    image_ci := ci
    vk_check(vk.CreateImage(device, &image_ci, nil, &res.handle))

    mem_requirements: vk.MemoryRequirements
    vk.GetImageMemoryRequirements(device, res.handle, &mem_requirements)

    // Create image memory
    memory_ai := vk.MemoryAllocateInfo {
        sType = .MEMORY_ALLOCATE_INFO,
        allocationSize = mem_requirements.size,
        memoryTypeIndex = find_mem_type(phys_device, mem_requirements.memoryTypeBits, { })
    }
    vk_check(vk.AllocateMemory(device, &memory_ai, nil, &res.mem))
    vk.BindImageMemory(device, res.handle, res.mem, 0)

    res.layout = .UNDEFINED
    image_barrier_safe_slow(&res, cmd_buf, .GENERAL)

    // Create view
    image_view_ci := vk.ImageViewCreateInfo {
        sType = .IMAGE_VIEW_CREATE_INFO,
        image = res.handle,
        viewType = .D2,
        format = image_ci.format,
        subresourceRange = {
            aspectMask = { .COLOR },
            levelCount = 1,
            layerCount = 1,
        }
    }
    vk_check(vk.CreateImageView(device, &image_view_ci, nil, &res.view))

    res.width = ci.extent.width
    res.height = ci.extent.height
    return res
}

upload_image_rgba8 :: proc(device: vk.Device, phys_device: vk.PhysicalDevice, queue: vk.Queue, cmd_pool: vk.CommandPool, queue_family_idx: u32, image_cpu: [][4]u8, width: u32, height: u32, usage: vk.ImageUsageFlags, srgb: bool) -> Image
{
    assert(width * height == u32(len(image_cpu)))

    staging_buf_usage := vk.BufferUsageFlags { .TRANSFER_SRC }
    staging_buf_properties := vk.MemoryPropertyFlags { .HOST_VISIBLE, .HOST_COHERENT }
    staging_buf := create_buffer(device, phys_device, auto_cast len(image_cpu) * size_of(image_cpu[0]), staging_buf_usage, staging_buf_properties, {})
    defer destroy_buffer(device, &staging_buf)

    data: rawptr
    vk.MapMemory(device, staging_buf.mem, 0, vk.DeviceSize(len(image_cpu) * size_of(image_cpu[0])), {}, &data)
    mem.copy(data, raw_data(image_cpu), len(image_cpu) * size_of(image_cpu[0]))
    vk.UnmapMemory(device, staging_buf.mem)

    dst_usage := usage | { .TRANSFER_DST }

    cmd_buf := begin_tmp_cmd_buf(device, cmd_pool)

    res := create_image(device, phys_device, cmd_buf, {
        sType = .IMAGE_CREATE_INFO,
        flags = {},
        imageType = .D2,
        format = .R8G8B8A8_SRGB if srgb else .R8G8B8A8_UNORM,
        extent = {
            width = width,
            height = height,
            depth = 1,
        },
        mipLevels = 1,
        arrayLayers = 1,
        samples = { ._1 },
        usage = dst_usage,
        sharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = raw_data([]u32 { queue_family_idx }),
        initialLayout = .UNDEFINED,
    })
    vk.CmdCopyBufferToImage2(cmd_buf, &vk.CopyBufferToImageInfo2 {
        sType = .COPY_BUFFER_TO_IMAGE_INFO_2,
        pNext = nil,
        srcBuffer = staging_buf.handle,
        dstImage = res.handle,
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
                width = width,
                height = height,
                depth = 1,
            },
        },
    })

    end_tmp_cmd_buf(device, cmd_pool, queue, cmd_buf)

    return res
}

destroy_image :: proc(device: vk.Device, image: ^Image)
{
    vk.DestroyImageView(device, image.view, nil)
    vk.FreeMemory(device, image.mem, nil)
    vk.DestroyImage(device, image.handle, nil)
    image^ = {}
}

// Command buffers

begin_tmp_cmd_buf :: proc(device: vk.Device, cmd_pool: vk.CommandPool) -> vk.CommandBuffer
{
    cmd_buf_ai := vk.CommandBufferAllocateInfo {
        sType = .COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool = cmd_pool,
        level = .PRIMARY,
        commandBufferCount = 1,
    }
    cmd_buf: vk.CommandBuffer
    vk_check(vk.AllocateCommandBuffers(device, &cmd_buf_ai, &cmd_buf))

    cmd_buf_bi := vk.CommandBufferBeginInfo {
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        flags = { .ONE_TIME_SUBMIT },
    }
    vk_check(vk.BeginCommandBuffer(cmd_buf, &cmd_buf_bi))

    return cmd_buf
}

end_tmp_cmd_buf :: proc(device: vk.Device, cmd_pool: vk.CommandPool, queue: vk.Queue, cmd_buf: vk.CommandBuffer)
{
    vk_check(vk.EndCommandBuffer(cmd_buf))

    // Submit and wait
    {
        fence_info := vk.FenceCreateInfo{
            sType = .FENCE_CREATE_INFO,
        }

        fence: vk.Fence
        vk_check(vk.CreateFence(device, &fence_info, nil, &fence))
        defer vk.DestroyFence(device, fence, nil)

        to_submit := []vk.CommandBuffer { cmd_buf }
        submit_info := vk.SubmitInfo{
            sType              = .SUBMIT_INFO,
            commandBufferCount = u32(len(to_submit)),
            pCommandBuffers    = raw_data(to_submit),
        }
        vk_check(vk.QueueSubmit(queue, 1, &submit_info, fence))

        // Block until upload is done
        vk_check(vk.WaitForFences(device, 1, &fence, true, u64(-1)))
    }

    to_free := []vk.CommandBuffer { cmd_buf }
    vk.FreeCommandBuffers(device, cmd_pool, u32(len(to_free)), raw_data(to_free))
}

// Barriers

image_barrier_safe_slow :: proc(image: ^Image, cmd_buf: vk.CommandBuffer, new_layout: vk.ImageLayout)
{
    barrier := []vk.ImageMemoryBarrier2 {
        {
            sType = .IMAGE_MEMORY_BARRIER_2,
            image = image.handle,
            subresourceRange = {
                aspectMask = { .COLOR },
                levelCount = 1,
                layerCount = 1,
            },
            oldLayout = image.layout,
            newLayout = new_layout,
            srcStageMask = { .ALL_COMMANDS },
            srcAccessMask = { .MEMORY_WRITE },
            dstStageMask = { .ALL_COMMANDS },
            dstAccessMask = { .MEMORY_READ, .MEMORY_WRITE },
        },
    }
    vk.CmdPipelineBarrier2(cmd_buf, &{
        sType = .DEPENDENCY_INFO,
        imageMemoryBarrierCount = u32(len(barrier)),
        pImageMemoryBarriers = raw_data(barrier),
    })

    image.layout = new_layout
}

image_barrier_transition_to_present :: proc(image: ^Image, cmd_buf: vk.CommandBuffer)
{
    image_barrier_safe_slow(image, cmd_buf, .PRESENT_SRC_KHR)
}

// Misc

vk_check :: proc(result: vk.Result, location := #caller_location)
{
    if result != .SUCCESS {
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

find_mem_type :: proc(phys_device: vk.PhysicalDevice, type_filter: u32, properties: vk.MemoryPropertyFlags) -> u32
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

get_buffer_device_address :: proc(device: vk.Device, buffer: Buffer) -> vk.DeviceAddress
{
    info := vk.BufferDeviceAddressInfo {
        sType = .BUFFER_DEVICE_ADDRESS_INFO,
        buffer = buffer.handle
    }
    return vk.GetBufferDeviceAddress(device, &info)
}

// Raytracing

RT_Info :: struct
{
    handle_align: u32,
    base_align: u32,
    handle_size: u32,
}

get_rt_info :: proc(phys_device: vk.PhysicalDevice) -> RT_Info
{
    res: RT_Info

    rt_properties := vk.PhysicalDeviceRayTracingPipelinePropertiesKHR {
        sType = .PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
    }
    properties := vk.PhysicalDeviceProperties2 {
        sType = .PHYSICAL_DEVICE_PROPERTIES_2,
        pNext = &rt_properties
    }
    vk.GetPhysicalDeviceProperties2(phys_device, &properties)

    res.handle_align = rt_properties.shaderGroupHandleAlignment
    res.base_align   = rt_properties.shaderGroupBaseAlignment
    res.handle_size  = rt_properties.shaderGroupHandleSize
    return res
}

Accel_Structure :: struct
{
    handle: vk.AccelerationStructureKHR,
    buf: Buffer,
    addr: vk.DeviceAddress
}

create_blas :: proc(device: vk.Device, phys_device: vk.PhysicalDevice, queue: vk.Queue, cmd_pool: vk.CommandPool, positions: Buffer, indices: Buffer, verts_count: u32, idx_count: u32) -> Accel_Structure
{
    blas: Accel_Structure

    tri_data := vk.AccelerationStructureGeometryTrianglesDataKHR {
        sType = .ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        vertexFormat = .R32G32B32_SFLOAT,
        vertexData = {
            deviceAddress = get_buffer_device_address(device, positions)
        },
        vertexStride = size_of([3]f32),
        maxVertex = verts_count,
        indexType = .UINT32,
        indexData = {
            deviceAddress = get_buffer_device_address(device, indices)
        },
    }

    blas_geometry := vk.AccelerationStructureGeometryKHR {
        sType = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        geometryType = .TRIANGLES,
        flags = { .OPAQUE },
        geometry = {
            triangles = tri_data
        }
    }

    primitive_count := idx_count / 3

    range_info := vk.AccelerationStructureBuildRangeInfoKHR {
        primitiveCount = primitive_count,
        primitiveOffset = 0,
        firstVertex = 0,
        transformOffset = 0,
    }

    range_info_ptrs := []^vk.AccelerationStructureBuildRangeInfoKHR {
        &range_info,
    }

    build_info := vk.AccelerationStructureBuildGeometryInfoKHR {
        sType = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        flags = { .PREFER_FAST_TRACE },
        geometryCount = 1,
        pGeometries = &blas_geometry,
        type = .BOTTOM_LEVEL,
    }

    size_info := vk.AccelerationStructureBuildSizesInfoKHR {
        sType = .ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
    }
    vk.GetAccelerationStructureBuildSizesKHR(device, .DEVICE, &build_info, &primitive_count, &size_info)

    blas_usages := vk.BufferUsageFlags { .ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS }
    blas.buf = create_buffer(device, phys_device, auto_cast size_info.accelerationStructureSize, blas_usages, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })

    // Create the scratch buffer for blas building
    scratch_usages := vk.BufferUsageFlags { .ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS, .STORAGE_BUFFER }
    scratch_buf := create_buffer(device, phys_device, auto_cast size_info.buildScratchSize, scratch_usages, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })

    // Build acceleration structure
    blas_ci := vk.AccelerationStructureCreateInfoKHR {
        sType = .ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        buffer = blas.buf.handle,
        size = size_info.accelerationStructureSize,
        type = .BOTTOM_LEVEL,
    }
    vk_check(vk.CreateAccelerationStructureKHR(device, &blas_ci, nil, &blas.handle))

    {
        cmd_buf := begin_tmp_cmd_buf(device, cmd_pool)
        defer end_tmp_cmd_buf(device, cmd_pool, queue, cmd_buf)
        build_info.dstAccelerationStructure = blas.handle
        build_info.scratchData.deviceAddress = get_buffer_device_address(device, scratch_buf)

        vk.CmdBuildAccelerationStructuresKHR(cmd_buf, 1, &build_info, auto_cast raw_data(range_info_ptrs))
    }

    // Get device address
    addr_info := vk.AccelerationStructureDeviceAddressInfoKHR {
        sType = .ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        accelerationStructure = blas.handle,
    }
    blas.addr = vk.GetAccelerationStructureDeviceAddressKHR(device, &addr_info)

    return blas
}
