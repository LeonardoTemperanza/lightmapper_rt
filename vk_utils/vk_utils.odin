
package test

import vk "vendor:vulkan"
import "core:mem"
import "base:runtime"
import "core:log"

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

create_sbt_buffer :: proc(device: vk.Device, phys_device: vk.PhysicalDevice, cmd_buf: vk.CommandBuffer, shader_handle_storage: []byte, num_groups: u32) -> Buffer
{
    rt_info := get_rt_info(phys_device)
    assert(auto_cast len(shader_handle_storage) == rt_info.handle_size * num_groups)

    size := num_groups * rt_info.base_align
    staging_buf := create_buffer(device, phys_device, size, { .TRANSFER_SRC }, { .HOST_VISIBLE, .HOST_COHERENT }, {})
    defer destroy_buffer(device, &staging_buf)

    data: rawptr
    vk.MapMemory(device, staging_buf.mem, 0, vk.DeviceSize(size), {}, &data)
    for group_idx in 0..<num_groups
    {
        mem.copy(rawptr(uintptr(data) + uintptr(group_idx * rt_info.base_align)),
                 &shader_handle_storage[group_idx * rt_info.handle_size],
                 int(rt_info.handle_size))
    }
    vk.UnmapMemory(device, staging_buf.mem)

    res := create_buffer(device, phys_device, size, { .SHADER_BINDING_TABLE_KHR, .TRANSFER_DST, .SHADER_DEVICE_ADDRESS }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    copy_buffer(cmd_buf, staging_buf, res, vk.DeviceSize(size))
    return res
}

copy_buffer :: proc(cmd_buf: vk.CommandBuffer, src: Buffer, dst: Buffer, size: vk.DeviceSize)
{
    copy_region := vk.BufferCopy {
        srcOffset = 0,
        dstOffset = 0,
        size = size,
    }
    vk.CmdCopyBuffer(cmd_buf, src.handle, dst.handle, 1, &copy_region)

    memory_barrier_safe_slow(cmd_buf)
}

upload_buffer :: proc(device: vk.Device, phys_device: vk.PhysicalDevice, cmd_buf: vk.CommandBuffer, buf: []byte, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, allocate_flags: vk.MemoryAllocateFlags) -> Buffer
{
    size := u32(len(buf))

    staging_buf_usage := vk.BufferUsageFlags { .TRANSFER_SRC }
    staging_buf_properties := vk.MemoryPropertyFlags { .HOST_VISIBLE, .HOST_COHERENT }
    staging_buf := create_buffer(device, phys_device, size, staging_buf_usage, staging_buf_properties, {})
    defer destroy_buffer(device, &staging_buf)

    data: rawptr
    vk.MapMemory(device, staging_buf.mem, 0, vk.DeviceSize(size), {}, &data)
    mem.copy(data, raw_data(buf), cast(int) size)
    vk.UnmapMemory(device, staging_buf.mem)

    res := create_buffer(device, phys_device, size, usage, properties, allocate_flags)
    copy_buffer(cmd_buf, staging_buf, res, vk.DeviceSize(size))

    barrier := vk.MemoryBarrier2 {
        sType = .MEMORY_BARRIER_2,
        srcStageMask = { .TRANSFER },
        srcAccessMask = { .TRANSFER_WRITE },
        dstStageMask = { .ACCELERATION_STRUCTURE_BUILD_KHR, .VERTEX_INPUT },
        dstAccessMask = { .ACCELERATION_STRUCTURE_READ_KHR, .VERTEX_ATTRIBUTE_READ },
    }
    vk.CmdPipelineBarrier2(cmd_buf, &{
        sType = .DEPENDENCY_INFO,
        memoryBarrierCount = 1,
        pMemoryBarriers = &barrier,
    })

    return res
}

destroy_buffer :: proc(device: vk.Device, buf: ^Buffer)
{
    vk.FreeMemory(device, buf.mem, nil)
    vk.DestroyBuffer(device, buf.handle, nil)

    buf^ = {}
}

// Images

Image :: struct
{
    img: vk.Image,
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
    vk_check(vk.CreateImage(device, &image_ci, nil, &res.img))

    mem_requirements: vk.MemoryRequirements
    vk.GetImageMemoryRequirements(device, res.img, &mem_requirements)

    // Create image memory
    memory_ai := vk.MemoryAllocateInfo {
        sType = .MEMORY_ALLOCATE_INFO,
        allocationSize = mem_requirements.size,
        memoryTypeIndex = find_mem_type(phys_device, mem_requirements.memoryTypeBits, { })
    }
    vk_check(vk.AllocateMemory(device, &memory_ai, nil, &res.mem))
    vk.BindImageMemory(device, res.img, res.mem, 0)

    res.layout = .UNDEFINED
    image_barrier_safe_slow(&res, cmd_buf, .GENERAL)

    // Create view
    image_view_ci := vk.ImageViewCreateInfo {
        sType = .IMAGE_VIEW_CREATE_INFO,
        image = res.img,
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

destroy_image :: proc(device: vk.Device, image: ^Image)
{
    vk.DestroyImageView(device, image.view, nil)
    vk.FreeMemory(device, image.mem, nil)
    vk.DestroyImage(device, image.img, nil)
    image^ = {}
}

// Command buffers

Tmp_Cmd_Buf :: struct
{
    test: u32,
}

create_tmp_cmd_buf :: proc(device: vk.Device) -> Tmp_Cmd_Buf
{
    return {}
}

destroy_tmp_cmd_buf :: proc(using tmp_cmd_buf: ^Tmp_Cmd_Buf, device: vk.Device)
{

}

// Barriers

memory_barrier_safe_slow :: proc(cmd_buf: vk.CommandBuffer)
{
    barrier := vk.MemoryBarrier2 {
        sType = .MEMORY_BARRIER_2,
        srcStageMask = { .ALL_COMMANDS },
        srcAccessMask = { .MEMORY_WRITE },
        dstStageMask = { .ALL_COMMANDS },
        dstAccessMask = { .MEMORY_READ, .MEMORY_WRITE },
    }
    vk.CmdPipelineBarrier2(cmd_buf, &{
        sType = .DEPENDENCY_INFO,
        memoryBarrierCount = 1,
        pMemoryBarriers = &barrier,
    })
}

image_barrier_safe_slow :: proc(image: ^Image, cmd_buf: vk.CommandBuffer, new_layout: vk.ImageLayout)
{
    barrier := []vk.ImageMemoryBarrier2 {
        {
            sType = .IMAGE_MEMORY_BARRIER_2,
            image = image.img,
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

RT_Info :: struct
{
    handle_alignment: u32,
    base_align: u32,
    handle_size: u32,
}

get_rt_info :: proc(phys_device: vk.PhysicalDevice) -> RT_Info
{
    res: RT_Info

    rt_properties := vk.PhysicalDeviceRayTracingPipelinePropertiesKHR {
        sType = .PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR
    }
    properties := vk.PhysicalDeviceProperties2 {
        sType = .PHYSICAL_DEVICE_PROPERTIES_2,
        pNext = &rt_properties
    }
    vk.GetPhysicalDeviceProperties2(phys_device, &properties)

    res.handle_alignment = rt_properties.shaderGroupHandleAlignment
    res.base_align       = rt_properties.shaderGroupBaseAlignment
    res.handle_size      = rt_properties.shaderGroupHandleSize
    return res
}
