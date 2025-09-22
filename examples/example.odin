
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

package main

import "core:fmt"
import "core:log"
import "core:slice"
import "core:os"
import "base:runtime"
import "vendor:glfw"
import vk "vendor:vulkan"

NUM_FRAMES_IN_FLIGHT :: 2
NUM_SWAPCHAIN_IMAGES :: 3

vk_logger: log.Logger
glfw_logger: log.Logger

Vulkan_Ctx :: struct
{
    phys_device: vk.PhysicalDevice,
    device: vk.Device,
}

Vulkan_Per_Frame :: struct
{
    fence: vk.Fence,
    acquire_semaphore: vk.Semaphore,
    cmd_pool: vk.CommandPool,
    cmd_buf: vk.CommandBuffer,
}

Swapchain :: struct
{
    handle: vk.SwapchainKHR,
    width, height: u32,
    images: []vk.Image,
    image_views: []vk.ImageView,
    present_semaphores: []vk.Semaphore,
}

main :: proc()
{
    console_logger := log.create_console_logger()
    defer log.destroy_console_logger(console_logger)
    vk_logger = log.create_console_logger()
    vk_logger.options = { .Level, .Terminal_Color }
    defer log.destroy_console_logger(vk_logger)
    context.logger = console_logger

    glfw_logger = console_logger

    glfw.SetErrorCallback(proc "c" (error: i32, desc: cstring) {
        context = runtime.default_context()
        context.logger = glfw_logger
        log.errorf("GLFW Error {}: {}", error, desc)
    })

    ok_g := glfw.Init()
    if !ok_g do fatal_error("Could not initialize GLFW.")

    glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
    window := glfw.CreateWindow(1024, 768, "Lightmapper RT", nil, nil)
    if window == nil do fatal_error("Could not create window.")

    vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))

    instance, debug_messenger := create_instance()
    defer
    {
        vk.DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nil)
        vk.DestroyInstance(instance, nil)
    }

    surface: vk.SurfaceKHR
    vk_check(glfw.CreateWindowSurface(instance, window, nil, &surface))
    defer vk.DestroySurfaceKHR(instance, surface, nil)

    phys_device, device, queue, queue_family_idx := create_device(instance, surface)
    defer vk.DestroyDevice(device, nil)

    width, height := glfw.GetFramebufferSize(window)

    swapchain := create_swapchain(phys_device, device, surface, u32(width), u32(height))
    defer destroy_swapchain(device, swapchain)

    shaders := create_shaders(device)
    defer destroy_shaders(device, shaders)

    vk_frames := create_vk_frames(device, queue_family_idx)
    frame_idx := u32(0)

    for !glfw.WindowShouldClose(window)
    {
        glfw.PollEvents()

        vk_frame := vk_frames[frame_idx]
        vk_check(vk.WaitForFences(device, 1, &vk_frame.fence, true, max(u64)))
        vk_check(vk.ResetFences(device, 1, &vk_frame.fence))

        image_idx: u32
        vk_check(vk.AcquireNextImageKHR(device, swapchain.handle, max(u64), vk_frame.acquire_semaphore, 0, &image_idx))

        present_semaphore := swapchain.present_semaphores[image_idx]

        vk_check(vk.ResetCommandPool(device, vk_frame.cmd_pool, {}))

        cmd_buf := vk_frame.cmd_buf

        vk_check(vk.BeginCommandBuffer(cmd_buf, &{
            sType = .COMMAND_BUFFER_BEGIN_INFO,
            flags = { .ONE_TIME_SUBMIT },
        }))

        transition_to_color_attachment_barrier := vk.ImageMemoryBarrier2 {
            sType = .IMAGE_MEMORY_BARRIER_2,
            image = swapchain.images[image_idx],
            subresourceRange = {
                aspectMask = {.COLOR},
                levelCount = 1,
                layerCount = 1,
            },
            oldLayout = .UNDEFINED,
            newLayout = .COLOR_ATTACHMENT_OPTIMAL,
            srcStageMask = {.ALL_COMMANDS},
            srcAccessMask = {.MEMORY_READ},
            dstStageMask = {.COLOR_ATTACHMENT_OUTPUT},
            dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
        }
        vk.CmdPipelineBarrier2(cmd_buf, &vk.DependencyInfo {
            sType = .DEPENDENCY_INFO,
            imageMemoryBarrierCount = 1,
            pImageMemoryBarriers = &transition_to_color_attachment_barrier,
        })

        color_attachment := vk.RenderingAttachmentInfo {
            sType = .RENDERING_ATTACHMENT_INFO,
            imageView = swapchain.image_views[image_idx],
            imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
            loadOp = .CLEAR,
            storeOp = .STORE,
            clearValue = {
                color = { float32 = { 1, 0, 0, 1 } }
            }
        }
        rendering_info := vk.RenderingInfo {
            sType = .RENDERING_INFO,
            renderArea = {
                offset = { 0, 0 },
                extent = { swapchain.width, swapchain.height }
            },
            layerCount = 1,
            colorAttachmentCount = 1,
            pColorAttachments = &color_attachment,
        }
        vk.CmdBeginRendering(cmd_buf, &rendering_info)

        render(cmd_buf, shaders)

        vk.CmdEndRendering(cmd_buf)

        transition_to_present_src_barrier := vk.ImageMemoryBarrier2 {
            sType = .IMAGE_MEMORY_BARRIER_2,
            image = swapchain.images[image_idx],
            subresourceRange = {
                aspectMask = { .COLOR },
                levelCount = 1,
                layerCount = 1,
            },
            oldLayout = .COLOR_ATTACHMENT_OPTIMAL,
            newLayout = .PRESENT_SRC_KHR,
            srcStageMask = { .COLOR_ATTACHMENT_OUTPUT },
            srcAccessMask = { .COLOR_ATTACHMENT_WRITE },
            dstStageMask = {},
            dstAccessMask = {},
        }
        vk.CmdPipelineBarrier2(cmd_buf, &{
            sType = .DEPENDENCY_INFO,
            imageMemoryBarrierCount = 1,
            pImageMemoryBarriers = &transition_to_present_src_barrier,
        })

        vk_check(vk.EndCommandBuffer(cmd_buf))

        wait_stage_flags := vk.PipelineStageFlags { .COLOR_ATTACHMENT_OUTPUT }
        submit_info := vk.SubmitInfo {
            sType = .SUBMIT_INFO,
            waitSemaphoreCount = 1,
            pWaitSemaphores = &vk_frame.acquire_semaphore,
            pWaitDstStageMask = &wait_stage_flags,
            commandBufferCount = 1,
            pCommandBuffers = &cmd_buf,
            signalSemaphoreCount = 1,
            pSignalSemaphores = &present_semaphore,
        }
        vk_check(vk.QueueSubmit(queue, 1, &submit_info, vk_frame.fence))

        vk_check(vk.QueuePresentKHR(queue, &{
            sType = .PRESENT_INFO_KHR,
            waitSemaphoreCount = 1,
            pWaitSemaphores = &present_semaphore,
            swapchainCount = 1,
            pSwapchains = &swapchain.handle,
            pImageIndices = &image_idx,
        }))

        frame_idx = (frame_idx + 1) % NUM_FRAMES_IN_FLIGHT

        free_all(context.temp_allocator)
    }
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

    extensions := slice.concatenate([][]cstring {
        glfw.GetRequiredInstanceExtensions(),
        {
            vk.EXT_DEBUG_UTILS_EXTENSION_NAME,
            vk.KHR_WIN32_SURFACE_EXTENSION_NAME,
        },
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

vk_debug_callback :: proc "system" (severity: vk.DebugUtilsMessageSeverityFlagsEXT,
                                    types: vk.DebugUtilsMessageTypeFlagsEXT,
                                    callback_data: ^vk.DebugUtilsMessengerCallbackDataEXT,
                                    user_data: rawptr) -> b32
{
    context = runtime.default_context()
    context.logger = vk_logger

    level: log.Level
    if .ERROR in severity        do level = .Error
    else if .WARNING in severity do level = .Warning
    else if .INFO in severity    do level = .Info
    else                         do level = .Debug
    log.log(level, callback_data.pMessage)

    return false
}

create_device :: proc(instance: vk.Instance, surface: vk.SurfaceKHR) -> (vk.PhysicalDevice, vk.Device, vk.Queue, u32)
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

    extensions := []cstring {
        vk.KHR_SWAPCHAIN_EXTENSION_NAME,
        vk.EXT_SHADER_OBJECT_EXTENSION_NAME,
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

    device_ci := vk.DeviceCreateInfo {
        sType = .DEVICE_CREATE_INFO,
        pNext = next,
        queueCreateInfoCount = u32(len(queue_create_infos)),
        pQueueCreateInfos = raw_data(queue_create_infos),
        enabledExtensionCount = u32(len(extensions)),
        ppEnabledExtensionNames = raw_data(extensions),
    }
    device: vk.Device
    vk_check(vk.CreateDevice(chosen_phys_device, &device_ci, nil, &device))

    vk.load_proc_addresses_device(device)
    if vk.BeginCommandBuffer == nil do fatal_error("Failed to load Vulkan device API")

    queue: vk.Queue
    vk.GetDeviceQueue(device, queue_family_idx, 0, &queue)

    return chosen_phys_device, device, queue, queue_family_idx
}

create_swapchain :: proc(phys_device: vk.PhysicalDevice, device: vk.Device, surface: vk.SurfaceKHR, width: u32, height: u32) -> Swapchain
{
    res: Swapchain

    surface_caps: vk.SurfaceCapabilitiesKHR
    vk_check(vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(phys_device, surface, &surface_caps))

    image_count := max(NUM_SWAPCHAIN_IMAGES, surface_caps.minImageCount)
    if surface_caps.maxImageCount != 0 do image_count = min(image_count, surface_caps.maxImageCount)

    surface_format_count: u32
    vk_check(vk.GetPhysicalDeviceSurfaceFormatsKHR(phys_device, surface, &surface_format_count, nil))
    surface_formats := make([]vk.SurfaceFormatKHR, surface_format_count, context.temp_allocator)
    vk_check(vk.GetPhysicalDeviceSurfaceFormatsKHR(phys_device, surface, &surface_format_count, raw_data(surface_formats)))

    surface_format := surface_formats[0]
    for candidate in surface_formats
    {
        if candidate == {.B8G8R8A8_SRGB, .SRGB_NONLINEAR}
        {
            surface_format = candidate
            break
        }
    }

    present_mode_count: u32
    vk_check(vk.GetPhysicalDeviceSurfacePresentModesKHR(phys_device, surface, &present_mode_count, nil))
    present_modes := make([]vk.PresentModeKHR, present_mode_count, context.temp_allocator)
    vk_check(vk.GetPhysicalDeviceSurfacePresentModesKHR(phys_device, surface, &present_mode_count, raw_data(present_modes)))

    present_mode := vk.PresentModeKHR.FIFO
    for candidate in present_modes {
        if candidate == .MAILBOX {
            present_mode = candidate
            break
        }
    }

    res.width = width
    res.height = height

    swapchain_ci := vk.SwapchainCreateInfoKHR {
        sType = .SWAPCHAIN_CREATE_INFO_KHR,
        surface = surface,
        minImageCount = image_count,
        imageFormat = surface_format.format,
        imageColorSpace = surface_format.colorSpace,
        imageExtent = { res.width, res.height },
        imageArrayLayers = 1,
        imageUsage = { .COLOR_ATTACHMENT },
        preTransform = surface_caps.currentTransform,
        compositeAlpha = { .OPAQUE },
        presentMode = present_mode,
        clipped = true,
    }
    vk_check(vk.CreateSwapchainKHR(device, &swapchain_ci, nil, &res.handle))

    vk_check(vk.GetSwapchainImagesKHR(device, res.handle, &image_count, nil))
    res.images = make([]vk.Image, image_count, context.allocator)
    vk_check(vk.GetSwapchainImagesKHR(device, res.handle, &image_count, raw_data(res.images)))

    res.image_views = make([]vk.ImageView, image_count, context.allocator)
    for image, i in res.images
    {
        image_ci := vk.ImageViewCreateInfo {
            sType = .IMAGE_VIEW_CREATE_INFO,
            image = image,
            viewType = .D2,
            format = surface_format.format,
            subresourceRange = {
                aspectMask = { .COLOR },
                levelCount = 1,
                layerCount = 1,
            },
        }
        vk_check(vk.CreateImageView(device, &image_ci, nil, &res.image_views[i]))
    }

    res.present_semaphores = make([]vk.Semaphore, image_count, context.allocator)

    semaphore_ci := vk.SemaphoreCreateInfo { sType = .SEMAPHORE_CREATE_INFO }
    for &semaphore in res.present_semaphores {
        vk_check(vk.CreateSemaphore(device, &semaphore_ci, nil, &semaphore))
    }

    return res
}

destroy_swapchain :: proc(device: vk.Device, swapchain: Swapchain)
{
    delete(swapchain.images)
    for semaphore in swapchain.present_semaphores {
        vk.DestroySemaphore(device, semaphore, nil)
    }
    delete(swapchain.present_semaphores)
    for image_view in swapchain.image_views {
        vk.DestroyImageView(device, image_view, nil)
    }
    delete(swapchain.image_views)
    vk.DestroySwapchainKHR(device, swapchain.handle, nil)
}

create_vk_frames :: proc(device: vk.Device, queue_family_idx: u32) -> [NUM_FRAMES_IN_FLIGHT]Vulkan_Per_Frame
{
    res: [NUM_FRAMES_IN_FLIGHT]Vulkan_Per_Frame
    for &frame in res
    {
        cmd_pool_ci := vk.CommandPoolCreateInfo {
            sType = .COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex = queue_family_idx,
            flags = { .TRANSIENT }
        }
        vk_check(vk.CreateCommandPool(device, &cmd_pool_ci, nil, &frame.cmd_pool))

        cmd_buf_ai := vk.CommandBufferAllocateInfo {
            sType = .COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool = frame.cmd_pool,
            level = .PRIMARY,
            commandBufferCount = 1,
        }
        vk_check(vk.AllocateCommandBuffers(device, &cmd_buf_ai, &frame.cmd_buf))

        semaphore_ci := vk.SemaphoreCreateInfo { sType = .SEMAPHORE_CREATE_INFO }
        vk_check(vk.CreateSemaphore(device, &semaphore_ci, nil, &frame.acquire_semaphore))

        fence_ci := vk.FenceCreateInfo {
            sType = .FENCE_CREATE_INFO,
            flags = { .SIGNALED },
        }
        vk_check(vk.CreateFence(device, &fence_ci, nil, &frame.fence))
    }

    return res
}

destroy_vk_frames :: proc(device: vk.Device, frames: [NUM_FRAMES_IN_FLIGHT]Vulkan_Per_Frame)
{
    for frame in frames
    {
        vk.DestroyCommandPool(device, frame.cmd_pool, nil)
        vk.DestroySemaphore(device, frame.acquire_semaphore, nil)
        vk.DestroyFence(device, frame.fence, nil)
    }
}

create_shaders :: proc(device: vk.Device) -> Shaders
{
    res: Shaders

    push_constant_ranges := []vk.PushConstantRange {
        {
            stageFlags = { .VERTEX, .FRAGMENT },
            size = 128,
        }
    }
    pipeline_layout_ci := vk.PipelineLayoutCreateInfo {
        sType = .PIPELINE_LAYOUT_CREATE_INFO,
        pushConstantRangeCount = u32(len(push_constant_ranges)),
        pPushConstantRanges = raw_data(push_constant_ranges),
    }
    vk_check(vk.CreatePipelineLayout(device, &pipeline_layout_ci, nil, &res.pipeline_layout))

    vert_code := load_file("shaders/shader.vert.spv", context.temp_allocator)
    frag_code := load_file("shaders/shader.frag.spv", context.temp_allocator)
    shader_cis := [2]vk.ShaderCreateInfoEXT {
        {
            sType = .SHADER_CREATE_INFO_EXT,
            codeType = .SPIRV,
            codeSize = len(vert_code),
            pCode = raw_data(vert_code),
            pName = "main",
            stage = { .VERTEX },
            nextStage = { .FRAGMENT },
            flags = { .LINK_STAGE },
            pushConstantRangeCount = u32(len(push_constant_ranges)),
            pPushConstantRanges = raw_data(push_constant_ranges)
        },
        {
            sType = .SHADER_CREATE_INFO_EXT,
            codeType = .SPIRV,
            codeSize = len(frag_code),
            pCode = raw_data(frag_code),
            pName = "main",
            stage = { .FRAGMENT },
            flags = { .LINK_STAGE },
            pushConstantRangeCount = u32(len(push_constant_ranges)),
            pPushConstantRanges = raw_data(push_constant_ranges),
        },
    }
    shaders: [2]vk.ShaderEXT
    vk_check(vk.CreateShadersEXT(device, 2, raw_data(&shader_cis), nil, raw_data(&shaders)))

    res.test_vert = shaders[0]
    res.test_frag = shaders[1]
    return res
}

destroy_shaders :: proc(device: vk.Device, shaders: Shaders)
{

}

/*
create_shaders :: proc()
{
    push_constants_ranges := []vk.PushConstantRange {
        {
            stageFlags = { .VERTEX, .FRAGMENT },
            size = 128,
        }
    }

    pipeline_layout_ci := vk.PipelineLayoutCreateInfo {
        sType = .PIPELINE_LAYOUT_CREATE_INFO,
        pushConstantRangeCount = u32(len(push_constant_ranges)),
        pPushConstantRanges = raw_data(push_constant_ranges),
    }
    vk_check(vk.CreatePipelineLayout(device, &pipeline_layout_ci, nil, &pipeline_layout))

    vert_code := [1]byte {}

    shader_cis := [2]vk.ShaderCreateInfoEXT {
        {
            sType = .SHADER_CREATE_INFO_EXT,
            codeType = .SPIRV,
            codeSize = len(vert_code),
            pCode = raw_data(vert_code),
            pName = "main",
            stage = { .VERTEX },
            nextStage = { .FRAGMENT },
            flags = { .LINK_STAGE },
            pushConstantRangeCount = u32(len(push_constant_ranges)),
            pPushConstantRanges = raw_data(push_constant_ranges),
        },

    }

    shaders: [2]vk.Shader

    vk_check(vk.CreateShadersEXT(device, u32(len(shader_cis)), raw_data(&shader_cis), nil, raw_data(shaders)))
}
*/

Shaders :: struct
{
    pipeline_layout: vk.PipelineLayout,
    test_vert: vk.ShaderEXT,
    test_frag: vk.ShaderEXT,
}

render :: proc(cmd_buf: vk.CommandBuffer, shaders: Shaders)
{
/*

    shader_stages := [2]vk.ShaderStageFlags { { .VERTEX }, { .FRAGMENT } }
    shaders := [2]vk.Shader {}
    // vk.CmdBindShadersEXT(cmd_buf, 2, &shader_stages[0], )

    vk.CmdSetViewportWithCount(cmd_buf, 1, &{
        width = f32(swapchain.width),
        height = f32(swapchain.height),
        minDepth = 0,
        maxDepth = 1,
    })
    vk.CmdSetScissorWithCount(cmd_buf, 1, &{
        extent = {
            width = swapchain.width,
            height = swapchain.height,
        }
    })
    vk.CmdSetRasterizerDiscardEnable(cmd_buf, false)

    vk.CmdSetVertexInputEXT(cmd_buf, 0, nil, 0, nil)
    vk.CmdSetPrimitiveTopology(cmd_buf, .TRIANGLE_LIST)
    vk.CmdSetPrimitiveRestartEnable(cmd_buf, false)

    vk.CmdSetRasterizationSamplesEXT(cmd, { ._1 })
    sample_mask := vk.SampleMask(1)
    vk.CmdSetSampleMaskEXT(cmd, { ._1 }, &sample_mask)
    vk.CmdSetAlphaToCoverageEnableEXT(cmd_buf, false)

    vk.CmdSetPolygonModeEXT(cmd_buf, .FILL)
    vk.CmdSetCullMode(cmd_buf, {})
    vk.CmdSetFrontFace(cmd_buf, .COUNTER_CLOCKWISE)

    vk.CmdSetDepthTestEnable(cmd_buf, false)
    vk.CmdSetDepthWriteEnable(cmd_buf, false)
    vk.CmdSetDepthBiasEnable(cmd_buf, false)

    vk.CmdSetStencilTestEnable(cmd_buf, false)
    b32_false := b32(false)
    vk.CmdSetColorBlendEnableEXT(cmd_buf, 0, 1, &b32_false)

    color_mask := vk.ColorComponentFlags { .R, .G, .B, .A }
    vk.CmdSetColorWriteMaskEXT(cmd_buf, 0, 1, &color_mask)

    Push :: struct {
        color: [3]f32,
    }
    push := Push { color = { 0, 0.5, 0 } }
    vk.CmdPushConstants(cmd, pipeline_layout, { .VERTEX, .FRAGMENT }, 0, size_of(push), &push)

    vk.CmdDraw(cmd, 3, 1, 0, 0)

    */
}

vk_check :: proc(result: vk.Result, location := #caller_location)
{
    if result != .SUCCESS
    {
        when ODIN_DEBUG {
            log.panicf("Vulkan failure: {}", result, location = location)
        } else {
            fatal_error("Vulkan failure: {}", result)
        }
    }
}

fatal_error :: proc(fmt: string, args: ..any)
{
    log.panicf(fmt)
}

load_file :: proc(filename: string, allocator: runtime.Allocator) -> []byte {
	data, ok := os.read_entire_file_from_filename(filename, allocator)
	log.assertf(ok, "Could not load file {}", filename)
	return data
}
