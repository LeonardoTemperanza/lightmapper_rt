
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
import "core:math/linalg"
import "core:math"
import "core:mem"
import "core:time"
import "core:sync"
import "core:c"
import sdl "vendor:sdl3"
import vk "vendor:vulkan"
import lm "../"

import "ufbx"

NUM_FRAMES_IN_FLIGHT :: 1
NUM_SWAPCHAIN_IMAGES :: 2

vk_logger: log.Logger
glfw_logger: log.Logger

Vk_Ctx :: struct
{
    phys_device: vk.PhysicalDevice,
    device: vk.Device,
    queue: vk.Queue,
    lm_queue: vk.Queue,
    queue_family_idx: u32,
    rt_handle_alignment: u32,
    rt_handle_size: u32,
    rt_base_align: u32,
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

GBuffers :: struct
{
    world_pos: Image,
    world_normals: Image,
}

LIGHTMAP_SIZE :: 4096

main :: proc()
{
    ok_i := sdl.Init({ .VIDEO })
    assert(ok_i)

    console_logger := log.create_console_logger()
    defer log.destroy_console_logger(console_logger)
    vk_logger = log.create_console_logger()
    vk_logger.options = { .Level, .Terminal_Color }
    defer log.destroy_console_logger(vk_logger)
    context.logger = console_logger

    ts_freq := sdl.GetPerformanceFrequency()

    window_flags :: sdl.WindowFlags {
        .RESIZABLE,
        .HIGH_PIXEL_DENSITY,
        .VULKAN,
    }
    window := sdl.CreateWindow("Lightmapper RT Example", 1800, 1800, window_flags)
    ensure(window != nil)

    vk.load_proc_addresses_global(cast(rawptr) sdl.Vulkan_GetVkGetInstanceProcAddr())

    instance, debug_messenger := create_instance()
    defer
    {
        vk.DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nil)
        vk.DestroyInstance(instance, nil)
    }

    surface: vk.SurfaceKHR
    ok_s := sdl.Vulkan_CreateSurface(window, instance, nil, &surface)
    if !ok_s do fatal_error("Could not create vulkan surface.")
    defer sdl.Vulkan_DestroySurface(instance, surface, nil)

    vk_ctx := create_ctx(instance, surface)
    defer destroy_ctx(&vk_ctx)

    width, height: c.int
    sdl.GetWindowSizeInPixels(window, &width, &height)

    swapchain := create_swapchain(&vk_ctx, surface, u32(width), u32(height))
    defer destroy_swapchain(&vk_ctx, swapchain)

    depth_image, depth_image_view := create_depth_texture(&vk_ctx, u32(width), u32(height))

    shaders := create_shaders(&vk_ctx)
    defer destroy_shaders(&vk_ctx, shaders)

    scene := load_scene_fbx(&vk_ctx, "D:/lightmapper_test_scenes/ArchVis_RT.fbx")
    // defer destroy_scene(&vk_ctx, &scene)

    vk_frames := create_vk_frames(&vk_ctx)
    frame_idx := u32(0)

    now_ts := sdl.GetPerformanceCounter()
    max_delta_time: f32 = 1.0 / 10.0  // 10fps

    desc_pool_ci := vk.DescriptorPoolCreateInfo {
        sType = .DESCRIPTOR_POOL_CREATE_INFO,
        flags = { .FREE_DESCRIPTOR_SET },
        maxSets = 50,
        poolSizeCount = 3,
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
            }
        })
    }
    desc_pool: vk.DescriptorPool
    vk_check(vk.CreateDescriptorPool(vk_ctx.device, &desc_pool_ci, nil, &desc_pool))

    // Create linear sampler
    lightmap_sampler_ci := vk.SamplerCreateInfo {
        sType = .SAMPLER_CREATE_INFO,
        magFilter = .LINEAR,
        minFilter = .LINEAR,
        mipmapMode = .LINEAR,
        addressModeU = .REPEAT,
        addressModeV = .REPEAT,
        addressModeW = .REPEAT,
    }

    lightmap_sampler: vk.Sampler
    vk_check(vk.CreateSampler(vk_ctx.device, &lightmap_sampler_ci, nil, &lightmap_sampler))

    lm_vk_ctx := lm.Vulkan_Context {
        phys_device = vk_ctx.phys_device,
        device = vk_ctx.device,
        //queue = vk_ctx.lm_queue,
        queue = vk_ctx.queue,
        queue_family_idx = vk_ctx.queue_family_idx,
        rt_handle_alignment = vk_ctx.rt_handle_alignment,
        rt_handle_size = vk_ctx.rt_handle_size,
        rt_base_align = vk_ctx.rt_base_align,
    }
    lm_ctx := lm.init_test(lm_vk_ctx)

    lm_scene := lm.Scene {
        instances = scene.instances,
        meshes = scene.meshes,
        tlas = scene.tlas,
    }
    bake := lm.start_bake(&lm_ctx, lm_scene, {}, 4096, 1000, 1)

    // time.sleep(30 * time.Second)

    // Create lightmap desc set
    desc_set_ai := vk.DescriptorSetAllocateInfo {
        sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool = desc_pool,
        descriptorSetCount = 1,
        pSetLayouts = raw_data([]vk.DescriptorSetLayout { shaders.lm_desc_set_layout })
    }

    lm_desc_set: vk.DescriptorSet
    vk_check(vk.AllocateDescriptorSets(vk_ctx.device, &desc_set_ai, &lm_desc_set))

    // Update lightmap desc set
    writes := []vk.WriteDescriptorSet {
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = lm_desc_set,
            dstBinding = 0,
            descriptorCount = 1,
            descriptorType = .COMBINED_IMAGE_SAMPLER,
            pImageInfo = raw_data([]vk.DescriptorImageInfo {
                {
                    imageView = bake.lightmap_backbuffer.view,
                    imageLayout = .GENERAL,
                    sampler = lightmap_sampler,
                }
            })
        }
    }
    vk.UpdateDescriptorSets(vk_ctx.device, 1, raw_data(writes), 0, nil)

    for
    {
        sync.mutex_lock(bake.debug_mutex1)
        defer { sync.mutex_unlock(bake.debug_mutex0) }

        fmt.println("frame")

        proceed := handle_window_events(window)
        if !proceed do break

        last_ts := now_ts
        now_ts = sdl.GetPerformanceCounter()
        DELTA_TIME = min(max_delta_time, f32(f64((now_ts - last_ts)*1000) / f64(ts_freq)) / 1000.0)

        vk_frame := vk_frames[frame_idx]
        vk_check(vk.WaitForFences(vk_ctx.device, 1, &vk_frame.fence, true, max(u64)))
        vk_check(vk.ResetFences(vk_ctx.device, 1, &vk_frame.fence))

        //fmt.println("usage past fence")

        lm_info := lm.acquire_next_lightmap_view_vk(bake)

        image_idx: u32
        vk_check(vk.AcquireNextImageKHR(vk_ctx.device, swapchain.handle, max(u64), vk_frame.acquire_semaphore, 0, &image_idx))

        present_semaphore := swapchain.present_semaphores[image_idx]

        vk_check(vk.ResetCommandPool(vk_ctx.device, vk_frame.cmd_pool, {}))

        cmd_buf := vk_frame.cmd_buf

        vk_check(vk.BeginCommandBuffer(cmd_buf, &{
            sType = .COMMAND_BUFFER_BEGIN_INFO,
            flags = { .ONE_TIME_SUBMIT },
        }))

        transition_to_color_attachment_barrier := vk.ImageMemoryBarrier2 {
            sType = .IMAGE_MEMORY_BARRIER_2,
            image = swapchain.images[image_idx],
            subresourceRange = {
                aspectMask = { .COLOR },
                levelCount = 1,
                layerCount = 1,
            },
            oldLayout = .UNDEFINED,
            newLayout = .COLOR_ATTACHMENT_OPTIMAL,
            srcStageMask = { .ALL_COMMANDS },
            srcAccessMask = { .MEMORY_READ },
            dstStageMask = { .COLOR_ATTACHMENT_OUTPUT },
            dstAccessMask = { .COLOR_ATTACHMENT_WRITE },
        }
        vk.CmdPipelineBarrier2(cmd_buf, &vk.DependencyInfo {
            sType = .DEPENDENCY_INFO,
            imageMemoryBarrierCount = 1,
            pImageMemoryBarriers = &transition_to_color_attachment_barrier,
        })

        render_scene(&vk_ctx, cmd_buf, depth_image_view, swapchain.image_views[image_idx], lm_desc_set, shaders, swapchain, scene)

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
        next: rawptr
        next = &vk.TimelineSemaphoreSubmitInfo {
            sType = .TIMELINE_SEMAPHORE_SUBMIT_INFO,
            pNext = next,
            waitSemaphoreValueCount = 2,
            pWaitSemaphoreValues = raw_data([]u64 {
                0,
                lm_info.wait_value
            }),
            signalSemaphoreValueCount = 2,
            pSignalSemaphoreValues = raw_data([]u64 {
                0,
                lm_info.signal_value,
            })
        }
        submit_info := vk.SubmitInfo {
            sType = .SUBMIT_INFO,
            pNext = next,
            commandBufferCount = 1,
            pCommandBuffers = &cmd_buf,
            waitSemaphoreCount = 2,
            pWaitSemaphores = raw_data([]vk.Semaphore {
                vk_frame.acquire_semaphore,
                lm_info.sem,
            }),
            pWaitDstStageMask = raw_data([]vk.PipelineStageFlags {
                wait_stage_flags,
                wait_stage_flags,
            }),
            signalSemaphoreCount = 2,
            pSignalSemaphores = raw_data([]vk.Semaphore {
                present_semaphore,
                lm_info.sem
            }),
        }
        vk_check(vk.QueueSubmit(vk_ctx.queue, 1, &submit_info, vk_frame.fence))

        vk_check(vk.QueuePresentKHR(vk_ctx.queue, &{
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

    validation_feature := vk.ValidationFeatureEnableEXT.SYNCHRONIZATION_VALIDATION
    next = &vk.ValidationFeaturesEXT {
        sType = .VALIDATION_FEATURES_EXT,
        pNext = next,
        enabledValidationFeatureCount = 1,
        pEnabledValidationFeatures = &validation_feature,
    }

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

create_ctx :: proc(instance: vk.Instance, surface: vk.SurfaceKHR) -> Vk_Ctx
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
            queueCount = 2,
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
        vk.KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME
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
    device: vk.Device
    vk_check(vk.CreateDevice(chosen_phys_device, &device_ci, nil, &device))

    vk.load_proc_addresses_device(device)
    if vk.BeginCommandBuffer == nil do fatal_error("Failed to load Vulkan device API")

    queue: vk.Queue
    vk.GetDeviceQueue(device, queue_family_idx, 0, &queue)
    lm_queue: vk.Queue
    vk.GetDeviceQueue(device, queue_family_idx, 1, &lm_queue)

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
        lm_queue = lm_queue,
        queue_family_idx = queue_family_idx,
        rt_handle_alignment = rt_handle_alignment,
        rt_base_align = rt_base_align,
        rt_handle_size = rt_handle_size,
    }
}

destroy_ctx :: proc(using ctx: ^Vk_Ctx)
{
    vk.DestroyDevice(device, nil)
}

create_swapchain :: proc(using ctx: ^Vk_Ctx, surface: vk.SurfaceKHR, width: u32, height: u32) -> Swapchain
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
        image_view_ci := vk.ImageViewCreateInfo {
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
        vk_check(vk.CreateImageView(device, &image_view_ci, nil, &res.image_views[i]))
    }

    res.present_semaphores = make([]vk.Semaphore, image_count, context.allocator)

    semaphore_ci := vk.SemaphoreCreateInfo { sType = .SEMAPHORE_CREATE_INFO }
    for &semaphore in res.present_semaphores {
        vk_check(vk.CreateSemaphore(device, &semaphore_ci, nil, &semaphore))
    }

    return res
}

destroy_swapchain :: proc(using ctx: ^Vk_Ctx, swapchain: Swapchain)
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

Frame_Data :: struct
{
    world_to_view: matrix[4, 4]f32,
    view_to_proj: matrix[4, 4]f32
}

create_vk_frames :: proc(using ctx: ^Vk_Ctx) -> [NUM_FRAMES_IN_FLIGHT]Vulkan_Per_Frame
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

destroy_vk_frames :: proc(using ctx: ^Vk_Ctx, frames: [NUM_FRAMES_IN_FLIGHT]Vulkan_Per_Frame)
{
    for frame in frames
    {
        vk.DestroyCommandPool(device, frame.cmd_pool, nil)
        vk.DestroySemaphore(device, frame.acquire_semaphore, nil)
        vk.DestroyFence(device, frame.fence, nil)
    }
}

create_shaders :: proc(using ctx: ^Vk_Ctx) -> Shaders
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
    }

    // NOTE: Not using context.temp_allocator because it doesn't guarantee 4 byte alignment,
    // and vulkan requires the alignment of the spirv to be 4 byte.
    test_vert := load_file("shaders/shader.vert.spv", context.allocator)
    defer delete(test_vert)
    test_frag := load_file("shaders/shader.frag.spv", context.allocator)
    defer delete(test_frag)
    model_to_proj_vert := load_file("shaders/model_to_proj.vert.spv", context.allocator)
    defer delete(model_to_proj_vert)
    lit_frag := load_file("shaders/lit.frag.spv", context.allocator)

    // Create shader objects.
    {
        shader_cis := [?]vk.ShaderCreateInfoEXT {
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(test_vert),
                pCode = raw_data(test_vert),
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
                codeSize = len(test_frag),
                pCode = raw_data(test_frag),
                pName = "main",
                stage = { .FRAGMENT },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
            },
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(model_to_proj_vert),
                pCode = raw_data(model_to_proj_vert),
                pName = "main",
                stage = { .VERTEX },
                nextStage = { .FRAGMENT },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
                setLayoutCount = 1,
                pSetLayouts = &res.lm_desc_set_layout
            },
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(lit_frag),
                pCode = raw_data(lit_frag),
                pName = "main",
                stage = { .FRAGMENT },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
                setLayoutCount = 1,
                pSetLayouts = &res.lm_desc_set_layout
            }
        }
        shaders: [len(shader_cis)]vk.ShaderEXT
        vk_check(vk.CreateShadersEXT(device, len(shaders), raw_data(&shader_cis), nil, raw_data(&shaders)))
        res.test_vert = shaders[0]
        res.test_frag = shaders[1]
        res.model_to_proj = shaders[2]
        res.lit = shaders[3]
    }

    return res
}

destroy_shaders :: proc(using ctx: ^Vk_Ctx, shaders: Shaders)
{
    vk.DestroyPipelineLayout(device, shaders.pipeline_layout, nil)
    vk.DestroyShaderEXT(device, shaders.test_vert, nil)
    vk.DestroyShaderEXT(device, shaders.test_frag, nil)
}

Shaders :: struct
{
    pipeline_layout: vk.PipelineLayout,
    test_vert: vk.ShaderEXT,
    test_frag: vk.ShaderEXT,

    model_to_proj: vk.ShaderEXT,
    lit: vk.ShaderEXT,

    // Desc set layouts
    lm_desc_set_layout: vk.DescriptorSetLayout,
}

render_scene :: proc(using ctx: ^Vk_Ctx, cmd_buf: vk.CommandBuffer, depth_view: vk.ImageView, color_view: vk.ImageView, lightmap_desc_set: vk.DescriptorSet, shaders: Shaders, swapchain: Swapchain, scene: lm.Scene)
{
    depth_attachment := vk.RenderingAttachmentInfo {
        sType = .RENDERING_ATTACHMENT_INFO,
        imageView = depth_view,
        imageLayout = .DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        loadOp = .CLEAR,
        storeOp = .STORE,
        clearValue = {
            depthStencil = { 1.0, 0.0 }
        },
    }
    color_attachment := vk.RenderingAttachmentInfo {
        sType = .RENDERING_ATTACHMENT_INFO,
        imageView = color_view,
        imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
        loadOp = .CLEAR,
        storeOp = .STORE,
        clearValue = {
            color = { float32 = { 0.8, 0.8, 0.8, 1 } }
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
        pDepthAttachment = &depth_attachment,
    }

    vk.CmdBeginRendering(cmd_buf, &rendering_info)

    shader_stages := []vk.ShaderStageFlags { { .VERTEX }, { .GEOMETRY }, { .FRAGMENT } }
    to_bind := []vk.ShaderEXT { shaders.model_to_proj, vk.ShaderEXT(0), shaders.lit }
    assert(len(shader_stages) == len(to_bind))
    vk.CmdBindShadersEXT(cmd_buf, u32(len(shader_stages)), raw_data(shader_stages), raw_data(to_bind))

    viewport := vk.Viewport {
        width = f32(swapchain.width),
        height = f32(swapchain.height),
        minDepth = 0.0,
        maxDepth = 1.0,
    }
    vk.CmdSetViewportWithCount(cmd_buf, 1, &viewport)
    scissor := vk.Rect2D {
        extent = {
            width = swapchain.width,
            height = swapchain.height,
        }
    }
    vk.CmdSetScissorWithCount(cmd_buf, 1, &scissor)
    vk.CmdSetRasterizerDiscardEnable(cmd_buf, false)

    vert_input_bindings := [?]vk.VertexInputBindingDescription2EXT {
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
    vert_attributes := [?]vk.VertexInputAttributeDescription2EXT {
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
    vk.CmdSetVertexInputEXT(cmd_buf, len(vert_input_bindings), &vert_input_bindings[0], len(vert_attributes), &vert_attributes[0])
    vk.CmdSetPrimitiveTopology(cmd_buf, .TRIANGLE_LIST)
    vk.CmdSetPrimitiveRestartEnable(cmd_buf, false)

    vk.CmdSetConservativeRasterizationModeEXT(cmd_buf, .DISABLED)
    vk.CmdSetRasterizationSamplesEXT(cmd_buf, { ._1 })
    sample_mask := vk.SampleMask(1)
    vk.CmdSetSampleMaskEXT(cmd_buf, { ._1 }, &sample_mask)
    vk.CmdSetAlphaToCoverageEnableEXT(cmd_buf, false)

    vk.CmdSetPolygonModeEXT(cmd_buf, .FILL)
    vk.CmdSetCullMode(cmd_buf, { .BACK })
    vk.CmdSetFrontFace(cmd_buf, .COUNTER_CLOCKWISE)

    vk.CmdSetDepthCompareOp(cmd_buf, .LESS)
    vk.CmdSetDepthTestEnable(cmd_buf, true)
    vk.CmdSetDepthWriteEnable(cmd_buf, true)
    vk.CmdSetDepthBiasEnable(cmd_buf, false)
    vk.CmdSetDepthClipEnableEXT(cmd_buf, true)

    vk.CmdSetStencilTestEnable(cmd_buf, false)
    b32_false := b32(false)
    vk.CmdSetColorBlendEnableEXT(cmd_buf, 0, 1, &b32_false)

    color_mask := vk.ColorComponentFlags { .R, .G, .B, .A }
    vk.CmdSetColorWriteMaskEXT(cmd_buf, 0, 1, &color_mask)

    // Bind textures
    tmp := lightmap_desc_set
    vk.CmdBindDescriptorSets(cmd_buf, .GRAPHICS, shaders.pipeline_layout, 0, 1, &tmp, 0, nil)

    render_viewport_aspect_ratio := f32(1.0)

    world_to_view := compute_world_to_view()
    view_to_proj := linalg.matrix4_perspective_f32(math.RAD_PER_DEG * 59.0, render_viewport_aspect_ratio, 0.1, 1000.0, false)

    loop: for instance, i in scene.instances {
    // instance := scene.instances[30]; loop: {
        mesh := scene.meshes[instance.mesh_idx]

        if !mesh.lm_uvs_present { break loop }

        offset := vk.DeviceSize(0)
        vk.CmdBindVertexBuffers(cmd_buf, 0, 1, &mesh.pos.handle, &offset)
        vk.CmdBindVertexBuffers(cmd_buf, 1, 1, &mesh.normals.handle, &offset)
        if mesh.lm_uvs_present {
            vk.CmdBindVertexBuffers(cmd_buf, 2, 1, &mesh.lm_uvs.handle, &offset)
        }
        vk.CmdBindIndexBuffer(cmd_buf, mesh.indices_gpu.handle, 0, .UINT32)

        Push :: struct {
            model_to_world: matrix[4, 4]f32,
            normal_mat: matrix[4, 4]f32,
            world_to_proj: matrix[4, 4]f32,
            lm_uv_offset: [2]f32,
            lm_uv_scale: f32,
        }
        push := Push {
            model_to_world = instance.transform,
            normal_mat = linalg.transpose(linalg.inverse(instance.transform)),
            world_to_proj = view_to_proj * world_to_view,
            lm_uv_offset = instance.lm_offset,
            lm_uv_scale = instance.lm_scale,
        }
        vk.CmdPushConstants(cmd_buf, shaders.pipeline_layout, { .VERTEX, .FRAGMENT }, 0, size_of(push), &push)

        vk.CmdDrawIndexed(cmd_buf, mesh.idx_count, 1, 0, 0, 0)
    }

    vk.CmdEndRendering(cmd_buf)
}

compute_world_to_view :: proc() -> matrix[4, 4]f32
{
    return first_person_camera_view()
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

load_file :: proc(filename: string, allocator: runtime.Allocator) -> []byte
{
    data, ok := os.read_entire_file_from_filename(filename, allocator)
    log.assertf(ok, "Could not load file {}", filename)
    return data
}

/////////////////////////////////////
// This part can be ignored.

/*
destroy_mesh :: proc(using ctx: ^Vk_Ctx, mesh: ^Mesh)
{
    destroy_buffer(ctx, &mesh.pos)
    destroy_buffer(ctx, &mesh.normals)
    if mesh.lm_uvs_present {
        destroy_buffer(ctx, &mesh.lm_uvs)
    }
    destroy_buffer(ctx, &mesh.indices_gpu)

    mesh^ = {}
}

destroy_scene :: proc(using ctx: ^Vk_Ctx, scene: ^Scene)
{
    delete(scene.instances)
    for &mesh in scene.meshes {
        destroy_mesh(ctx, &mesh)
    }
    delete(scene.meshes)
    scene^ = {}
}
*/

xform_to_mat :: proc(pos: [3]f64, rot: quaternion256, scale: [3]f64) -> matrix[4, 4]f32
{
    return cast(matrix[4, 4]f32) (#force_inline linalg.matrix4_translate(pos) *
           #force_inline linalg.matrix4_from_quaternion(rot) *
           #force_inline linalg.matrix4_scale(scale))
}

create_sbt_buffer :: proc(using ctx: ^Vk_Ctx, shader_handle_storage: []byte, num_groups: u32) -> Buffer
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

create_vertex_buffer :: proc(using ctx: ^Vk_Ctx, verts: []$T) -> Buffer
{
    size := cast(vk.DeviceSize) (len(verts) * size_of(verts[0]))

    staging_buf := create_buffer(ctx, size_of(verts[0]), len(verts), { .TRANSFER_SRC }, { .HOST_VISIBLE, .HOST_COHERENT }, {})
    defer destroy_buffer(ctx, &staging_buf)

    data: rawptr
    vk.MapMemory(device, staging_buf.mem, 0, size, {}, &data)
    mem.copy(data, raw_data(verts), cast(int) size)
    vk.UnmapMemory(device, staging_buf.mem)

    res := create_buffer(ctx, size_of(verts[0]), len(verts), { .VERTEX_BUFFER, .TRANSFER_DST, .SHADER_DEVICE_ADDRESS, .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .STORAGE_BUFFER }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    copy_buffer(ctx, staging_buf, res, size)

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
        dstStageMask = { .ACCELERATION_STRUCTURE_BUILD_KHR, .VERTEX_INPUT },
        dstAccessMask = { .ACCELERATION_STRUCTURE_READ_KHR, .VERTEX_ATTRIBUTE_READ },
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

create_index_buffer :: proc(using ctx: ^Vk_Ctx, indices: []u32) -> Buffer
{
    size := cast(vk.DeviceSize) (len(indices) * size_of(indices[0]))

    staging_buf := create_buffer(ctx, size_of(indices[0]), len(indices), { .TRANSFER_SRC }, { .HOST_VISIBLE, .HOST_COHERENT }, {})
    defer destroy_buffer(ctx, &staging_buf)

    data: rawptr
    vk.MapMemory(device, staging_buf.mem, 0, size, {}, &data)
    mem.copy(data, raw_data(indices), cast(int) size)
    vk.UnmapMemory(device, staging_buf.mem)

    res := create_buffer(ctx, size_of(indices[0]), len(indices), { .INDEX_BUFFER, .TRANSFER_DST, .SHADER_DEVICE_ADDRESS, .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .STORAGE_BUFFER }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    copy_buffer(ctx, staging_buf, res, size)

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
        dstStageMask = { .ACCELERATION_STRUCTURE_BUILD_KHR, .VERTEX_INPUT },
        dstAccessMask = { .ACCELERATION_STRUCTURE_READ_KHR, .VERTEX_ATTRIBUTE_READ },
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

create_instances_buffer :: proc(using ctx: ^Vk_Ctx, instances: []vk.AccelerationStructureInstanceKHR) -> Buffer
{
    size := cast(vk.DeviceSize) (len(instances) * size_of(instances[0]))

    staging_buf := create_buffer(ctx, size_of(instances[0]), len(instances), { .TRANSFER_SRC }, { .HOST_VISIBLE, .HOST_COHERENT }, {})
    defer destroy_buffer(ctx, &staging_buf)

    data: rawptr
    vk.MapMemory(device, staging_buf.mem, 0, size, {}, &data)
    mem.copy(data, raw_data(instances), cast(int) size)
    vk.UnmapMemory(device, staging_buf.mem)

    res := create_buffer(ctx, size_of(instances[0]), len(instances), { .ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS, .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .TRANSFER_DST }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })
    copy_buffer(ctx, staging_buf, res, size)

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
        dstStageMask = { .ACCELERATION_STRUCTURE_BUILD_KHR },
        dstAccessMask = { .ACCELERATION_STRUCTURE_READ_KHR },
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

Buffer :: struct
{
    buf: vk.Buffer,
    mem: vk.DeviceMemory,
    size: vk.DeviceSize,
}

Image :: struct
{
    img: vk.Image,
    mem: vk.DeviceMemory,
    view: vk.ImageView,
    width: u32,
    height: u32,
}

create_buffer :: proc(using ctx: ^Vk_Ctx, el_size: int, count: int, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, allocate_flags: vk.MemoryAllocateFlags) -> Buffer
{
    res: Buffer
    res.size = vk.DeviceSize(el_size * count)

    buf_ci := vk.BufferCreateInfo {
        sType = .BUFFER_CREATE_INFO,
        size = cast(vk.DeviceSize) (el_size * count),
        usage = usage,
        sharingMode = .EXCLUSIVE,
    }
    vk_check(vk.CreateBuffer(device, &buf_ci, nil, &res.buf))

    mem_requirements: vk.MemoryRequirements
    vk.GetBufferMemoryRequirements(device, res.buf, &mem_requirements)

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

    vk.BindBufferMemory(device, res.buf, res.mem, 0)

    return res
}

copy_buffer :: proc(using ctx: ^Vk_Ctx, src: Buffer, dst: Buffer, size: vk.DeviceSize)
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
    vk.CmdCopyBuffer(cmd_buf, src.buf, dst.buf, 1, &copy_region)

    vk_check(vk.EndCommandBuffer(cmd_buf))

    submit_info := vk.SubmitInfo {
        sType = .SUBMIT_INFO,
        commandBufferCount = 1,
        pCommandBuffers = &cmd_buf,
    }
    vk_check(vk.QueueSubmit(queue, 1, &submit_info, {}))
    vk_check(vk.QueueWaitIdle(queue))
}

destroy_buffer :: proc(using ctx: ^Vk_Ctx, buf: ^Buffer)
{
    vk.FreeMemory(device, buf.mem, nil)
    vk.DestroyBuffer(device, buf.buf, nil)

    buf^ = {}
}

create_image :: proc(using ctx: ^Vk_Ctx, ci: vk.ImageCreateInfo, name := "") -> Image
{
    res: Image

    image_ci := ci
    vk_check(vk.CreateImage(device, &image_ci, nil, &res.img))

    fmt.printfln("Created %v: 0x%x", name, res.img)

    mem_requirements: vk.MemoryRequirements
    vk.GetImageMemoryRequirements(device, res.img, &mem_requirements)

    // Create image memory
    memory_ai := vk.MemoryAllocateInfo {
        sType = .MEMORY_ALLOCATE_INFO,
        allocationSize = mem_requirements.size,
        memoryTypeIndex = find_mem_type(ctx, mem_requirements.memoryTypeBits, { })
    }
    vk_check(vk.AllocateMemory(device, &memory_ai, nil, &res.mem))
    vk.BindImageMemory(device, res.img, res.mem, 0)

    // Perform transition to LAYOUT_GENERAL
    {
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

        transition_to_general_barrier := vk.ImageMemoryBarrier2 {
            sType = .IMAGE_MEMORY_BARRIER_2,
            image = res.img,
            subresourceRange = {
                aspectMask = { .COLOR },
                levelCount = 1,
                layerCount = 1,
            },
            oldLayout = .UNDEFINED,
            newLayout = .GENERAL,
            srcStageMask = { .ALL_COMMANDS },
            srcAccessMask = { .MEMORY_READ },
            dstStageMask = { .RAY_TRACING_SHADER_KHR },
            dstAccessMask = { .SHADER_WRITE, .SHADER_READ },
        }
        vk.CmdPipelineBarrier2(cmd_buf, &vk.DependencyInfo {
            sType = .DEPENDENCY_INFO,
            imageMemoryBarrierCount = 1,
            pImageMemoryBarriers = &transition_to_general_barrier,
        })

        vk_check(vk.EndCommandBuffer(cmd_buf))
    }

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

find_mem_type :: proc(using ctx: ^Vk_Ctx, type_filter: u32, properties: vk.MemoryPropertyFlags) -> u32
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

Key_State :: struct
{
    pressed: bool,
    pressing: bool,
    released: bool,
}

Input :: struct
{
    pressing_right_click: bool,
    keys: #sparse[sdl.Scancode]Key_State,

    mouse_dx: f32,  // pixels/dpi (inches), right is positive
    mouse_dy: f32,  // pixels/dpi (inches), up is positive
}

INPUT: Input
DELTA_TIME: f32

world_to_view_mat :: proc(cam_pos: [3]f32, cam_rot: quaternion128) -> matrix[4, 4]f32
{
    view_rot := linalg.normalize(linalg.quaternion_inverse(cam_rot))
    view_pos := -cam_pos
    return #force_inline linalg.matrix4_from_quaternion(view_rot) *
           #force_inline linalg.matrix4_translate(view_pos)
}

find_depth_format :: proc(using ctx: ^Vk_Ctx) -> vk.Format
{
    candidates := [?]vk.Format {
        .D32_SFLOAT,
        .D32_SFLOAT_S8_UINT,
        .D24_UNORM_S8_UINT
    }
    for format in candidates
    {
        props: vk.FormatProperties
        vk.GetPhysicalDeviceFormatProperties(phys_device, format, &props)
        if .DEPTH_STENCIL_ATTACHMENT in props.optimalTilingFeatures {
            return format
        }
    }

    fatal_error("Failed to find a good supported depth format!")
    return .D32_SFLOAT
}

create_depth_texture :: proc(using ctx: ^Vk_Ctx, width, height: u32) -> (vk.Image, vk.ImageView)
{
    image_ci := vk.ImageCreateInfo {
        sType = .IMAGE_CREATE_INFO,
        flags = {},
        imageType = .D2,
        format = find_depth_format(ctx),
        extent = {
            width = width,
            height = height,
            depth = 1,
        },
        mipLevels = 1,
        arrayLayers = 1,
        samples = { ._1 },
        usage = { .DEPTH_STENCIL_ATTACHMENT },
        sharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &queue_family_idx,
        initialLayout = .UNDEFINED,
    }
    image: vk.Image
    vk_check(vk.CreateImage(device, &image_ci, nil, &image))

    mem_requirements: vk.MemoryRequirements
    vk.GetImageMemoryRequirements(device, image, &mem_requirements)

    // Create image memory
    memory_ai := vk.MemoryAllocateInfo {
        sType = .MEMORY_ALLOCATE_INFO,
        allocationSize = mem_requirements.size,
        memoryTypeIndex = find_mem_type(ctx, mem_requirements.memoryTypeBits, { })
    }
    image_mem: vk.DeviceMemory
    vk_check(vk.AllocateMemory(device, &memory_ai, nil, &image_mem))
    vk.BindImageMemory(device, image, image_mem, 0)

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

    transition_to_depth_barrier := vk.ImageMemoryBarrier2 {
        sType = .IMAGE_MEMORY_BARRIER_2,
        image = image,
        subresourceRange = {
            aspectMask = { .DEPTH },
            levelCount = 1,
            layerCount = 1,
        },
        oldLayout = .UNDEFINED,
        newLayout = .DEPTH_ATTACHMENT_OPTIMAL,
        srcStageMask = { .ALL_COMMANDS },
        srcAccessMask = { .MEMORY_READ },
        dstStageMask = { .EARLY_FRAGMENT_TESTS },
        dstAccessMask = { .DEPTH_STENCIL_ATTACHMENT_READ, .DEPTH_STENCIL_ATTACHMENT_WRITE },
    }
    vk.CmdPipelineBarrier2(cmd_buf, &{
        sType = .DEPENDENCY_INFO,
        imageMemoryBarrierCount = 1,
        pImageMemoryBarriers = &transition_to_depth_barrier,
    })

    vk_check(vk.EndCommandBuffer(cmd_buf))

    image_view_ci := vk.ImageViewCreateInfo {
        sType = .IMAGE_VIEW_CREATE_INFO,
        image = image,
        viewType = .D2,
        format = image_ci.format,
        subresourceRange = {
            aspectMask = { .DEPTH },
            levelCount = 1,
            layerCount = 1,
        }
    }
    image_view: vk.ImageView
    vk_check(vk.CreateImageView(device, &image_view_ci, nil, &image_view))

    return image, image_view
}


handle_window_events :: proc(window: ^sdl.Window) -> (proceed: bool)
{
    // Reset "one-shot" inputs
    for &key in INPUT.keys
    {
        key.pressed = false
        key.released = false
    }
    INPUT.mouse_dx = 0
    INPUT.mouse_dy = 0

    event: sdl.Event
    proceed = true
    for sdl.PollEvent(&event)
    {
        #partial switch event.type
        {
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
                        INPUT.pressing_right_click = true
                    }
                } else if event.type == .MOUSE_BUTTON_UP {
                    if event.button == sdl.BUTTON_RIGHT {
                        INPUT.pressing_right_click = false
                    }
                }
            }
            case .KEY_DOWN, .KEY_UP:
            {
                event := event.key
                if event.repeat do break

                if event.type == .KEY_DOWN
                {
                    INPUT.keys[event.scancode].pressed = true
                    INPUT.keys[event.scancode].pressing = true
                }
                else
                {
                    INPUT.keys[event.scancode].pressing = false
                    INPUT.keys[event.scancode].released = true
                }
            }
            case .MOUSE_MOTION:
            {
                event := event.motion
                INPUT.mouse_dx += event.xrel
                INPUT.mouse_dy -= event.yrel  // In sdl, up is negative
            }
        }
    }

    return
}

first_person_camera_view :: proc() -> matrix[4, 4]f32
{
    @(static) cam_pos: [3]f32 = { 0, 2.5, -10 }

    @(static) angle: [2]f32

    cam_rot: quaternion128 = 1

    mouse_sensitivity := math.to_radians_f32(0.2)  // Radians per pixel
    mouse: [2]f32
    if INPUT.pressing_right_click
    {
        mouse.x = INPUT.mouse_dx * mouse_sensitivity
        mouse.y = INPUT.mouse_dy * mouse_sensitivity
    }

    angle += mouse

    // Wrap angle.x
    for angle.x < 0 do angle.x += 2*math.PI
    for angle.x > 2*math.PI do angle.x -= 2*math.PI

    angle.y = clamp(angle.y, math.to_radians_f32(-90), math.to_radians_f32(90))
    y_rot := linalg.quaternion_angle_axis(angle.y, [3]f32 { -1, 0, 0 })
    x_rot := linalg.quaternion_angle_axis(angle.x, [3]f32 { 0, 1, 0 })
    cam_rot = x_rot * y_rot

    // Movement
    @(static) cur_vel: [3]f32
    move_speed: f32 : 6.0
    move_speed_fast: f32 : 15.0
    move_accel: f32 : 300.0

    keyboard_dir_xz: [3]f32
    keyboard_dir_y: f32
    if INPUT.pressing_right_click
    {
        keyboard_dir_xz.x = f32(int(INPUT.keys[.D].pressing) - int(INPUT.keys[.A].pressing))
        keyboard_dir_xz.z = f32(int(INPUT.keys[.W].pressing) - int(INPUT.keys[.S].pressing))
        keyboard_dir_y    = f32(int(INPUT.keys[.E].pressing) - int(INPUT.keys[.Q].pressing))

        // It's a "direction" input so its length
        // should be no more than 1
        if linalg.dot(keyboard_dir_xz, keyboard_dir_xz) > 1 {
            keyboard_dir_xz = linalg.normalize(keyboard_dir_xz)
        }

        if abs(keyboard_dir_y) > 1 {
            keyboard_dir_y = math.sign(keyboard_dir_y)
        }
    }

    target_vel := keyboard_dir_xz * move_speed
    target_vel = linalg.quaternion_mul_vector3(cam_rot, target_vel)
    target_vel.y += keyboard_dir_y * move_speed

    cur_vel = approach_linear(cur_vel, target_vel, move_accel * DELTA_TIME)
    cam_pos += cur_vel * DELTA_TIME

    return world_to_view_mat(cam_pos, cam_rot)

    approach_linear :: proc(cur: [3]f32, target: [3]f32, delta: f32) -> [3]f32
    {
        diff := target - cur
        dist := linalg.length(diff)

        if dist <= delta do return target
        return cur + diff / dist * delta
    }
}

create_blas :: proc(using ctx: ^Vk_Ctx, positions: Buffer, indices: Buffer, verts_count: u32, idx_count: u32) -> Accel_Structure
{
    blas: Accel_Structure

    tri_data := vk.AccelerationStructureGeometryTrianglesDataKHR {
        sType = .ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        vertexFormat = .R32G32B32_SFLOAT,
        vertexData = {
            deviceAddress = get_buffer_device_address(ctx, positions)
        },
        vertexStride = size_of([3]f32),
        maxVertex = verts_count,
        indexType = .UINT32,
        indexData = {
            deviceAddress = get_buffer_device_address(ctx, indices)
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

    blas.buf = create_buffer(ctx, cast(int) size_info.accelerationStructureSize, 1, { .ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })

    // Create the scratch buffer for blas building
    scratch_buf := create_buffer(ctx, cast(int) size_info.buildScratchSize, 1, { .ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS, .STORAGE_BUFFER }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })

    // Build acceleration structure
    blas_ci := vk.AccelerationStructureCreateInfoKHR {
        sType = .ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        buffer = blas.buf.buf,
        size = size_info.accelerationStructureSize,
        type = .BOTTOM_LEVEL,
    }
    vk_check(vk.CreateAccelerationStructureKHR(device, &blas_ci, nil, &blas.handle))

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

    build_info.dstAccelerationStructure = blas.handle
    build_info.scratchData.deviceAddress = get_buffer_device_address(ctx, scratch_buf)
    vk.CmdBuildAccelerationStructuresKHR(cmd_buf, 1, &build_info, auto_cast raw_data(range_info_ptrs))

    barrier := vk.MemoryBarrier2 {
        sType = .MEMORY_BARRIER_2,
        srcStageMask = { .ACCELERATION_STRUCTURE_BUILD_KHR },
        srcAccessMask = { .ACCELERATION_STRUCTURE_WRITE_KHR },
        dstStageMask = { .ACCELERATION_STRUCTURE_BUILD_KHR },
        dstAccessMask = { .ACCELERATION_STRUCTURE_READ_KHR },
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

    // Get device address
    addr_info := vk.AccelerationStructureDeviceAddressInfoKHR {
        sType = .ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        accelerationStructure = blas.handle,
    }
    blas.addr = vk.GetAccelerationStructureDeviceAddressKHR(device, &addr_info)
    // fmt.println("get accel struct addr:", blas.addr)

    return blas
}

create_tlas :: proc(using ctx: ^Vk_Ctx, instances: []lm.Instance, meshes: []lm.Mesh) -> Tlas
{
    as: Accel_Structure

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
            instanceCustomIndex = u32(instance.mesh_idx),
            mask = 0xFF,
            instanceShaderBindingTableRecordOffset = 0,
            // NOTE: Unintuitive bindings! This cast is necessary!
            flags = auto_cast(vk.GeometryInstanceFlagsKHR { .TRIANGLE_FACING_CULL_DISABLE }),
            accelerationStructureReference = u64(meshes[instance.mesh_idx].blas.addr)
        }

        fmt.println(u32(vk_instance.flags))
    }

    instances_buf := create_instances_buffer(ctx, vk_instances)

    instances_data := vk.AccelerationStructureGeometryInstancesDataKHR {
        sType = .ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        arrayOfPointers = false,
        data = {
            deviceAddress = get_buffer_device_address(ctx, instances_buf)
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

    as.buf = create_buffer(ctx, cast(int) size_info.accelerationStructureSize, 1, { .ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })

    // Create the scratch buffer for tlas building
    scratch_buf := create_buffer(ctx, cast(int) size_info.buildScratchSize, 1, { .ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS, .STORAGE_BUFFER }, { .DEVICE_LOCAL }, { .DEVICE_ADDRESS })

    // Build acceleration structure
    blas_ci := vk.AccelerationStructureCreateInfoKHR {
        sType = .ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        buffer = as.buf.buf,
        size = size_info.accelerationStructureSize,
        type = .TOP_LEVEL,
    }
    vk_check(vk.CreateAccelerationStructureKHR(device, &blas_ci, nil, &as.handle))

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
    build_info.scratchData.deviceAddress = get_buffer_device_address(ctx, scratch_buf)
    vk.CmdBuildAccelerationStructuresKHR(cmd_buf, 1, &build_info, auto_cast raw_data(range_info_ptrs))

    barrier := vk.MemoryBarrier2 {
        sType = .MEMORY_BARRIER_2,
        srcStageMask = { .ACCELERATION_STRUCTURE_BUILD_KHR },
        srcAccessMask = { .ACCELERATION_STRUCTURE_WRITE_KHR },
        dstStageMask = { .RAY_TRACING_SHADER_KHR },
        dstAccessMask = { .ACCELERATION_STRUCTURE_READ_KHR },
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

Accel_Structure :: struct
{
    handle: vk.AccelerationStructureKHR,
    buf: Buffer,
    addr: vk.DeviceAddress
}

Tlas :: struct
{
    as: Accel_Structure,
    instances_buf: Buffer,
}

get_buffer_device_address :: proc(using ctx: ^Vk_Ctx, buffer: Buffer) -> vk.DeviceAddress
{
    info := vk.BufferDeviceAddressInfo {
        sType = .BUFFER_DEVICE_ADDRESS_INFO,
        buffer = buffer.buf
    }
    return vk.GetBufferDeviceAddress(device, &info)
}

v3_to_v4 :: proc(v: [3]f32, w: Maybe(f32) = nil) -> (res: [4]f32)
{
    res.xyz = v.xyz
    if num, ok := w.?; ok {
        res.w = num
    }
    return
}
