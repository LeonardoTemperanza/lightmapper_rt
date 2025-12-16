
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

import "core:log"
import "base:runtime"
import "core:math/linalg"
import "core:math"
import "core:sync"
import "core:c"
import os "core:os"
import sdl "vendor:sdl3"
import vk "vendor:vulkan"

import vku "../../vk_utils"
import lm "../../"
import loader "../loader"

Filter_Mode :: enum
{
    Point,  // For testing
    Bilinear,
    Bicubic
}

FILTER_MODE :: Filter_Mode.Bicubic
SYNCHRONOUS :: true

NUM_FRAMES_IN_FLIGHT :: 1
NUM_SWAPCHAIN_IMAGES :: 2
WINDOW_SIZE_X: u32
WINDOW_SIZE_Y: u32

vk_logger: log.Logger
glfw_logger: log.Logger

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

LIGHTMAP_SIZE :: 8 * 1024

main :: proc()
{
    if len(os.args) != 2
    {
        log.error("Incorrect args. Usage: example.exe <GLTF scene path>.")
        return
    }

    scene_path := os.args[1]

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
        .HIGH_PIXEL_DENSITY,
        .VULKAN,
        .FULLSCREEN,
    }
    window := sdl.CreateWindow("Lightmapper RT Example", 1920, 1080, window_flags)
    win_width, win_height: c.int
    assert(sdl.GetWindowSize(window, &win_width, &win_height))
    WINDOW_SIZE_X = auto_cast max(0, win_width)
    WINDOW_SIZE_Y = auto_cast max(0, win_height)
    ensure(window != nil)

    vk.load_proc_addresses_global(cast(rawptr) sdl.Vulkan_GetVkGetInstanceProcAddr())

    vk_ctx := lm.init_vk_context(window, vk_debug_callback)
    defer lm.destroy_vk_context(&vk_ctx)

    width, height: c.int
    sdl.GetWindowSizeInPixels(window, &width, &height)

    swapchain := create_swapchain(&vk_ctx, u32(width), u32(height))
    defer destroy_swapchain(&vk_ctx, swapchain)

    depth_image, depth_image_view := create_depth_texture(&vk_ctx, u32(width), u32(height))
    defer vk.DestroyImage(vk_ctx.device, depth_image, nil)

    shaders := create_shaders(&vk_ctx)
    defer destroy_shaders(&vk_ctx, shaders)

    upload_cmd_pool_ci := vk.CommandPoolCreateInfo {
        sType = .COMMAND_POOL_CREATE_INFO,
        queueFamilyIndex = vk_ctx.queue_family_idx,
        flags = { .TRANSIENT }
    }
    upload_cmd_pool: vk.CommandPool
    vk_check(vk.CreateCommandPool(vk_ctx.device, &upload_cmd_pool_ci, nil, &upload_cmd_pool))
    defer vk.DestroyCommandPool(vk_ctx.device, upload_cmd_pool, nil)

    lm_vk_ctx := lm.Lightmapper_Vulkan_Context {
        phys_device = vk_ctx.phys_device,
        device = vk_ctx.device,
        queue = vk_ctx.lm_queue,
        //queue = vk_ctx.queue,
        queue_family_idx = vk_ctx.queue_family_idx,
    }
    lm_ctx := lm.init_test(lm_vk_ctx)

    instances, textures, ok_l := loader.load_scene_gltf(&vk_ctx, &lm_ctx, upload_cmd_pool, scene_path, LIGHTMAP_SIZE, texels_per_world_unit = 60, min_instance_texels = 256, max_instance_texels = 2048)
    if !ok_l do log.error("Failed to load scene %v", scene_path)

    vk_frames := create_vk_frames(&vk_ctx)
    frame_idx := u32(0)

    now_ts := sdl.GetPerformanceCounter()
    max_delta_time: f32 = 1.0 / 10.0  // 10fps

    desc_pool_ci := vk.DescriptorPoolCreateInfo {
        sType = .DESCRIPTOR_POOL_CREATE_INFO,
        flags = { .FREE_DESCRIPTOR_SET },
        maxSets = 2000,
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
                descriptorCount = 500,
            },
            {
                type = .COMBINED_IMAGE_SAMPLER,
                descriptorCount = 500,
            },
            {
                type = .STORAGE_BUFFER,
                descriptorCount = 10,
            }
        })
    }
    desc_pool: vk.DescriptorPool
    vk_check(vk.CreateDescriptorPool(vk_ctx.device, &desc_pool_ci, nil, &desc_pool))

    // Create linear sampler
    lightmap_sampler_ci := vk.SamplerCreateInfo {
        sType = .SAMPLER_CREATE_INFO,
        magFilter = .NEAREST if FILTER_MODE == .Point else .LINEAR,
        minFilter = .NEAREST if FILTER_MODE == .Point else .LINEAR,
        mipmapMode = .LINEAR,
        addressModeU = .REPEAT,
        addressModeV = .REPEAT,
        addressModeW = .REPEAT,
    }
    lightmap_sampler: vk.Sampler
    vk_check(vk.CreateSampler(vk_ctx.device, &lightmap_sampler_ci, nil, &lightmap_sampler))

    linear_sampler_ci := vk.SamplerCreateInfo {
        sType = .SAMPLER_CREATE_INFO,
        magFilter = .LINEAR,
        minFilter = .LINEAR,
        mipmapMode = .LINEAR,
        addressModeU = .REPEAT,
        addressModeV = .REPEAT,
        addressModeW = .REPEAT,
    }
    linear_sampler: vk.Sampler
    vk_check(vk.CreateSampler(vk_ctx.device, &linear_sampler_ci, nil, &linear_sampler))

    // Create material descriptor sets
    desc_sets: map[lm.Texture_Handle]vk.DescriptorSet
    for tex in textures
    {
        desc_set_ai := vk.DescriptorSetAllocateInfo {
            sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = desc_pool,
            descriptorSetCount = 1,
            pSetLayouts = raw_data([]vk.DescriptorSetLayout { shaders.mat_desc_set_layout })
        }
        desc_set: vk.DescriptorSet
        vk_check(vk.AllocateDescriptorSets(vk_ctx.device, &desc_set_ai, &desc_set))

        tex_handle := lm.get_texture(&lm_ctx, tex)

        writes := []vk.WriteDescriptorSet {
            {
                sType = .WRITE_DESCRIPTOR_SET,
                dstSet = desc_set,
                dstBinding = 0,
                descriptorCount = 1,
                descriptorType = .COMBINED_IMAGE_SAMPLER,
                pImageInfo = raw_data([]vk.DescriptorImageInfo {
                    {
                        imageView = tex_handle.view,
                        imageLayout = tex_handle.layout,
                        sampler = linear_sampler,
                    }
                })
            }
        }
        vk.UpdateDescriptorSets(vk_ctx.device, u32(len(writes)), raw_data(writes), 0, nil)

        desc_sets[tex] = desc_set
    }

    dir_light := lm.Dir_Light {
        angle = math.RAD_PER_DEG * 0.2,
        dir = linalg.normalize([3]f32 { 0.2, -1.0, -0.2 }),
        emission = [3]f32 { 200000.0, 184000.0, 164000.0 },
    }
    bake := lm.start_bake(&lm_ctx, instances[:], true, dir_light, LIGHTMAP_SIZE, 1000, 1)

    // Create main render target
    render_target: Image
    {
        cmd_buf := vku.begin_tmp_cmd_buf(vk_ctx.device, upload_cmd_pool)
        defer vku.end_tmp_cmd_buf(vk_ctx.device, upload_cmd_pool, vk_ctx.queue, cmd_buf)

        render_target = vku.create_image(vk_ctx.device, vk_ctx.phys_device, cmd_buf, {
            sType = .IMAGE_CREATE_INFO,
            flags = {},
            imageType = .D2,
            format = .R16G16B16A16_SFLOAT,
            extent = {
                width = WINDOW_SIZE_X,
                height = WINDOW_SIZE_Y,
                depth = 1,
            },
            mipLevels = 1,
            arrayLayers = 1,
            samples = { ._1 },
            usage = { .COLOR_ATTACHMENT, .SAMPLED },
            sharingMode = .EXCLUSIVE,
            queueFamilyIndexCount = 1,
            pQueueFamilyIndices = &vk_ctx.queue_family_idx,
            initialLayout = .UNDEFINED,
        })
    }

    // Create lightmap desc set
    desc_set_ai := vk.DescriptorSetAllocateInfo {
        sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool = desc_pool,
        descriptorSetCount = 1,
        pSetLayouts = raw_data([]vk.DescriptorSetLayout { shaders.lm_desc_set_layout })
    }

    lm_desc_set: vk.DescriptorSet
    vk_check(vk.AllocateDescriptorSets(vk_ctx.device, &desc_set_ai, &lm_desc_set))
    tonemap_desc_set: vk.DescriptorSet
    vk_check(vk.AllocateDescriptorSets(vk_ctx.device, &desc_set_ai, &tonemap_desc_set))

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
    vk.UpdateDescriptorSets(vk_ctx.device, u32(len(writes)), raw_data(writes), 0, nil)

    // Update tonemap desc set
    writes_2 := []vk.WriteDescriptorSet {
        {
            sType = .WRITE_DESCRIPTOR_SET,
            dstSet = tonemap_desc_set,
            dstBinding = 0,
            descriptorCount = 1,
            descriptorType = .COMBINED_IMAGE_SAMPLER,
            pImageInfo = raw_data([]vk.DescriptorImageInfo {
                {
                    imageView = render_target.view,
                    imageLayout = .GENERAL,
                    sampler = lightmap_sampler,
                }
            })
        }
    }
    vk.UpdateDescriptorSets(vk_ctx.device, u32(len(writes_2)), raw_data(writes_2), 0, nil)

    for
    {
        when SYNCHRONOUS
        {
            sync.mutex_lock(bake.debug_mutex1)
            defer { sync.mutex_unlock(bake.debug_mutex0) }
        }

        // fmt.println("frame")

        proceed := handle_window_events(window)
        if !proceed do break

        last_ts := now_ts
        now_ts = sdl.GetPerformanceCounter()
        DELTA_TIME = min(max_delta_time, f32(f64((now_ts - last_ts)*1000) / f64(ts_freq)) / 1000.0)

        vk_frame := vk_frames[frame_idx]
        vk_check(vk.WaitForFences(vk_ctx.device, 1, &vk_frame.fence, true, max(u64)))
        vk_check(vk.ResetFences(vk_ctx.device, 1, &vk_frame.fence))

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

        render_scene(&vk_ctx, cmd_buf, depth_image_view, render_target.view, lm_desc_set, shaders, swapchain, &lm_ctx, instances[:], &desc_sets, auto_cast width, auto_cast height)
        vku.image_barrier_safe_slow(&render_target, cmd_buf, .GENERAL)
        tonemap_image(&vk_ctx, cmd_buf, shaders, tonemap_desc_set, render_target, swapchain.image_views[image_idx])

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

    vk_check(vk.QueueWaitIdle(vk_ctx.queue))
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

create_swapchain :: proc(using ctx: ^lm.App_Vulkan_Context, width: u32, height: u32) -> Swapchain
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

destroy_swapchain :: proc(using ctx: ^lm.App_Vulkan_Context, swapchain: Swapchain)
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

create_vk_frames :: proc(using ctx: ^lm.App_Vulkan_Context) -> [NUM_FRAMES_IN_FLIGHT]Vulkan_Per_Frame
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

destroy_vk_frames :: proc(using ctx: ^lm.App_Vulkan_Context, frames: [NUM_FRAMES_IN_FLIGHT]Vulkan_Per_Frame)
{
    for frame in frames
    {
        vk.DestroyCommandPool(device, frame.cmd_pool, nil)
        vk.DestroySemaphore(device, frame.acquire_semaphore, nil)
        vk.DestroyFence(device, frame.fence, nil)
    }
}

create_shaders :: proc(using ctx: ^lm.App_Vulkan_Context) -> Shaders
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
        mat_desc_set_layout_ci := vk.DescriptorSetLayoutCreateInfo {
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
        vk_check(vk.CreateDescriptorSetLayout(device, &mat_desc_set_layout_ci, nil, &res.mat_desc_set_layout))

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
        pipeline_set_layouts := []vk.DescriptorSetLayout {
            res.lm_desc_set_layout,
            res.mat_desc_set_layout,
        }

        pipeline_layout_ci := vk.PipelineLayoutCreateInfo {
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            pushConstantRangeCount = u32(len(push_constant_ranges)),
            pPushConstantRanges = raw_data(push_constant_ranges),
            setLayoutCount = u32(len(pipeline_set_layouts)),
            pSetLayouts = raw_data(pipeline_set_layouts),
        }
        vk_check(vk.CreatePipelineLayout(device, &pipeline_layout_ci, nil, &res.pipeline_layout))
    }

    // NOTE: Not using context.temp_allocator because it doesn't guarantee 4 byte alignment,
    // and vulkan requires the alignment of the spirv to be 4 byte.
    model_to_proj_vert := load_file("examples/shaders/model_to_proj.vert.spv", context.allocator)
    defer delete(model_to_proj_vert)
    lit_frag := load_file("examples/shaders/lit.frag.spv", context.allocator)
    defer delete(lit_frag)
    tonemap_vert := load_file("examples/shaders/tonemap.vert.spv", context.allocator)
    defer delete(tonemap_vert)
    tonemap_frag := load_file("examples/shaders/tonemap.frag.spv", context.allocator)
    defer delete(tonemap_frag)

    // Create shader objects.
    {
        lit_layouts := []vk.DescriptorSetLayout {
            res.lm_desc_set_layout,
            res.mat_desc_set_layout,
        }

        shader_cis := [?]vk.ShaderCreateInfoEXT {
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
                setLayoutCount = u32(len(lit_layouts)),
                pSetLayouts = raw_data(lit_layouts)
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
                setLayoutCount = u32(len(lit_layouts)),
                pSetLayouts = raw_data(lit_layouts)
            },
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(tonemap_vert),
                pCode = raw_data(tonemap_vert),
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
                codeSize = len(tonemap_frag),
                pCode = raw_data(tonemap_frag),
                pName = "main",
                stage = { .FRAGMENT },
                nextStage = {},
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
                setLayoutCount = 1,
                pSetLayouts = &res.lm_desc_set_layout
            },
        }
        shaders: [len(shader_cis)]vk.ShaderEXT
        vk_check(vk.CreateShadersEXT(device, len(shaders), raw_data(&shader_cis), nil, raw_data(&shaders)))

        res.model_to_proj = shaders[0]
        res.lit = shaders[1]
        res.tonemap_vert = shaders[2]
        res.tonemap_frag = shaders[3]
    }

    return res
}

Shaders :: struct
{
    pipeline_layout: vk.PipelineLayout,

    model_to_proj: vk.ShaderEXT,
    lit: vk.ShaderEXT,
    tonemap_vert: vk.ShaderEXT,
    tonemap_frag: vk.ShaderEXT,

    // Desc set layouts
    mat_desc_set_layout: vk.DescriptorSetLayout,
    lm_desc_set_layout: vk.DescriptorSetLayout,
}

destroy_shaders :: proc(using ctx: ^lm.App_Vulkan_Context, shaders: Shaders)
{
    vk.DestroyPipelineLayout(device, shaders.pipeline_layout, nil)
    vk.DestroyShaderEXT(device, shaders.model_to_proj, nil)
    vk.DestroyShaderEXT(device, shaders.lit, nil)
    vk.DestroyShaderEXT(device, shaders.tonemap_vert, nil)
    vk.DestroyShaderEXT(device, shaders.tonemap_frag, nil)
}

render_scene :: proc(using ctx: ^lm.App_Vulkan_Context, cmd_buf: vk.CommandBuffer, depth_view: vk.ImageView, color_view: vk.ImageView, lightmap_desc_set: vk.DescriptorSet, shaders: Shaders, swapchain: Swapchain, lm_ctx: ^lm.Context, instances: []lm.Instance, mat_descs: ^map[lm.Texture_Handle]vk.DescriptorSet, width: u32, height: u32)
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
        {  // UVs
            sType = .VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            binding = 3,
            stride = size_of([2]f32),
            inputRate = .VERTEX,
            divisor = 1,
        }
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
        {
            sType = .VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            location = 3,
            binding = 3,
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

    render_viewport_aspect_ratio := f32(width) / f32(height)

    world_to_view := compute_world_to_view()
    view_to_proj := linalg.matrix4_perspective_f32(math.RAD_PER_DEG * 59.0, render_viewport_aspect_ratio, 0.1, 1000.0, false)

    for instance in instances
    {
        mesh := lm.get_mesh(lm_ctx, instance.mesh)

        offset := vk.DeviceSize(0)
        vk.CmdBindVertexBuffers(cmd_buf, 0, 1, &mesh.pos.handle, &offset)
        vk.CmdBindVertexBuffers(cmd_buf, 1, 1, &mesh.normals.handle, &offset)
        vk.CmdBindVertexBuffers(cmd_buf, 2, 1, &mesh.lm_uvs.handle, &offset)
        vk.CmdBindVertexBuffers(cmd_buf, 3, 1, &mesh.uvs.handle, &offset)
        vk.CmdBindIndexBuffer(cmd_buf, mesh.indices.handle, 0, .UINT32)

        tex := lm.get_texture(lm_ctx, instance.albedo_tex)
        if tex != nil {
            vk.CmdBindDescriptorSets(cmd_buf, .GRAPHICS, shaders.pipeline_layout, 1, 1, &mat_descs[instance.albedo_tex], 0, nil)
        }

        Push :: struct {
            model_to_world: matrix[4, 4]f32,
            normal_mat: matrix[4, 4]f32,
            world_to_proj: matrix[4, 4]f32,
            lm_uv_offset: [2]f32,
            lm_uv_scale: f32,
            is_bicubic: b32,
        }
        push := Push {
            model_to_world = instance.transform,
            normal_mat = linalg.transpose(linalg.inverse(instance.transform)),
            world_to_proj = view_to_proj * world_to_view,
            lm_uv_offset = instance.lm_offset,
            lm_uv_scale = instance.lm_scale,
            is_bicubic = FILTER_MODE == .Bicubic
        }
        vk.CmdPushConstants(cmd_buf, shaders.pipeline_layout, { .VERTEX, .FRAGMENT }, 0, size_of(push), &push)

        vk.CmdDrawIndexed(cmd_buf, u32(len(mesh.indices_cpu)), 1, 0, 0, 0)
    }

    vk.CmdEndRendering(cmd_buf)
}

tonemap_image :: proc(using ctx: ^lm.App_Vulkan_Context, cmd_buf: vk.CommandBuffer, shaders: Shaders, tonemap_desc_set: vk.DescriptorSet, src: Image, dst_view: vk.ImageView)
{
    color_attachment := vk.RenderingAttachmentInfo {
        sType = .RENDERING_ATTACHMENT_INFO,
        imageView = dst_view,
        imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
        loadOp = .CLEAR,
        storeOp = .STORE,
        clearValue = {
            color = { float32 = { 0.0, 0.0, 0.0, 1.0 } }
        }
    }
    rendering_info := vk.RenderingInfo {
        sType = .RENDERING_INFO,
        renderArea = {
            offset = { 0, 0 },
            extent = { src.width, src.height }
        },
        layerCount = 1,
        colorAttachmentCount = 1,
        pColorAttachments = &color_attachment,
        pDepthAttachment = nil,
    }
    vk.CmdBeginRendering(cmd_buf, &rendering_info)

    vk.CmdSetCullMode(cmd_buf, nil)
    vk.CmdSetDepthTestEnable(cmd_buf, false)
    vk.CmdSetDepthWriteEnable(cmd_buf, false)

    tmp := tonemap_desc_set
    vk.CmdBindDescriptorSets(cmd_buf, .GRAPHICS, shaders.pipeline_layout, 0, 1, &tmp, 0, nil)

    color_mask := vk.ColorComponentFlags { .R, .G, .B, .A }
    vk.CmdSetColorWriteMaskEXT(cmd_buf, 0, 1, &color_mask)

    shader_stages := []vk.ShaderStageFlags { { .VERTEX }, { .GEOMETRY }, { .FRAGMENT } }
    to_bind := []vk.ShaderEXT { shaders.tonemap_vert, vk.ShaderEXT(0), shaders.tonemap_frag }
    assert(len(shader_stages) == len(to_bind))
    vk.CmdBindShadersEXT(cmd_buf, u32(len(shader_stages)), raw_data(shader_stages), raw_data(to_bind))

    vk.CmdDraw(cmd_buf, 6, 1, 0, 0)

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

xform_to_mat :: proc(pos: [3]f64, rot: quaternion256, scale: [3]f64) -> matrix[4, 4]f32
{
    return cast(matrix[4, 4]f32) (#force_inline linalg.matrix4_translate(pos) *
           #force_inline linalg.matrix4_from_quaternion(rot) *
           #force_inline linalg.matrix4_scale(scale))
}

Buffer :: vku.Buffer
Image :: vku.Image

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

find_depth_format :: proc(using ctx: ^lm.App_Vulkan_Context) -> vk.Format
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

create_depth_texture :: proc(using ctx: ^lm.App_Vulkan_Context, width, height: u32) -> (vk.Image, vk.ImageView)
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
        memoryTypeIndex = vku.find_mem_type(phys_device, mem_requirements.memoryTypeBits, { })
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
    @(static) cam_pos: [3]f32 = { 0, 2.5, 0.0 }

    @(static) angle: [2]f32

    /*
    min_t := f32(25.0)
    max_t := f32(35.0)
    @(static) t := f32(0.0)
    t = min(t + DELTA_TIME, max_t)
    */

    /*
    first_pos := [3]f32 { 2.2657561, 1.3119615, -1.8265065 }
    second_pos := [3]f32 { -0.96947265, 1.1673785, -0.88610756 }
    first_angle := [2]f32 { 6.1121435, -0.1221731 }
    second_angle := [2]f32 { 5.4768438, -0.041887973 }

    ease_in_out_cubic :: proc(t: f32) -> f32
    {
        return 4.0 * t * t * t if t < 0.5 else 1.0 - math.pow(-2.0 * t + 2.0, 3.0) / 2.0;
    }
    */

    //cam_pos = math.lerp(first_pos, second_pos, ease_in_out_cubic(max(0.0, t - min_t) / (max_t - min_t)))
    //angle = math.lerp(first_angle, second_angle, ease_in_out_cubic(max(0.0, t - min_t) / (max_t - min_t)))

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

v3_to_v4 :: proc(v: [3]f32, w: Maybe(f32) = nil) -> (res: [4]f32)
{
    res.xyz = v.xyz
    if num, ok := w.?; ok {
        res.w = num
    }
    return
}
