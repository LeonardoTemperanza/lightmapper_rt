
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
import "core:math/linalg"
import intr "base:intrinsics"
import "core:slice"
import "core:mem"
import "core:log"
import "base:runtime"
import thr "core:thread"

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

// Initializes this library with its own Vulkan context.
// Only really makes sense for projects not using Vulkan.
init_from_scratch :: proc() -> Context
{
    return {}
}

// Initializes this library with an already existing Vulkan context.
// NOTE: This library is completely asynchronous, so use a separate queue here!
init_from_vulkan_context :: proc(phys_device: vk.PhysicalDevice, device: vk.Device, queue: vk.Queue, queue_family_idx: u32) -> Context
{
    res: Context
    res.vk_ctx.phys_device = phys_device
    res.vk_ctx.device = device
    res.vk_ctx.queue = queue
    res.vk_ctx.queue_family_idx = queue_family_idx

    res.shaders = create_shaders(&res.vk_ctx)

    return res
}

init_test :: proc(vk_ctx: Vulkan_Context) -> Context
{
    res: Context
    res.vk_ctx = vk_ctx

    res.shaders = create_shaders(&res.vk_ctx)

    return res
}

// Scene description

Scene :: struct
{
    instances: [dynamic]Instance,
    meshes: [dynamic]Mesh,
    tlas: Tlas,
}

/*
Scene :: union
{
    Scene_Vulkan,
    Scene_CPU
}
*/

/*
Scene_Vulkan :: struct
{
    instances:
}
*/

Instance :: struct
{
    transform: matrix[4, 4]f32,
    mesh_idx: u32,
    lm_idx: u32,
    lm_offset: [2]f32,
    lm_scale: f32,
}

Mesh :: struct
{
    indices_cpu: [dynamic]u32,
    pos_cpu: [dynamic][3]f32,

    pos: Buffer,
    normals: Buffer,
    lm_uvs: Buffer,
    indices_gpu: Buffer,
    idx_count: u32,

    lm_uvs_present: bool,

    blas: Accel_Structure,
}

Tlas :: struct
{
    as: Accel_Structure,
    instances_buf: Buffer,
}

Accel_Structure :: struct
{
    handle: vk.AccelerationStructureKHR,
    buf: Buffer,
    addr: vk.DeviceAddress
}

Scene_Structures :: struct
{

}

build_scene_structures :: proc(scene: Scene) -> Scene_Structures
{
    return {}
}

// update_scene_structures

// Baking

Bake :: struct
{
    using ctx: ^Context,
    thread: ^thr.Thread,

    // Settings
    num_accums: u32,
    lightmap_size: u32,
    scene: Scene,

    acquired_view: bool,
}

// Starts a baking process in a separate thread.
//
// Inputs:
// - lightmap_size: Size in pixels of the lightmap to be built.
// - num_accums: Number of accumulations for pathtracing.
// - num_samples_per_pixel: Number of pathtrace samples per pixel done on each accumulation.
start_bake :: proc(using ctx: ^Context, scene: Scene, scene_structures: Scene_Structures,
                   lightmap_size: u32 = 4096, num_accums: u32 = 200, num_samples_per_pixel: u32 = 1,
                   ) -> ^Bake
{
    bake := new(Bake)
    bake.ctx = ctx

    bake.num_accums = num_accums
    bake.lightmap_size = lightmap_size

    bake.thread = thr.create(bake_thread)
    bake.thread.init_context = context
    bake.thread.user_index = 0
    bake.thread.data = bake
    bake.scene = scene

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
    free(bake)
}

acquire_lightmap_view_vk :: proc(using bake: ^Bake) -> vk.ImageView
{
    return vk.ImageView(0)
}

release_lightmap_view_vk :: proc(using bake: ^Bake)
{

}

// Internals

bake_thread :: proc(t: ^thr.Thread)
{
    ctx := cast(^Bake) t.data
    bake_main(ctx)
}

bake_main :: proc(using bake: ^Bake)
{
    // Setup
    // Create queue and command buffer here!
    cmd_pool_ci := vk.CommandPoolCreateInfo {
        sType = .COMMAND_POOL_CREATE_INFO,
        queueFamilyIndex = queue_family_idx,
        flags = { .TRANSIENT }
    }
    cmd_pool: vk.CommandPool
    vk_check(vk.CreateCommandPool(device, &cmd_pool_ci, nil, &cmd_pool))

    cmd_buf_ai := vk.CommandBufferAllocateInfo {
        sType = .COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool = cmd_pool,
        level = .PRIMARY,
        commandBufferCount = 1,
    }
    cmd_buf: vk.CommandBuffer
    vk_check(vk.AllocateCommandBuffers(device, &cmd_buf_ai, &cmd_buf))

    semaphore_ci := vk.SemaphoreCreateInfo { sType = .SEMAPHORE_CREATE_INFO }
    acquire_semaphore: vk.Semaphore
    vk_check(vk.CreateSemaphore(device, &semaphore_ci, nil, &acquire_semaphore))

    fence_ci := vk.FenceCreateInfo {
        sType = .FENCE_CREATE_INFO,
        flags = { .SIGNALED },
    }
    fence: vk.Fence
    vk_check(vk.CreateFence(device, &fence_ci, nil, &fence))

    gbufs := create_gbuffers(bake)

    lightmap := create_image(ctx, {
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
        usage = { .STORAGE, .SAMPLED },
        sharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &vk_ctx.queue_family_idx,
        initialLayout = .UNDEFINED,
    }, "lightmap")

    // RT Descriptor set.
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

    desc_set_ai := vk.DescriptorSetAllocateInfo {
        sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool = desc_pool,
        descriptorSetCount = 1,
        pSetLayouts = raw_data([]vk.DescriptorSetLayout { shaders.rt_desc_set_layout })
    }

    rt_desc_set: vk.DescriptorSet
    vk_check(vk.AllocateDescriptorSets(device, &desc_set_ai, &rt_desc_set))

    update_rt_desc_set(device, rt_desc_set, scene.tlas.as.handle, lightmap, gbufs)

    vk_check(vk.BeginCommandBuffer(cmd_buf, &{
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        flags = { .ONE_TIME_SUBMIT },
    }))

    // Lightmap backbuffer
    lightmap_backbuffer := create_image(ctx, {
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
        usage = { .STORAGE, .SAMPLED },
        sharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &vk_ctx.queue_family_idx,
        initialLayout = .UNDEFINED,
    }, "lightmap_backbuffer")

    // GBuffers
    render_gbuffers(bake, cmd_buf, gbufs)

    vk_check(vk.EndCommandBuffer(cmd_buf))

    // Pathtracing
    for iter in 0..<num_accums
    {
        vk_check(vk.WaitForFences(vk_ctx.device, 1, &fence, true, max(u64)))
        vk_check(vk.ResetFences(vk_ctx.device, 1, &fence))
        vk_check(vk.ResetCommandPool(vk_ctx.device, cmd_pool, {}))

        // If signalled to stop, stop.

        vk_check(vk.BeginCommandBuffer(cmd_buf, &{
            sType = .COMMAND_BUFFER_BEGIN_INFO,
            flags = { .ONE_TIME_SUBMIT },
        }))

        // GBuffers barriers
        {
            gbufs_barriers := []vk.ImageMemoryBarrier2 {
                {
                    sType = .IMAGE_MEMORY_BARRIER_2,
                    image = gbufs.world_pos.img,
                    subresourceRange = {
                        aspectMask = { .COLOR },
                        levelCount = 1,
                        layerCount = 1,
                    },
                    oldLayout = .UNDEFINED,
                    newLayout = .GENERAL,
                    srcStageMask = { .COLOR_ATTACHMENT_OUTPUT },
                    srcAccessMask = { .COLOR_ATTACHMENT_WRITE },
                    dstStageMask = { .RAY_TRACING_SHADER_KHR },
                    dstAccessMask = { .SHADER_READ },
                },
                {
                    sType = .IMAGE_MEMORY_BARRIER_2,
                    image = gbufs.world_normals.img,
                    subresourceRange = {
                        aspectMask = { .COLOR },
                        levelCount = 1,
                        layerCount = 1,
                    },
                    oldLayout = .UNDEFINED,
                    newLayout = .GENERAL,
                    srcStageMask = { .COLOR_ATTACHMENT_OUTPUT },
                    srcAccessMask = { .COLOR_ATTACHMENT_WRITE },
                    dstStageMask = { .RAY_TRACING_SHADER_KHR },
                    dstAccessMask = { .SHADER_READ },
                },
            }
            vk.CmdPipelineBarrier2(cmd_buf, &{
                sType = .DEPENDENCY_INFO,
                imageMemoryBarrierCount = u32(len(gbufs_barriers)),
                pImageMemoryBarriers = raw_data(gbufs_barriers),
            })
        }

        // LIGHTMAP BARRIER
        {
            lightmap_barrier := []vk.ImageMemoryBarrier2 {
                {
                    sType = .IMAGE_MEMORY_BARRIER_2,
                    image = lightmap.img,
                    subresourceRange = {
                        aspectMask = { .COLOR },
                        levelCount = 1,
                        layerCount = 1,
                    },
                    oldLayout = .UNDEFINED,
                    newLayout = .GENERAL,
                    srcStageMask = { .FRAGMENT_SHADER },
                    srcAccessMask = { .SHADER_SAMPLED_READ },
                    dstStageMask = { .RAY_TRACING_SHADER_KHR },
                    dstAccessMask = { .SHADER_STORAGE_WRITE, .SHADER_STORAGE_READ },
                },
            }
            vk.CmdPipelineBarrier2(cmd_buf, &{
                sType = .DEPENDENCY_INFO,
                imageMemoryBarrierCount = u32(len(lightmap_barrier)),
                pImageMemoryBarriers = raw_data(lightmap_barrier),
            })
        }

        pathtrace_iter(bake, cmd_buf, rt_desc_set, iter)

        // Lightmap barrier
        {
            lightmap_barrier := []vk.ImageMemoryBarrier2 {
                {
                    sType = .IMAGE_MEMORY_BARRIER_2,
                    image = lightmap.img,
                    subresourceRange = {
                        aspectMask = { .COLOR },
                        levelCount = 1,
                        layerCount = 1,
                    },
                    oldLayout = .GENERAL,
                    newLayout = .GENERAL,
                    srcStageMask = { .RAY_TRACING_SHADER_KHR },
                    srcAccessMask = { .SHADER_STORAGE_WRITE },
                    dstStageMask = { .FRAGMENT_SHADER },
                    dstAccessMask = { .SHADER_SAMPLED_READ },
                },
            }
            vk.CmdPipelineBarrier2(cmd_buf, &{
                sType = .DEPENDENCY_INFO,
                imageMemoryBarrierCount = u32(len(lightmap_barrier)),
                pImageMemoryBarriers = raw_data(lightmap_barrier),
            })
        }

        vk_check(vk.EndCommandBuffer(cmd_buf))

        wait_stage_flags := vk.PipelineStageFlags { .COLOR_ATTACHMENT_OUTPUT }
        submit_info := vk.SubmitInfo {
            sType = .SUBMIT_INFO,
            //waitSemaphoreCount = 1,
            //pWaitSemaphores = &vk_frame.acquire_semaphore,
            pWaitDstStageMask = &wait_stage_flags,
            commandBufferCount = 1,
            pCommandBuffers = &cmd_buf,
            //signalSemaphoreCount = 1,
            //pSignalSemaphores = &present_semaphore,
        }
        vk_check(vk.QueueSubmit(vk_ctx.queue, 1, &submit_info, fence))

        // Submit results to backbuffer, if available.
        {
            
        }
    }

    // Cleanup
}

render_gbuffers :: proc(using bake: ^Bake, cmd_buf: vk.CommandBuffer, gbuffers: GBuffers)
{
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

    // World pos
    {
        color_attachment := vk.RenderingAttachmentInfo {
            sType = .RENDERING_ATTACHMENT_INFO,
            imageView = gbuffers.world_pos.view,
            imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
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

        shader_stages := [2]vk.ShaderStageFlags { { .VERTEX }, { .FRAGMENT } }
        to_bind := [2]vk.ShaderEXT { shaders.uv_space, shaders.gbuffer_world_pos }
        vk.CmdBindShadersEXT(cmd_buf, 2, &shader_stages[0], &to_bind[0] )

        draw_gbuffer(bake, cmd_buf, shaders.pipeline_layout)
    }

    // World normal
    {
        color_attachment := vk.RenderingAttachmentInfo {
            sType = .RENDERING_ATTACHMENT_INFO,
            imageView = gbuffers.world_normals.view,
            imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
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

        shader_stages := [2]vk.ShaderStageFlags { { .VERTEX }, { .FRAGMENT } }
        to_bind := [2]vk.ShaderEXT { shaders.uv_space, shaders.gbuffer_world_normals }
        vk.CmdBindShadersEXT(cmd_buf, 2, &shader_stages[0], &to_bind[0] )

        draw_gbuffer(bake, cmd_buf, shaders.pipeline_layout)
    }
}

draw_gbuffer :: proc(using bake: ^Bake, cmd_buf: vk.CommandBuffer, pipeline_layout: vk.PipelineLayout)
{
    vk.CmdSetExtraPrimitiveOverestimationSizeEXT(cmd_buf, 0.0)
    vk.CmdSetConservativeRasterizationModeEXT(cmd_buf, .OVERESTIMATE)
    vk.CmdSetRasterizationSamplesEXT(cmd_buf, { ._1 })
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

    for instance, i in scene.instances
    {
        mesh := scene.meshes[instance.mesh_idx]

        if !mesh.lm_uvs_present { continue }

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
        vk.CmdBindVertexBuffers(cmd_buf, 1, 1, &mesh.normals.handle, &offset)
        if mesh.lm_uvs_present {
            vk.CmdBindVertexBuffers(cmd_buf, 2, 1, &mesh.lm_uvs.handle, &offset)
        }
        vk.CmdBindIndexBuffer(cmd_buf, mesh.indices_gpu.handle, 0, .UINT32)

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

        vk.CmdDrawIndexed(cmd_buf, mesh.idx_count, 1, 0, 0, 0)
    }
}

pathtrace_iter :: proc(using bake: ^Bake, cmd_buf: vk.CommandBuffer, rt_desc_set: vk.DescriptorSet, accum_counter: u32)
{
    vk.CmdBindPipeline(cmd_buf, .RAY_TRACING_KHR, shaders.rt_pipeline)

    tmp := rt_desc_set
    vk.CmdBindDescriptorSets(cmd_buf, .RAY_TRACING_KHR, shaders.rt_pipeline_layout, 0, 1, &tmp, 0, nil)

    sbt_addr := u64(get_buffer_device_address(device, shaders.sbt_buf))

    raygen_region := vk.StridedDeviceAddressRegionKHR {
        deviceAddress = vk.DeviceAddress(sbt_addr + 0 * u64(rt_base_align)),
        stride = vk.DeviceSize(rt_handle_alignment),
        size = vk.DeviceSize(rt_handle_alignment),
    }
    raymiss_region := vk.StridedDeviceAddressRegionKHR {
        deviceAddress = vk.DeviceAddress(sbt_addr + 1 * u64(rt_base_align)),
        stride = vk.DeviceSize(rt_handle_alignment),
        size = vk.DeviceSize(rt_handle_alignment),
    }
    rayhit_region := vk.StridedDeviceAddressRegionKHR {
        deviceAddress = vk.DeviceAddress(sbt_addr + 2 * u64(rt_base_align)),
        stride = vk.DeviceSize(rt_handle_alignment),
        size = vk.DeviceSize(rt_handle_alignment),
    }
    callable_region := vk.StridedDeviceAddressRegionKHR {}

    Push :: struct {
        accum_counter: u32,
        seed: u32,
    }
    push := Push {
        accum_counter = accum_counter,
        seed = 0,
    }
    vk.CmdPushConstants(cmd_buf, shaders.rt_pipeline_layout, { .RAYGEN_KHR }, 0, size_of(push), &push)

    vk.CmdTraceRaysKHR(cmd_buf, &raygen_region, &raymiss_region, &rayhit_region, &callable_region, lightmap_size, lightmap_size, 1)
}

Shaders :: struct
{
    // GBuffer generation
    lm_desc_set_layout: vk.DescriptorSetLayout,
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
        pipeline_layout_ci := vk.PipelineLayoutCreateInfo {
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            pushConstantRangeCount = u32(len(push_constant_ranges)),
            pPushConstantRanges = raw_data(push_constant_ranges),
            setLayoutCount = 1,
            pSetLayouts = &res.lm_desc_set_layout,
        }
        vk_check(vk.CreatePipelineLayout(device, &pipeline_layout_ci, nil, &res.pipeline_layout))

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

    uv_space_vert_code              := #load("shaders/uv_space.vert.spv")
    gbuffer_world_pos_frag_code     := #load("shaders/gbuffer_world_pos.frag.spv")
    gbuffer_world_normals_frag_code := #load("shaders/gbuffer_world_normals.frag.spv")
    raygen_code                     := #load("shaders/raygen.rgen.spv")
    raymiss_code                    := #load("shaders/raymiss.rmiss.spv")
    rayhit_code                     := #load("shaders/rayhit.rchit.spv")

    // NOTE: #load(<string-path>, <type>) produces unaligned reads and writes (https://github.com/odin-lang/Odin/issues/5771)
    uv_space_vert_code_aligned: []byte = mem.make_aligned([]byte, len(uv_space_vert_code), 4, context.temp_allocator) or_else panic("failed to align mesh shader bytecode")
    mem.copy(raw_data(uv_space_vert_code_aligned), raw_data(uv_space_vert_code), len(uv_space_vert_code))
    gbuffer_world_pos_frag_code_aligned: []byte = mem.make_aligned([]byte, len(gbuffer_world_pos_frag_code), 4, context.temp_allocator) or_else panic("failed to align mesh shader bytecode")
    mem.copy(raw_data(gbuffer_world_pos_frag_code_aligned), raw_data(gbuffer_world_pos_frag_code), len(gbuffer_world_pos_frag_code))
    gbuffer_world_normals_frag_code_aligned: []byte = mem.make_aligned([]byte, len(gbuffer_world_normals_frag_code), 4, context.temp_allocator) or_else panic("failed to align mesh shader bytecode")
    mem.copy(raw_data(gbuffer_world_normals_frag_code_aligned), raw_data(gbuffer_world_normals_frag_code), len(gbuffer_world_normals_frag_code))
    raygen_code_aligned: []byte = mem.make_aligned([]byte, len(raygen_code), 4, context.temp_allocator) or_else panic("failed to align mesh shader bytecode")
    mem.copy(raw_data(raygen_code_aligned), raw_data(raygen_code), len(raygen_code))
    raymiss_code_aligned: []byte = mem.make_aligned([]byte, len(raymiss_code), 4, context.temp_allocator) or_else panic("failed to align mesh shader bytecode")
    mem.copy(raw_data(raymiss_code_aligned), raw_data(raymiss_code), len(raymiss_code))
    rayhit_code_aligned: []byte = mem.make_aligned([]byte, len(rayhit_code), 4, context.temp_allocator) or_else panic("failed to align mesh shader bytecode")
    mem.copy(raw_data(rayhit_code_aligned), raw_data(rayhit_code), len(rayhit_code))

    // Create shader objects.
    {
        shader_cis := [?]vk.ShaderCreateInfoEXT {
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(uv_space_vert_code_aligned) * size_of(uv_space_vert_code_aligned[0]),
                pCode = raw_data(uv_space_vert_code_aligned),
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
                codeSize = len(gbuffer_world_pos_frag_code_aligned) * size_of(gbuffer_world_pos_frag_code_aligned[0]),
                pCode = raw_data(gbuffer_world_pos_frag_code_aligned),
                pName = "main",
                stage = { .FRAGMENT },
                flags = { },
                pushConstantRangeCount = u32(len(push_constant_ranges)),
                pPushConstantRanges = raw_data(push_constant_ranges),
            },
            {
                sType = .SHADER_CREATE_INFO_EXT,
                codeType = .SPIRV,
                codeSize = len(gbuffer_world_normals_frag_code_aligned) * size_of(gbuffer_world_normals_frag_code_aligned[0]),
                pCode = raw_data(gbuffer_world_normals_frag_code_aligned),
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
    // NOTE: The constant is wrong, this is now fixed but not in the latest release.
    SHADER_UNUSED :: ~u32(0)

    raygen_shader: vk.ShaderModule
    raymiss_shader: vk.ShaderModule
    rayhit_shader: vk.ShaderModule

    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(raygen_code_aligned) * size_of(raygen_code_aligned[0]),
            pCode = auto_cast raw_data(raygen_code_aligned),
        }
        vk_check(vk.CreateShaderModule(device, &shader_module_ci, nil, &raygen_shader))
    }
    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(raymiss_code_aligned) * size_of(raymiss_code_aligned[0]),
            pCode = auto_cast raw_data(raymiss_code_aligned),
        }
        vk_check(vk.CreateShaderModule(device, &shader_module_ci, nil, &raymiss_shader))
    }
    {
        shader_module_ci := vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            flags = {},
            codeSize = len(rayhit_code_aligned) * size_of(rayhit_code_aligned[0]),
            pCode = auto_cast raw_data(rayhit_code_aligned),
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

GBuffers :: struct
{
    world_pos: Image,
    world_normals: Image,
}

create_gbuffers :: proc(using bake: ^Bake) -> GBuffers
{
    world_pos := create_image(ctx, {
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
    }, "gbuf_worldpos")

    world_normals := create_image(ctx, {
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
    }, "gbuf_worldnormals")

    return {
        world_pos,
        world_normals
    }
}

Image :: struct
{
    img: vk.Image,
    mem: vk.DeviceMemory,
    view: vk.ImageView,
    width: u32,
    height: u32,
}

create_image :: proc(using ctx: ^Context, ci: vk.ImageCreateInfo, name := "") -> Image
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

get_buffer_device_address :: proc(device: vk.Device, buffer: Buffer) -> vk.DeviceAddress
{
    info := vk.BufferDeviceAddressInfo {
        sType = .BUFFER_DEVICE_ADDRESS_INFO,
        buffer = buffer.handle
    }
    return vk.GetBufferDeviceAddress(device, &info)
}

update_rt_desc_set :: proc(device: vk.Device, to_update: vk.DescriptorSet, tlas: vk.AccelerationStructureKHR, lightmap: Image, gbuffers: GBuffers)
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
                    imageLayout = .GENERAL,
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
                    imageLayout = .GENERAL,
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
                    imageLayout = .GENERAL,
                }
            })
        }
    }
    vk.UpdateDescriptorSets(device, u32(len(writes)), raw_data(writes), 0, nil)
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
