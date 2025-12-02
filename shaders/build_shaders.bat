@echo off

: set flags=--debug
set flags=

glslangvalidator -V gbuffer_world_pos.frag.glsl -o gbuffer_world_pos.frag.spv
glslangvalidator -V gbuffer_world_normals.frag.glsl -o gbuffer_world_normals.frag.spv
glslangvalidator -V gbuffer_tri_idx.frag.glsl -o gbuffer_tri_idx.frag.spv
glslangvalidator -V uv_space.vert.glsl -o uv_space.vert.spv
glslangvalidator -V dilate.comp.glsl -o dilate.comp.spv
glslangvalidator -V seams.vert.glsl -o seams.vert.spv
glslangvalidator -V seams.frag.glsl -o seams.frag.spv
glslangvalidator --target-env spirv1.6 -V pathtrace.rgen.glsl -o pathtrace.rgen.spv
glslangvalidator --target-env spirv1.6 -V pathtrace.rmiss.glsl -o pathtrace.rmiss.spv
glslangvalidator --target-env spirv1.6 -V pathtrace.rchit.glsl -o pathtrace.rchit.spv
glslangvalidator --target-env spirv1.6 -V push_samples.rgen.glsl -o push_samples.rgen.spv
glslangvalidator --target-env spirv1.6 -V push_samples.rmiss.glsl -o push_samples.rmiss.spv
glslangvalidator --target-env spirv1.6 -V push_samples.rchit.glsl -o push_samples.rchit.spv
