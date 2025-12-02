@echo off

: set flags=--debug
set flags=

glslangvalidator -V model_to_proj.vert.glsl -o model_to_proj.vert.spv
glslangvalidator -V lit.frag.glsl -o lit.frag.spv
glslangvalidator -V tonemap.vert.glsl -o tonemap.vert.spv
glslangvalidator -V tonemap.frag.glsl -o tonemap.frag.spv
