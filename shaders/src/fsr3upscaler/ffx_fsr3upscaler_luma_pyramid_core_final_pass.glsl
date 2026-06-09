// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Core write-only luma pyramid, auto-exposure (final) pass. After the downsample
// chain has reduced SPD mip 0 to its 1x1 level, this single-threaded dispatch
// reads that scene-average [logLuma, luma] and writes the auto-exposure into
// frame info. This replicates the `index == MipCount() - 1` branch of `SpdStore`
// in ffx_fsr3upscaler_luma_pyramid.h.

#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require

#define FSR3UPSCALER_BIND_SRV_SPD_MIPS               0
#define FSR3UPSCALER_BIND_UAV_FRAME_INFO             1
#define FSR3UPSCALER_BIND_CB_CONSTANTS               2
#define FSR3UPSCALER_BIND_SAMPLER_POINT_CLAMP        3
#define FSR3UPSCALER_BIND_SAMPLER_LINEAR_CLAMP       4

#include "fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h"
#include "fsr3upscaler/ffx_fsr3upscaler_common.h"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    if (gl_GlobalInvocationID.x != 0u || gl_GlobalInvocationID.y != 0u)
    {
        return;
    }

    // The bound SRV is the 1x1 level of the reduction chain: the scene average.
    const FfxFloat32x2 fSceneAverage = LoadSpdMip(FfxInt32x2(0, 0));

    FfxFloat32x4 frameInfo = LoadFrameInfo();
    const FfxFloat32 fSceneAvgLuma = fSceneAverage.y;
    const FfxFloat32 fPrevLogLuma = frameInfo[FRAME_INFO_LOG_LUMA];
    FfxFloat32 fLogLuma = fSceneAverage.x;

    if (fPrevLogLuma < resetAutoExposureAverageSmoothing) // Compare Lavg, so small or negative values
    {
        fLogLuma = fPrevLogLuma + (fLogLuma - fPrevLogLuma) * (1.0f - exp(-DeltaTime()));
        fLogLuma = ffxMax(0.0f, fLogLuma);
    }

    frameInfo[FRAME_INFO_EXPOSURE] = ComputeAutoExposureFromLavg(fLogLuma);
    frameInfo[FRAME_INFO_LOG_LUMA] = fLogLuma;
    frameInfo[FRAME_INFO_SCENE_AVERAGE_LUMA] = fSceneAvgLuma;

    StoreFrameInfo(frameInfo);
}
