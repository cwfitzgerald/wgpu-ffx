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

// Core write-only shading-change pyramid, mip 0. Replicates FFX's
// `SpdReduceLoadSourceImage4` (a 2x2 box average of the per-texel shading-change
// difference metric) and `SpdStore` at index 0, writing the half-resolution
// [difference, signSum] into SPD mip 0. The downsample passes then produce mips 1
// and 2, which the shading-change pass samples.

#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require

#define FSR3UPSCALER_BIND_SRV_CURRENT_LUMA                  0
#define FSR3UPSCALER_BIND_SRV_PREVIOUS_LUMA                 1
#define FSR3UPSCALER_BIND_SRV_DILATED_MOTION_VECTORS        2
#define FSR3UPSCALER_BIND_SRV_INPUT_EXPOSURE                3
#define FSR3UPSCALER_BIND_UAV_SPD_MIP_DEST                  4
#define FSR3UPSCALER_BIND_CB_CONSTANTS                      5
#define FSR3UPSCALER_BIND_SAMPLER_POINT_CLAMP               6
#define FSR3UPSCALER_BIND_SAMPLER_LINEAR_CLAMP              7

#include "fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h"
#include "fsr3upscaler/ffx_fsr3upscaler_common.h"
#include "fsr3upscaler/ffx_fsr3upscaler_shading_change_diff.h"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main()
{
    // SPD mip 0 is the half-resolution level: floor(RenderSize() / 2), matching
    // the physical mip chain (allocated from max_render / 2).
    const FfxInt32x2 iMip0Size = max(RenderSize() >> 1, FfxInt32x2(1, 1));
    const FfxInt32x2 iDstPos = FfxInt32x2(gl_GlobalInvocationID.xy);

    if (any(greaterThanEqual(iDstPos, iMip0Size)))
    {
        return;
    }

    FfxFloat32x4 fSum = FfxFloat32x4(0.0, 0.0, 0.0, 0.0);
    for (FfxInt32 dy = 0; dy < 2; ++dy)
    {
        for (FfxInt32 dx = 0; dx < 2; ++dx)
        {
            fSum += SpdLoadSourceImage(FfxFloat32x2(iDstPos * 2 + FfxInt32x2(dx, dy)), 0);
        }
    }

    StoreSpdMipDest(iDstPos, (fSum * 0.25).xy);
}
