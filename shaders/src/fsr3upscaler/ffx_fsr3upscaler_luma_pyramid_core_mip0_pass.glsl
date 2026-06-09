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

// Core write-only luma pyramid, mip 0. Replicates FFX's
// `SpdReduceLoadSourceImage4` (a 2x2 box average of the per-texel source value)
// and `SpdStore` at index 0: it writes the half-resolution [logLuma, luma] into
// SPD mip 0 (scratch for the reduction chain) and the averaged farthest depth
// into farthest_depth_mip1. The subsequent downsample passes reduce SPD mip 0 to
// 1x1 for the auto-exposure (final) pass.

#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require

#define FSR3UPSCALER_BIND_SRV_CURRENT_LUMA           0
#define FSR3UPSCALER_BIND_SRV_FARTHEST_DEPTH         1
#define FSR3UPSCALER_BIND_UAV_SPD_MIP_DEST           2
#define FSR3UPSCALER_BIND_UAV_FARTHEST_DEPTH_MIP1    3
#define FSR3UPSCALER_BIND_CB_CONSTANTS               4
#define FSR3UPSCALER_BIND_SAMPLER_POINT_CLAMP        5
#define FSR3UPSCALER_BIND_SAMPLER_LINEAR_CLAMP       6

#include "fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h"
#include "fsr3upscaler/ffx_fsr3upscaler_common.h"

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

    FfxFloat32 fSumLogLuma = 0.0;
    FfxFloat32 fSumLuma = 0.0;
    FfxFloat32 fSumDepth = 0.0;
    for (FfxInt32 dy = 0; dy < 2; ++dy)
    {
        for (FfxInt32 dx = 0; dx < 2; ++dx)
        {
            const FfxInt32x2 iSrcPos = ClampLoad(iDstPos * 2 + FfxInt32x2(dx, dy), FfxInt32x2(0, 0), FfxInt32x2(RenderSize()));

            const FfxFloat32 fLuma = LoadCurrentLuma(iSrcPos);
            fSumLogLuma += ffxMax(FSR3UPSCALER_EPSILON, log(fLuma));
            fSumLuma += fLuma;
            fSumDepth += LoadFarthestDepth(iSrcPos);
        }
    }

    StoreSpdMipDest(iDstPos, FfxFloat32x2(fSumLogLuma * 0.25, fSumLuma * 0.25));
    StoreFarthestDepthMip1(iDstPos, fSumDepth * 0.25);
}
