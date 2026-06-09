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

// Core write-only SPD downsample: produces SPD mip `S+1` from mip `S` as a plain
// 2x2 box average, replicating FFX's `SpdReduce4`. One dispatch per level; the
// source level `S` arrives via the SPD_CORE uniform. Used for both the luma and
// shading-change pyramids on the Core profile (the mip data is rg). The per-level
// texel dimensions are derived from `RenderSize()` so only the real subregion of
// the max-sized SPD texture is processed under dynamic resolution.

#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require

#define FSR3UPSCALER_BIND_SRV_SPD_MIPS               0
#define FSR3UPSCALER_BIND_UAV_SPD_MIP_DEST           1
#define FSR3UPSCALER_BIND_CB_CONSTANTS               2
#define FSR3UPSCALER_BIND_CB_SPD_CORE                3
#define FSR3UPSCALER_BIND_SAMPLER_POINT_CLAMP        4
#define FSR3UPSCALER_BIND_SAMPLER_LINEAR_CLAMP       5

#include "fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h"
#include "fsr3upscaler/ffx_fsr3upscaler_common.h"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Texel dimensions of SPD mip `mip` (mip 0 is the half-resolution level),
// clamped to at least 1x1: floor(RenderSize() / 2^(mip + 1)). Floor halving
// matches the physical mip chain of the spd_mips texture (allocated from
// max_render / 2), so the data region never exceeds the bound mip's dimensions.
FfxInt32x2 SpdMipSize(FfxInt32 mip)
{
    return max(RenderSize() >> (mip + 1), FfxInt32x2(1, 1));
}

void main()
{
    const FfxInt32 srcLevel = FfxInt32(SpdCoreSrcLevel());
    const FfxInt32x2 iDstSize = SpdMipSize(srcLevel + 1);
    const FfxInt32x2 iDstPos = FfxInt32x2(gl_GlobalInvocationID.xy);

    if (any(greaterThanEqual(iDstPos, iDstSize)))
    {
        return;
    }

    const FfxInt32x2 iSrcSize = SpdMipSize(srcLevel);

    FfxFloat32x2 fSum = FfxFloat32x2(0.0, 0.0);
    for (FfxInt32 dy = 0; dy < 2; ++dy)
    {
        for (FfxInt32 dx = 0; dx < 2; ++dx)
        {
            const FfxInt32x2 iSrcPos = min(iDstPos * 2 + FfxInt32x2(dx, dy), iSrcSize - 1);
            fSum += LoadSpdMip(iSrcPos);
        }
    }

    StoreSpdMipDest(iDstPos, fSum * 0.25);
}
