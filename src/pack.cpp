/*
 * Copyright (C) 2018 tianylijun@163.com. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 * Contributors:
 *     Lee (tianylijun@163.com)
 */

#include <stdio.h>
#include <string.h>
#include "armNeon.h"
#include "common.h"
#include "pack.h"

void tinySgemmConvPackA4x4_fp32_fp32(float *pA, float *pPackA, uint32_t M, uint32_t K)
{
    uint32_t i = 0, j = 0;
    uint32_t MDiv4, MHas2, MHas1, KDiv4, KHas2, KHas1;
    float *pSrcStart, *pDstStart;

    POINTER_CHECK_NO_RET(pA);
    POINTER_CHECK_NO_RET(pPackA);

    MDiv4 = M>>2;
    MHas2 = (M>>1)&1;
    MHas1 = M&1;
    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;

    for (j = 0; j < MDiv4; ++j)
    {
        pSrcStart = pA     + j*4*K;
        pDstStart = pPackA + j*4*K;

        for (i = 0; i < KDiv4; ++i)
        {
            float32x4x4_t vsrc_32x4x4;

            vsrc_32x4x4.val[0][0] = pSrcStart[0];
            vsrc_32x4x4.val[1][0] = pSrcStart[1];
            vsrc_32x4x4.val[2][0] = pSrcStart[2];
            vsrc_32x4x4.val[3][0] = pSrcStart[3];

            vsrc_32x4x4.val[0][1] = pSrcStart[K];
            vsrc_32x4x4.val[1][1] = pSrcStart[K+1];
            vsrc_32x4x4.val[2][1] = pSrcStart[K+2];
            vsrc_32x4x4.val[3][1] = pSrcStart[K+3];

            vsrc_32x4x4.val[0][2] = pSrcStart[2*K];
            vsrc_32x4x4.val[1][2] = pSrcStart[2*K+1];
            vsrc_32x4x4.val[2][2] = pSrcStart[2*K+2];
            vsrc_32x4x4.val[3][2] = pSrcStart[2*K+3];

            vsrc_32x4x4.val[0][3] = pSrcStart[3*K];
            vsrc_32x4x4.val[1][3] = pSrcStart[3*K+1];
            vsrc_32x4x4.val[2][3] = pSrcStart[3*K+2];
            vsrc_32x4x4.val[3][3] = pSrcStart[3*K+3];

            vst1q_f32_x4(pDstStart, vsrc_32x4x4);
            pSrcStart += 4;
            pDstStart += 16;
        }

        if(KHas2)
        {
            float32x4x2_t vsrc_32x4x2;
            vsrc_32x4x2.val[0][0] = pSrcStart[0];
            vsrc_32x4x2.val[1][0] = pSrcStart[1];

            vsrc_32x4x2.val[0][1] = pSrcStart[K];
            vsrc_32x4x2.val[1][1] = pSrcStart[K+1];

            vsrc_32x4x2.val[0][2] = pSrcStart[2*K];
            vsrc_32x4x2.val[1][2] = pSrcStart[2*K+1];

            vsrc_32x4x2.val[0][3] = pSrcStart[3*K];
            vsrc_32x4x2.val[1][3] = pSrcStart[3*K+1];

            vst1q_f32_x2(pDstStart, vsrc_32x4x2);
            pSrcStart += 2;
            pDstStart += 8;
        }

        if(KHas1)
        {
            pDstStart[0] = pSrcStart[0];
            pDstStart[1] = pSrcStart[K];
            pDstStart[2] = pSrcStart[2*K];
            pDstStart[3] = pSrcStart[3*K];
        }
    }

    if(MHas2)
    {
        pSrcStart = pA + MDiv4*4*K;
        pDstStart = pPackA + MDiv4*4*K;

        for (i = 0; i < KDiv4; ++i)
        {
            float32x4x2_t vsrc_32x4x2;
            vsrc_32x4x2.val[0][0] = pSrcStart[0];
            vsrc_32x4x2.val[0][2] = pSrcStart[1];
            vsrc_32x4x2.val[1][0] = pSrcStart[2];
            vsrc_32x4x2.val[1][2] = pSrcStart[3];

            vsrc_32x4x2.val[0][1] = pSrcStart[K];
            vsrc_32x4x2.val[0][3] = pSrcStart[K+1];
            vsrc_32x4x2.val[1][1] = pSrcStart[K+2];
            vsrc_32x4x2.val[1][3] = pSrcStart[K+3];

            vst1q_f32_x2(pDstStart, vsrc_32x4x2);
            pSrcStart += 4;
            pDstStart += 8;
        }

        if(KHas2)
        {
            pDstStart[0] = pSrcStart[0];
            pDstStart[2] = pSrcStart[1];
            pDstStart[1] = pSrcStart[K];
            pDstStart[3] = pSrcStart[K+1];
            pSrcStart += 2;
            pDstStart += 4;
        }

        if(KHas1)
        {
            pDstStart[0] = pSrcStart[0];
            pDstStart[1] = pSrcStart[K];
        }
    }

    if (MHas1)
        memcpy(pPackA + (M-1)*K, pA + (M-1)*K, K*sizeof(*pA));
}

#ifdef __aarch64__
static void tinySgemmConvPackB4x24_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    uint32_t i = 0, Nx4 = 0;
    uint32_t KDiv4, KHas2, KHas1;
    float *pSrcStart_0, *pSrcStart_1, *pSrcStart_2, *pSrcStart_3, *pDstStart;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;
    pSrcStart_0 = pB;
    pSrcStart_1 = pB + N;
    pSrcStart_2 = pB + 2*N;
    pSrcStart_3 = pB + 3*N;
    Nx4 = N*4;
    pDstStart = pPackB;

    for (i = 0; i < KDiv4; ++i)
    {
        float32x4x4_t vsrc_32x4x4_0, vsrc_32x4x4_1, vsrc_32x4x4_2, vsrc_32x4x4_3;
        float32x4x2_t vsrc_32x4x2_0, vsrc_32x4x2_1, vsrc_32x4x2_2, vsrc_32x4x2_3;

        vsrc_32x4x4_0 = vld1q_f32_x4(pSrcStart_0);
        vsrc_32x4x2_0 = vld1q_f32_x2(pSrcStart_0+16);
        vst1q_f32_x4(pDstStart,   vsrc_32x4x4_0);
        vst1q_f32_x2(pDstStart+16, vsrc_32x4x2_0);

        vsrc_32x4x4_1 = vld1q_f32_x4(pSrcStart_1);
        vsrc_32x4x2_1 = vld1q_f32_x2(pSrcStart_1+16);
        vst1q_f32_x4(pDstStart+24,  vsrc_32x4x4_1);
        vst1q_f32_x2(pDstStart+40, vsrc_32x4x2_1);

        vsrc_32x4x4_2 = vld1q_f32_x4(pSrcStart_2);
        vsrc_32x4x2_2 = vld1q_f32_x2(pSrcStart_2+16);
        vst1q_f32_x4(pDstStart+48, vsrc_32x4x4_2);
        vst1q_f32_x2(pDstStart+64, vsrc_32x4x2_2);

        vsrc_32x4x4_3 = vld1q_f32_x4(pSrcStart_3);
        vsrc_32x4x2_3 = vld1q_f32_x2(pSrcStart_3+16);
        vst1q_f32_x4(pDstStart+72, vsrc_32x4x4_3);
        vst1q_f32_x2(pDstStart+88, vsrc_32x4x2_3);

        pDstStart += 96;
        pSrcStart_0 += Nx4;
        pSrcStart_1 += Nx4;
        pSrcStart_2 += Nx4;
        pSrcStart_3 += Nx4;
    }

    if (KHas2)
    {
        float32x4x4_t vsrc_32x4x4_0, vsrc_32x4x4_1;
        float32x4x2_t vsrc_32x4x2_0, vsrc_32x4x2_1;

        vsrc_32x4x4_0 = vld1q_f32_x4(pSrcStart_0);
        vsrc_32x4x2_0 = vld1q_f32_x2(pSrcStart_0+16);
        vst1q_f32_x4(pDstStart, vsrc_32x4x4_0);
        vst1q_f32_x2(pDstStart+16, vsrc_32x4x2_0);

        vsrc_32x4x4_1 = vld1q_f32_x4(pSrcStart_1);
        vsrc_32x4x2_1 = vld1q_f32_x2(pSrcStart_1+16);
        vst1q_f32_x4(pDstStart+24, vsrc_32x4x4_1);
        vst1q_f32_x2(pDstStart+40, vsrc_32x4x2_1);

        pDstStart   += 48;
        pSrcStart_0 += 2*N;
    }

    if (KHas1)
    {
        float32x4x4_t vsrc_32x4x4_0;
        float32x4x2_t vsrc_32x4x2_0;

        vsrc_32x4x4_0 = vld1q_f32_x4(pSrcStart_0);
        vsrc_32x4x2_0 = vld1q_f32_x2(pSrcStart_0+16);
        vst1q_f32_x4(pDstStart, vsrc_32x4x4_0);
        vst1q_f32_x2(pDstStart+16, vsrc_32x4x2_0);
    }
}
#endif

static void tinySgemmConvPackB4x16_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    uint32_t i = 0, Nx4 = 0;
    uint32_t KDiv4, KHas2, KHas1;
    float *pSrcStart_0, *pSrcStart_1, *pSrcStart_2, *pSrcStart_3, *pDstStart;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;
    pSrcStart_0 = pB;
    pSrcStart_1 = pB + N;
    pSrcStart_2 = pB + 2*N;
    pSrcStart_3 = pB + 3*N;
    Nx4 = N*4;
    pDstStart = pPackB;

    for (i = 0; i < KDiv4; ++i)
    {
        float32x4x4_t vsrc_32x4x4_0, vsrc_32x4x4_1, vsrc_32x4x4_2, vsrc_32x4x4_3;

        vsrc_32x4x4_0 = vld1q_f32_x4(pSrcStart_0);
        vst1q_f32_x4(pDstStart, vsrc_32x4x4_0);

        vsrc_32x4x4_1 = vld1q_f32_x4(pSrcStart_1);
        vst1q_f32_x4(pDstStart+16, vsrc_32x4x4_1);

        vsrc_32x4x4_2 = vld1q_f32_x4(pSrcStart_2);
        vst1q_f32_x4(pDstStart+32, vsrc_32x4x4_2);

        vsrc_32x4x4_3 = vld1q_f32_x4(pSrcStart_3);
        vst1q_f32_x4(pDstStart+48, vsrc_32x4x4_3);

        pDstStart += 64;
        pSrcStart_0 += Nx4;
        pSrcStart_1 += Nx4;
        pSrcStart_2 += Nx4;
        pSrcStart_3 += Nx4;
    }

    if (KHas2)
    {
        float32x4x4_t vsrc_32x4x4_0, vsrc_32x4x4_1;

        vsrc_32x4x4_0 = vld1q_f32_x4(pSrcStart_0);
        vst1q_f32_x4(pDstStart, vsrc_32x4x4_0);

        vsrc_32x4x4_1 = vld1q_f32_x4(pSrcStart_1);
        vst1q_f32_x4(pDstStart+16, vsrc_32x4x4_1);

        pDstStart   += 32;
        pSrcStart_0 += 2*N;
    }

    if (KHas1)
    {
        float32x4x4_t vsrc_32x4x4 = vld1q_f32_x4(pSrcStart_0);
        vst1q_f32_x4(pDstStart, vsrc_32x4x4);
    }
}

#if 1
extern "C" void tinySgemmConvPackB4x12_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N);
#else
static void tinySgemmConvPackB4x12_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    uint32_t i = 0, Nx4 = 0;
    uint32_t KDiv4, KHas2, KHas1;
    float *pSrcStart_0, *pSrcStart_1, *pSrcStart_2, *pSrcStart_3, *pDstStart;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;
    pSrcStart_0 = pB;
    pSrcStart_1 = pB + N;
    pSrcStart_2 = pB + 2*N;
    pSrcStart_3 = pB + 3*N;
    Nx4 = N*4;

    pDstStart = pPackB;

    for (i = 0; i < KDiv4; ++i)
    {
        float32x4x3_t vsrc_32x4x3_0, vsrc_32x4x3_1, vsrc_32x4x3_2, vsrc_32x4x3_3;

        vsrc_32x4x3_0 = vld1q_f32_x3(pSrcStart_0);
        vst1q_f32_x3(pDstStart, vsrc_32x4x3_0);

        vsrc_32x4x3_1 = vld1q_f32_x3(pSrcStart_1);
        vst1q_f32_x3(pDstStart + 12, vsrc_32x4x3_1);

        vsrc_32x4x3_2 = vld1q_f32_x3(pSrcStart_2);
        vst1q_f32_x3(pDstStart + 24, vsrc_32x4x3_2);

        vsrc_32x4x3_3 = vld1q_f32_x3(pSrcStart_3);
        vst1q_f32_x3(pDstStart + 36, vsrc_32x4x3_3);

        pDstStart += 48;
        pSrcStart_0 += Nx4;
        pSrcStart_1 += Nx4;
        pSrcStart_2 += Nx4;
        pSrcStart_3 += Nx4;
    }

    if (KHas2)
    {
        float32x4x3_t vsrc_32x4x3_0, vsrc_32x4x3_1;

        vsrc_32x4x3_0 = vld1q_f32_x3(pSrcStart_0);
        vst1q_f32_x3(pDstStart, vsrc_32x4x3_0);

        vsrc_32x4x3_1 = vld1q_f32_x3(pSrcStart_1);
        vst1q_f32_x3(pDstStart + 12, vsrc_32x4x3_1);

        pDstStart   += 24;
        pSrcStart_0 += 2*N;
    }

    if (KHas1)
    {
        float32x4x3_t vsrc_32x4x3 = vld1q_f32_x3(pSrcStart_0);
        vst1q_f32_x3(pDstStart, vsrc_32x4x3);
    }
}
#endif

static void tinySgemmConvPackB4x8_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    uint32_t i = 0, Nx4 = 0;
    uint32_t KDiv4, KHas2, KHas1;
    float *pSrcStart_0, *pSrcStart_1, *pSrcStart_2, *pSrcStart_3, *pDstStart;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;
    pSrcStart_0 = pB;
    pSrcStart_1 = pB + N;
    pSrcStart_2 = pB + 2*N;
    pSrcStart_3 = pB + 3*N;
    Nx4 = N*4;
    pDstStart = pPackB;

    for (i = 0; i < KDiv4; ++i)
    {
        float32x4x2_t vsrc_32x4x2_0, vsrc_32x4x2_1, vsrc_32x4x2_2, vsrc_32x4x2_3;

        vsrc_32x4x2_0 = vld1q_f32_x2(pSrcStart_0);
        vst1q_f32_x2(pDstStart, vsrc_32x4x2_0);

        vsrc_32x4x2_1 = vld1q_f32_x2(pSrcStart_1);
        vst1q_f32_x2(pDstStart + 8, vsrc_32x4x2_1);

        vsrc_32x4x2_2 = vld1q_f32_x2(pSrcStart_2);
        vst1q_f32_x2(pDstStart + 16, vsrc_32x4x2_2);

        vsrc_32x4x2_3 = vld1q_f32_x2(pSrcStart_3);
        vst1q_f32_x2(pDstStart + 24, vsrc_32x4x2_3);

        pDstStart += 32;
        pSrcStart_0 += Nx4;
        pSrcStart_1 += Nx4;
        pSrcStart_2 += Nx4;
        pSrcStart_3 += Nx4;
    }

    if (KHas2)
    {
        float32x4x2_t vsrc_32x4x2_0, vsrc_32x4x2_1;

        vsrc_32x4x2_0 = vld1q_f32_x2(pSrcStart_0);
        vst1q_f32_x2(pDstStart, vsrc_32x4x2_0);

        vsrc_32x4x2_1 = vld1q_f32_x2(pSrcStart_1);
        vst1q_f32_x2(pDstStart + 8, vsrc_32x4x2_1);

        pDstStart   += 16;
        pSrcStart_0 += 2*N;
    }

    if (KHas1)
    {
        float32x4x2_t vsrc_32x4x2 = vld1q_f32_x2(pSrcStart_0);
        vst1q_f32_x2(pDstStart, vsrc_32x4x2);
    }
}

static void tinySgemmConvPackB4x4_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    uint32_t i = 0, Nx4 = 0;
    uint32_t KDiv4, KHas2, KHas1;
    float *pSrcStart_0, *pSrcStart_1, *pSrcStart_2, *pSrcStart_3, *pDstStart;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;
    pSrcStart_0 = pB;
    pSrcStart_1 = pB + N;
    pSrcStart_2 = pB + 2*N;
    pSrcStart_3 = pB + 3*N;
    Nx4 = N*4;
    pDstStart = pPackB;

    for (i = 0; i < KDiv4; ++i)
    {
        float32x4x4_t vsrc_32x4x4;
        vsrc_32x4x4.val[0] = vld1q_f32(pSrcStart_0);
        vsrc_32x4x4.val[1] = vld1q_f32(pSrcStart_1);
        vsrc_32x4x4.val[2] = vld1q_f32(pSrcStart_2);
        vsrc_32x4x4.val[3] = vld1q_f32(pSrcStart_3);
        vst1q_f32_x4(pDstStart, vsrc_32x4x4);

        pDstStart += 16;
        pSrcStart_0 += Nx4;
        pSrcStart_1 += Nx4;
        pSrcStart_2 += Nx4;
        pSrcStart_3 += Nx4;
    }

    if (KHas2)
    {
        float32x4x2_t vsrc_32x4x2;
        vsrc_32x4x2.val[0] = vld1q_f32(pSrcStart_0);
        vsrc_32x4x2.val[1] = vld1q_f32(pSrcStart_1);
        vst1q_f32_x2(pDstStart, vsrc_32x4x2);

        pDstStart   += 8;
        pSrcStart_0 += 2*N;
    }

    if (KHas1)
    {
        float32x4_t vsrc_32x4 = vld1q_f32(pSrcStart_0);
        vst1q_f32(pDstStart, vsrc_32x4);
    }
}

static void tinySgemmConvPackB4x2_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    uint32_t i = 0, Nx4 = 0;
    uint32_t KDiv4, KHas2, KHas1;
    float *pSrcStart_0, *pSrcStart_1, *pSrcStart_2, *pSrcStart_3, *pDstStart;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;
    pSrcStart_0 = pB;
    pSrcStart_1 = pB + N;
    pSrcStart_2 = pB + 2*N;
    pSrcStart_3 = pB + 3*N;
    Nx4 = N*4;
    pDstStart = pPackB;

    for (i = 0; i < KDiv4; ++i)
    {
        float32x2x4_t vsrc_32x2x4;
        vsrc_32x2x4.val[0] = vld1_f32(pSrcStart_0);
        vsrc_32x2x4.val[1] = vld1_f32(pSrcStart_1);
        vsrc_32x2x4.val[2] = vld1_f32(pSrcStart_2);
        vsrc_32x2x4.val[3] = vld1_f32(pSrcStart_3);
        vst1_f32_x4(pDstStart, vsrc_32x2x4);

        pDstStart += 8;
        pSrcStart_0 += Nx4;
        pSrcStart_1 += Nx4;
        pSrcStart_2 += Nx4;
        pSrcStart_3 += Nx4;

    }

    if (KHas2)
    {
        float32x2x2_t vsrc_32x2x2;
        vsrc_32x2x2.val[0] = vld1_f32(pSrcStart_0);
        vsrc_32x2x2.val[1] = vld1_f32(pSrcStart_1);
        vst1_f32_x2(pDstStart, vsrc_32x2x2);

        pDstStart   += 4;
        pSrcStart_0 += 2*N;
    }

    if (KHas1)
    {
        float32x2_t vsrc_32x2 = vld1_f32(pSrcStart_0);
        vst1_f32(pDstStart, vsrc_32x2);
    }
}

static void tinySgemmConvPackB4x1_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    float *pSrcStart = pB;
    float *pDstStart = pPackB;
    for (uint32_t i = 0; i < K; ++i)
        pDstStart[i] = pSrcStart[i*N];
}

void tinySgemmConvPackBLeftN_fp32_fp32(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    uint32_t leftN, leftNHas16, leftNHas8, leftNHas4, leftNHas2, leftNHas1;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    leftN      = N%TINY_SGEMM_UNIT_N;
    leftNHas16 = (leftN>>4)&1;
    leftNHas8  = (leftN>>3)&1;
    leftNHas4  = (leftN>>2)&1;
    leftNHas2  = (leftN>>1)&1;
    leftNHas1  = leftN&1;

    if (leftNHas16)
    {
        tinySgemmConvPackB4x16_fp32_fp32_unit(pB, pPackB, K, N);
        pB     += 16;
        pPackB += 16*K;
    }

    if (leftNHas8)
    {
        tinySgemmConvPackB4x8_fp32_fp32_unit(pB, pPackB, K, N);
        pB     += 8;
        pPackB += 8*K;
    }

    if (leftNHas4)
    {
        tinySgemmConvPackB4x4_fp32_fp32_unit(pB, pPackB, K, N);
        pB     += 4;
        pPackB += 4*K;
    }

    if (leftNHas2)
    {
        tinySgemmConvPackB4x2_fp32_fp32_unit(pB, pPackB, K, N);
        pB     += 2;
        pPackB += 2*K;
    }

    if (leftNHas1)
        tinySgemmConvPackB4x1_fp32_fp32_unit(pB, pPackB, K, N);
}

void tinySgemmConvPackBUnitN_fp32_fp32(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
#ifdef __aarch64__
    tinySgemmConvPackB4x24_fp32_fp32_unit(pB, pPackB, K, N);
#else
    tinySgemmConvPackB4x12_fp32_fp32_unit(pB, pPackB, K, N);
#endif
}
