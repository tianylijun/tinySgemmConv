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
#include <stdlib.h>
#include "armNeon.h"
#include "config.h"
#include "tinySgemmConv.h"
#include "innerTinySgemmConv.h"
#include "sgemm.h"

#ifdef __aarch64__
#define VMLAQ_N_F32(a,b,c,d) vfmaq_laneq_f32(a, b, c, d)
#else
#define VMLAQ_N_F32(a,b,c,d) vmlaq_lane_f32(a, b, c, d)
#endif

#ifdef __aarch64__
extern "C" void sgemm4xKx24_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis);
extern "C" void sgemm2xKx24_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis);
extern "C" void sgemm1xKx24_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis);

void sgemmMxKx24_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t MDiv4, MHas2, MHas1;
    MDiv4 = M>>2;
    MHas2 = (M>>1)&1;
    MHas1 = M&1;

    for (uint32_t i = 0; i < MDiv4; ++i)
    {
        sgemm4xKx24_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 4*K;
        pC += 4*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 4;
        if (NULL != pBasis)
            pBasis += 4;
    }

    if (MHas2)
    {
        sgemm2xKx24_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 2*K;
        pC += 2*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 2;
        if (NULL != pBasis)
            pBasis += 2;
    }

    if (MHas1)
        sgemm1xKx24_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
}
#endif

#ifndef __aarch64__
/* fp32 unit sgemm block is, A:4x4  B:4x12 C:4x12 */
extern "C" void sgemm4xKx12_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis);
extern "C" void sgemm2xKx12_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis);
extern "C" void sgemm1xKx12_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis);

void sgemmMxKx12_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t MDiv4, MHas2, MHas1;
    MDiv4 = M>>2;
    MHas2 = (M>>1)&1;
    MHas1 = M&1;

    for (uint32_t i = 0; i < MDiv4; ++i)
    {
        sgemm4xKx12_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 4*K;
        pC += 4*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 4;
        if (NULL != pBasis)
            pBasis += 4;
    }

    if (MHas2)
    {
        sgemm2xKx12_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 2*K;
        pC += 2*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 2;
        if (NULL != pBasis)
            pBasis += 2;
    }

    if (MHas1)
        sgemm1xKx12_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
}

#else

extern "C" void sgemm4xKx16_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis);
extern "C" void sgemm2xKx16_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis);
extern "C" void sgemm1xKx16_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis);

void sgemmMxKx16_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t MDiv4, MHas2, MHas1;
    MDiv4 = M>>2;
    MHas2 = (M>>1)&1;
    MHas1 = M&1;

    for (uint32_t i = 0; i < MDiv4; ++i)
    {
        sgemm4xKx16_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 4*K;
        pC += 4*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 4;
        if (NULL != pBasis)
            pBasis += 4;
    }

    if (MHas2)
    {
        sgemm2xKx16_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 2*K;
        pC += 2*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 2;
        if (NULL != pBasis)
            pBasis += 2;
    }

    if (MHas1)
        sgemm1xKx16_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
}
#endif

extern "C" void sgemm4xKx8_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis);

static void sgemm2xKx8_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t KDiv4 = K>>2;
    uint32_t KHas2 = (K>>1)&1;
    uint32_t KHas1 = K&1;

    uint32x4_t vzero;
    float32x4x2_t vsrcC32x4x2_0, vsrcC32x4x2_1;   /* 4 registers */
    vzero = veorq_u32(vzero, vzero);
    vsrcC32x4x2_0.val[0] = vreinterpretq_f32_u32(vzero);
    vsrcC32x4x2_0.val[1] = vsrcC32x4x2_0.val[0];
    vsrcC32x4x2_1 = vsrcC32x4x2_0;

    /* A:2x4 B:4x8 C:2x8 */
    for (uint32_t i = 0; i < KDiv4; ++i)
    {
        float *pCurB = pB+i*32;
        float *pCurA = pA+i*8;
        float32x4x2_t vsrcA32x4x2;                /* 2 registers */
        float32x4x4_t vsrcB32x4x4, vsrcB32x4x4_1; /* 8 registers */

        vsrcB32x4x4 = vld1q_f32_x4(pCurB);
        pCurB      += 16;
        vsrcA32x4x2 = vld1q_f32_x2(pCurA);

        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4.val[0], vsrcA32x4x2.val[0][0]);
        ARM_LOAD_PREFETCH_64(pCurB);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4.val[1], vsrcA32x4x2.val[0][0]);
        vsrcC32x4x2_1.val[0] = vmlaq_n_f32(vsrcC32x4x2_1.val[0], vsrcB32x4x4.val[0], vsrcA32x4x2.val[0][1]);
        vsrcC32x4x2_1.val[1] = vmlaq_n_f32(vsrcC32x4x2_1.val[1], vsrcB32x4x4.val[1], vsrcA32x4x2.val[0][1]);

        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4.val[2], vsrcA32x4x2.val[0][2]);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4.val[3], vsrcA32x4x2.val[0][2]);
        vsrcC32x4x2_1.val[0] = vmlaq_n_f32(vsrcC32x4x2_1.val[0], vsrcB32x4x4.val[2], vsrcA32x4x2.val[0][3]);
        vsrcB32x4x4_1        = vld1q_f32_x4(pCurB);
        vsrcC32x4x2_1.val[1] = vmlaq_n_f32(vsrcC32x4x2_1.val[1], vsrcB32x4x4.val[3], vsrcA32x4x2.val[0][3]);

        ARM_LOAD_PREFETCH_64(pCurB+16);
        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4_1.val[0], vsrcA32x4x2.val[1][0]);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4_1.val[1], vsrcA32x4x2.val[1][0]);
        vsrcC32x4x2_1.val[0] = vmlaq_n_f32(vsrcC32x4x2_1.val[0], vsrcB32x4x4_1.val[0], vsrcA32x4x2.val[1][1]);
        vsrcC32x4x2_1.val[1] = vmlaq_n_f32(vsrcC32x4x2_1.val[1], vsrcB32x4x4_1.val[1], vsrcA32x4x2.val[1][1]);
        ARM_LOAD_PREFETCH_64(pCurA+16);
        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4_1.val[2], vsrcA32x4x2.val[1][2]);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4_1.val[3], vsrcA32x4x2.val[1][2]);
        vsrcC32x4x2_1.val[0] = vmlaq_n_f32(vsrcC32x4x2_1.val[0], vsrcB32x4x4_1.val[2], vsrcA32x4x2.val[1][3]);
        vsrcC32x4x2_1.val[1] = vmlaq_n_f32(vsrcC32x4x2_1.val[1], vsrcB32x4x4_1.val[3], vsrcA32x4x2.val[1][3]);
    }

    /* A:2x2 B:2x8 C:2x8 */
    if (KHas2)
    {
        float *pCurB = pB+KDiv4*32;
        float *pCurA = pA+KDiv4*8;
        float32x4_t vsrcA32x4;      /* 1 registers */
        float32x4x4_t vsrcB32x4x4;  /* 4 registers */

        vsrcB32x4x4 = vld1q_f32_x4(pCurB);
        pCurB      += 16;
        vsrcA32x4   = vld1q_f32(pCurA);

        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4.val[0], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_64(pCurB);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4.val[1], vsrcA32x4[0]);
        vsrcC32x4x2_1.val[0] = vmlaq_n_f32(vsrcC32x4x2_1.val[0], vsrcB32x4x4.val[0], vsrcA32x4[1]);
        vsrcC32x4x2_1.val[1] = vmlaq_n_f32(vsrcC32x4x2_1.val[1], vsrcB32x4x4.val[1], vsrcA32x4[1]);

        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4.val[2], vsrcA32x4[2]);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4.val[3], vsrcA32x4[2]);
        vsrcC32x4x2_1.val[0] = vmlaq_n_f32(vsrcC32x4x2_1.val[0], vsrcB32x4x4.val[2], vsrcA32x4[3]);
        vsrcC32x4x2_1.val[1] = vmlaq_n_f32(vsrcC32x4x2_1.val[1], vsrcB32x4x4.val[3], vsrcA32x4[3]);
    }

    /* A:2x1 B:1x8 C:2x8 */
    if (KHas1)
    {
        float *pCurB = pB+(K-1)*8;
        float *pCurA = pA+(K-1)*2;
        float32x4x2_t vsrcB32x4x2;
        vsrcB32x4x2 = vld1q_f32_x2(pCurB);
        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x2.val[0], pCurA[0]);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x2.val[1], pCurA[0]);
        vsrcC32x4x2_1.val[0] = vmlaq_n_f32(vsrcC32x4x2_1.val[0], vsrcB32x4x2.val[0], pCurA[1]);
        vsrcC32x4x2_1.val[1] = vmlaq_n_f32(vsrcC32x4x2_1.val[1], vsrcB32x4x2.val[1], pCurA[1]);
    }

    if (pBasis)
    {
        /* one basis for one channel */
        float32x4_t vbasis32x4_0 = vdupq_n_f32(pBasis[0]);
        float32x4_t vbasis32x4_1 = vdupq_n_f32(pBasis[1]);
        vsrcC32x4x2_0.val[0] = vaddq_f32(vsrcC32x4x2_0.val[0], vbasis32x4_0);
        vsrcC32x4x2_0.val[1] = vaddq_f32(vsrcC32x4x2_0.val[1], vbasis32x4_0);
        vsrcC32x4x2_1.val[0] = vaddq_f32(vsrcC32x4x2_1.val[0], vbasis32x4_1);
        vsrcC32x4x2_1.val[1] = vaddq_f32(vsrcC32x4x2_1.val[1], vbasis32x4_1);
    }

    if (bRelu)
    {
        uint32x4_t vmask;
        float32x4_t vzero;
        vmask = veorq_u32(vmask, vmask);
        vzero = vreinterpretq_f32_u32(vmask);

        vmask                = vcleq_f32(vsrcC32x4x2_0.val[0], vzero);
        vsrcC32x4x2_0.val[0] = vbslq_f32(vmask, vzero, vsrcC32x4x2_0.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x2_0.val[1], vzero);
        vsrcC32x4x2_0.val[1] = vbslq_f32(vmask, vzero, vsrcC32x4x2_0.val[1]);

        vmask                = vcleq_f32(vsrcC32x4x2_1.val[0], vzero);
        vsrcC32x4x2_1.val[0] = vbslq_f32(vmask, vzero, vsrcC32x4x2_1.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x2_1.val[1], vzero);
        vsrcC32x4x2_1.val[1] = vbslq_f32(vmask, vzero, vsrcC32x4x2_1.val[1]);
    }
    else if (pPrelu)
    {
        uint32x4_t vmask;
        float32x4_t vscale32x4, vmul, vzero;
        vmask = veorq_u32(vmask, vmask);
        vzero = vreinterpretq_f32_u32(vmask);
        if (bSharedPrelu) /* all channel use same prelu */
            vscale32x4 = vdupq_n_f32(pPrelu[0]);
        else
            vscale32x4 = vld1q_f32(pPrelu);

        vmask                = vcleq_f32(vsrcC32x4x2_0.val[0], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x2_0.val[0], vscale32x4[0]);
        vsrcC32x4x2_0.val[0] = vbslq_f32(vmask, vmul, vsrcC32x4x2_0.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x2_0.val[1], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x2_0.val[1], vscale32x4[0]);
        vsrcC32x4x2_0.val[1] = vbslq_f32(vmask, vmul, vsrcC32x4x2_0.val[1]);

        vmask                = vcleq_f32(vsrcC32x4x2_1.val[0], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x2_1.val[0], vscale32x4[1]);
        vsrcC32x4x2_1.val[0] = vbslq_f32(vmask, vmul, vsrcC32x4x2_1.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x2_1.val[1], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x2_1.val[1], vscale32x4[1]);
        vsrcC32x4x2_1.val[1] = vbslq_f32(vmask, vmul, vsrcC32x4x2_1.val[1]);
    }

    vst1q_f32_x2(pC,     vsrcC32x4x2_0);
    vst1q_f32_x2(pC+N,   vsrcC32x4x2_1);
}

static void sgemm1xKx8_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t KDiv4 = K>>2;
    uint32_t KHas2 = (K>>1)&1;
    uint32_t KHas1 = K&1;

    uint32x4_t vzero;
    float32x4x2_t vsrcC32x4x2_0;   /* 2 registers */
    vzero = veorq_u32(vzero, vzero);
    vsrcC32x4x2_0.val[0] = vreinterpretq_f32_u32(vzero);
    vsrcC32x4x2_0.val[1] = vsrcC32x4x2_0.val[0];

    /* A:1x4 B:4x8 C:1x8 */
    for (uint32_t i = 0; i < KDiv4; ++i)
    {
        float *pCurB = pB+i*32;
        float *pCurA = pA+i*4;
        float32x4_t vsrcA32x4;                    /* 2 registers */
        float32x4x4_t vsrcB32x4x4, vsrcB32x4x4_1; /* 8 registers */

        vsrcB32x4x4 = vld1q_f32_x4(pCurB);
        pCurB      += 16;
        vsrcA32x4  = vld1q_f32(pCurA);

        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4.val[0], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_64(pCurB);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4.val[1], vsrcA32x4[0]);
        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4.val[2], vsrcA32x4[1]);
        vsrcB32x4x4_1        = vld1q_f32_x4(pCurB);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4.val[3], vsrcA32x4[1]);

        ARM_LOAD_PREFETCH_64(pCurB+16);
        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4_1.val[0], vsrcA32x4[2]);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4_1.val[1], vsrcA32x4[2]);
        ARM_LOAD_PREFETCH_16(pCurA+4);
        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4_1.val[2], vsrcA32x4[3]);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4_1.val[3], vsrcA32x4[3]);
    }

    /* A:1x2 B:2x8 C:1x8 */
    if (KHas2)
    {
        float *pCurB = pB+KDiv4*32;
        float *pCurA = pA+KDiv4*4;
        float32x2_t vsrcA32x2;      /* 1 registers */
        float32x4x4_t vsrcB32x4x4;  /* 4 registers */

        vsrcB32x4x4 = vld1q_f32_x4(pCurB);
        pCurB      += 16;
        vsrcA32x2   = vld1_f32(pCurA);

        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4.val[0], vsrcA32x2[0]);
        ARM_LOAD_PREFETCH_32(pCurB);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4.val[1], vsrcA32x2[0]);

        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x4.val[2], vsrcA32x2[1]);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x4.val[3], vsrcA32x2[1]);
    }

    /* A:1x1 B:1x8 C:1x8 */
    if (KHas1)
    {
        float *pCurB = pB+(K-1)*8;
        float *pCurA = pA+(K-1);
        float32x4x2_t vsrcB32x4x2;
        vsrcB32x4x2 = vld1q_f32_x2(pCurB);
        vsrcC32x4x2_0.val[0] = vmlaq_n_f32(vsrcC32x4x2_0.val[0], vsrcB32x4x2.val[0], pCurA[0]);
        vsrcC32x4x2_0.val[1] = vmlaq_n_f32(vsrcC32x4x2_0.val[1], vsrcB32x4x2.val[1], pCurA[0]);
    }

    if (pBasis)
    {
        /* one basis for one channel */
        float32x4_t vbasis32x4_0 = vdupq_n_f32(pBasis[0]);
        vsrcC32x4x2_0.val[0] = vaddq_f32(vsrcC32x4x2_0.val[0], vbasis32x4_0);
        vsrcC32x4x2_0.val[1] = vaddq_f32(vsrcC32x4x2_0.val[1], vbasis32x4_0);
    }

    if (bRelu)
    {
        uint32x4_t vmask;
        float32x4_t vzero;
        vmask = veorq_u32(vmask, vmask);
        vzero = vreinterpretq_f32_u32(vmask);

        vmask                = vcleq_f32(vsrcC32x4x2_0.val[0], vzero);
        vsrcC32x4x2_0.val[0] = vbslq_f32(vmask, vzero, vsrcC32x4x2_0.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x2_0.val[1], vzero);
        vsrcC32x4x2_0.val[1] = vbslq_f32(vmask, vzero, vsrcC32x4x2_0.val[1]);
    }
    else if (pPrelu)
    {
        uint32x4_t vmask;
        float32x4_t vmul, vzero;
        vmask = veorq_u32(vmask, vmask);
        vzero = vreinterpretq_f32_u32(vmask);

        vmask                = vcleq_f32(vsrcC32x4x2_0.val[0], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x2_0.val[0], pPrelu[0]);
        vsrcC32x4x2_0.val[0] = vbslq_f32(vmask, vmul, vsrcC32x4x2_0.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x2_0.val[1], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x2_0.val[1], pPrelu[0]);
        vsrcC32x4x2_0.val[1] = vbslq_f32(vmask, vmul, vsrcC32x4x2_0.val[1]);
    }

    vst1q_f32_x2(pC,     vsrcC32x4x2_0);
}

void sgemmMxKx8_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t MDiv4, MHas2, MHas1;
    MDiv4 = M>>2;
    MHas2 = (M>>1)&1;
    MHas1 = M&1;

    for (uint32_t i = 0; i < MDiv4; ++i)
    {
        sgemm4xKx8_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 4*K;
        pC += 4*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 4;
        if (NULL != pBasis)
            pBasis += 4;
    }

    if (MHas2)
    {
        sgemm2xKx8_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 2*K;
        pC += 2*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 2;
        if (NULL != pBasis)
            pBasis += 2;
    }

    if (MHas1)
        sgemm1xKx8_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
}

extern "C" void sgemm4xKx4_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis);

static void sgemm2xKx4_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t KDiv4 = K>>2;
    uint32_t KHas2 = (K>>1)&1;
    uint32_t KHas1 = K&1;

    uint32x4_t vzero;
    float32x4x2_t vsrcC32x4x2;
    vzero = veorq_u32(vzero, vzero);
    vsrcC32x4x2.val[0] = vreinterpretq_f32_u32(vzero);
    vsrcC32x4x2.val[1] = vsrcC32x4x2.val[0];

    /* A:2x4 B:4x4 C:2x4 */
    for (uint32_t i = 0; i < KDiv4; ++i)
    {
        float *pCurB = pB+i*16;
        float *pCurA = pA+i*8;
        float32x4x2_t vsrcA32x4x2;
        float32x4x4_t vsrcB32x4x4;

        vsrcB32x4x4 = vld1q_f32_x4(pCurB);
        vsrcA32x4x2 = vld1q_f32_x2(pCurA);
        pCurA      += 8;
        vsrcC32x4x2.val[0] = vmlaq_n_f32(vsrcC32x4x2.val[0], vsrcB32x4x4.val[0], vsrcA32x4x2.val[0][0]);
        ARM_LOAD_PREFETCH_64(pCurB+16);
        vsrcC32x4x2.val[1] = vmlaq_n_f32(vsrcC32x4x2.val[1], vsrcB32x4x4.val[0], vsrcA32x4x2.val[0][1]);
        vsrcC32x4x2.val[0] = vmlaq_n_f32(vsrcC32x4x2.val[0], vsrcB32x4x4.val[1], vsrcA32x4x2.val[0][2]);
        vsrcC32x4x2.val[1] = vmlaq_n_f32(vsrcC32x4x2.val[1], vsrcB32x4x4.val[1], vsrcA32x4x2.val[0][2]);
        ARM_LOAD_PREFETCH_32(pCurA);
        vsrcC32x4x2.val[0] = vmlaq_n_f32(vsrcC32x4x2.val[0], vsrcB32x4x4.val[2], vsrcA32x4x2.val[1][0]);
        vsrcC32x4x2.val[1] = vmlaq_n_f32(vsrcC32x4x2.val[1], vsrcB32x4x4.val[2], vsrcA32x4x2.val[1][1]);
        vsrcC32x4x2.val[0] = vmlaq_n_f32(vsrcC32x4x2.val[0], vsrcB32x4x4.val[3], vsrcA32x4x2.val[1][2]);
        vsrcC32x4x2.val[1] = vmlaq_n_f32(vsrcC32x4x2.val[1], vsrcB32x4x4.val[3], vsrcA32x4x2.val[1][3]);
    }

    /* A:2x2 B:2x4 C:2x4 */
    if (KHas2)
    {
        float32x4_t vsrcA32x4;
        float32x4x2_t vsrcB32x4x2;
        float *pCurB = pB+KDiv4*16;
        float *pCurA = pA+KDiv4*8;

        vsrcB32x4x2 = vld1q_f32_x2(pCurB);
        vsrcA32x4   = vld1q_f32(pCurA);
        vsrcC32x4x2.val[0] = vmlaq_n_f32(vsrcC32x4x2.val[0], vsrcB32x4x2.val[0], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_16(pCurB+8);
        vsrcC32x4x2.val[1] = vmlaq_n_f32(vsrcC32x4x2.val[1], vsrcB32x4x2.val[0], vsrcA32x4[1]);
        vsrcC32x4x2.val[0] = vmlaq_n_f32(vsrcC32x4x2.val[0], vsrcB32x4x2.val[1], vsrcA32x4[2]);
        vsrcC32x4x2.val[1] = vmlaq_n_f32(vsrcC32x4x2.val[1], vsrcB32x4x2.val[1], vsrcA32x4[3]);
    }

    /* A:2x1 B:1x4 C:2x4 */
    if (KHas1)
    {
        float32x2_t vsrcA32x2;
        float32x4_t vsrcB32x4;
        float *pCurB = pB+(K-1)*4;
        float *pCurA = pA+(K-1)*2;

        vsrcB32x4 = vld1q_f32(pCurB);
        vsrcA32x2 = vld1_f32(pCurA);
        vsrcC32x4x2.val[0] = vmlaq_n_f32(vsrcC32x4x2.val[0], vsrcB32x4, vsrcA32x2[0]);
        vsrcC32x4x2.val[1] = vmlaq_n_f32(vsrcC32x4x2.val[1], vsrcB32x4, vsrcA32x2[1]);
    }

    if (pBasis)
    {
        /* one basis for one channel */
        float32x4_t vbasis32x4_0 = vdupq_n_f32(pBasis[0]);
        float32x4_t vbasis32x4_1 = vdupq_n_f32(pBasis[1]);
        vsrcC32x4x2.val[0] = vaddq_f32(vsrcC32x4x2.val[0], vbasis32x4_0);
        vsrcC32x4x2.val[1] = vaddq_f32(vsrcC32x4x2.val[1], vbasis32x4_1);
    }

    if (bRelu)
    {
        uint32x4_t vmask;
        float32x4_t vzero;
        vmask = veorq_u32(vmask, vmask);
        vzero = vreinterpretq_f32_u32(vmask);

        vmask              = vcleq_f32(vsrcC32x4x2.val[0], vzero);
        vsrcC32x4x2.val[0] = vbslq_f32(vmask, vzero, vsrcC32x4x2.val[0]);

        vmask              = vcleq_f32(vsrcC32x4x2.val[1], vzero);
        vsrcC32x4x2.val[1] = vbslq_f32(vmask, vzero, vsrcC32x4x2.val[1]);
    }
    else if (pPrelu)
    {
        uint32x4_t vmask;
        float32x4_t vscale32x4, vmul, vzero;
        vmask = veorq_u32(vmask, vmask);
        vzero = vreinterpretq_f32_u32(vmask);
        if (bSharedPrelu) /* all channel use same prelu */
            vscale32x4 = vdupq_n_f32(pPrelu[0]);
        else
            vscale32x4 = vld1q_f32(pPrelu);

        vmask              = vcleq_f32(vsrcC32x4x2.val[0], vzero);
        vmul               = vmulq_n_f32(vsrcC32x4x2.val[0], vscale32x4[0]);
        vsrcC32x4x2.val[0] = vbslq_f32(vmask, vmul, vsrcC32x4x2.val[0]);

        vmask              = vcleq_f32(vsrcC32x4x2.val[1], vzero);
        vmul               = vmulq_n_f32(vsrcC32x4x2.val[1], vscale32x4[1]);
        vsrcC32x4x2.val[1] = vbslq_f32(vmask, vmul, vsrcC32x4x2.val[1]);
    }

    vst1q_f32(pC,     vsrcC32x4x2.val[0]);
    vst1q_f32(pC+N,   vsrcC32x4x2.val[1]);
}

static void sgemm1xKx4_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t KDiv4 = K>>2;
    uint32_t KHas2 = (K>>1)&1;
    uint32_t KHas1 = K&1;

    uint32x4_t vzero;
    float32x4_t vsrcC32x4;
    vzero = veorq_u32(vzero, vzero);
    vsrcC32x4 = vreinterpretq_f32_u32(vzero);

    /* A:1x4 B:4x4 C:1x4 */
    for (uint32_t i = 0; i < KDiv4; ++i)
    {
        float *pCurB = pB+i*16;
        float *pCurA = pA+i*4;
        float32x4_t vsrcA32x4;
        float32x4x4_t vsrcB32x4x4;

        vsrcB32x4x4 = vld1q_f32_x4(pCurB);
        vsrcA32x4 = vld1q_f32(pCurA);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcB32x4x4.val[0], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_64(pCurB+16);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcB32x4x4.val[1], vsrcA32x4[1]);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcB32x4x4.val[2], vsrcA32x4[2]);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcB32x4x4.val[3], vsrcA32x4[3]);
    }

    /* A:1x2 B:2x4 C:1x4 */
    if (KHas2)
    {
        float32x2_t vsrcA32x2;
        float32x4x2_t vsrcB32x4x2;
        float *pCurB = pB+KDiv4*16;
        float *pCurA = pA+KDiv4*4;

        vsrcB32x4x2 = vld1q_f32_x2(pCurB);
        vsrcA32x2   = vld1_f32(pCurA);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcB32x4x2.val[0], vsrcA32x2[0]);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcB32x4x2.val[1], vsrcA32x2[1]);
    }

    /* A:1x1 B:1x4 C:1x4 */
    if (KHas1)
    {
        float32x4_t vsrcB32x4;
        float *pCurB = pB+(K-1)*4;
        float *pCurA = pA+(K-1);
        vsrcB32x4 = vld1q_f32(pCurB);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcB32x4, pCurA[0]);

    }

    if (pBasis)
    {
        /* one basis for one channel */
        float32x4_t vbasis32x4 = vdupq_n_f32(pBasis[0]);
        vsrcC32x4 = vaddq_f32(vsrcC32x4, vbasis32x4);
    }

    if (bRelu)
    {
        uint32x4_t vmask;
        float32x4_t vzero;
        vmask = veorq_u32(vmask, vmask);
        vzero = vreinterpretq_f32_u32(vmask);

        vmask     = vcleq_f32(vsrcC32x4, vzero);
        vsrcC32x4 = vbslq_f32(vmask, vzero, vsrcC32x4);
    }
    else if (pPrelu)
    {
        uint32x4_t vmask;
        float32x4_t vmul, vzero;
        vmask = veorq_u32(vmask, vmask);
        vzero = vreinterpretq_f32_u32(vmask);

        vmask     = vcleq_f32(vsrcC32x4, vzero);
        vmul      = vmulq_n_f32(vsrcC32x4, pPrelu[0]);
        vsrcC32x4 = vbslq_f32(vmask, vmul, vsrcC32x4);
    }

    vst1q_f32(pC,     vsrcC32x4);
}

void sgemmMxKx4_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t MDiv4, MHas2, MHas1;
    MDiv4 = M>>2;
    MHas2 = (M>>1)&1;
    MHas1 = M&1;

    for (uint32_t i = 0; i < MDiv4; ++i)
    {
        sgemm4xKx4_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 4*K;
        pC += 4*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 4;
        if (NULL != pBasis)
            pBasis += 4;
    }

    if (MHas2)
    {
        sgemm2xKx4_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 2*K;
        pC += 2*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 2;
        if (NULL != pBasis)
            pBasis += 2;
    }

    if (MHas1)
        sgemm1xKx4_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
}

/* fp32 unit sgemm block is, A:4x4  B:4x2 C:4x2 */
static void sgemm4xKx2_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t KDiv4 = K>>2;
    uint32_t KHas2 = (K>>1)&1;
    uint32_t KHas1 = K&1;

    uint32x2_t vzero;
    float32x2x4_t vsrcC32x2x4;
    vzero = veor_u32(vzero, vzero);
    vsrcC32x2x4.val[0] = vreinterpret_f32_u32(vzero);
    vsrcC32x2x4.val[1] = vsrcC32x2x4.val[0];
    vsrcC32x2x4.val[2] = vsrcC32x2x4.val[0];
    vsrcC32x2x4.val[3] = vsrcC32x2x4.val[1];

    for (uint32_t i = 0; i < KDiv4; ++i)
    {
        float *pCurB = pB+i*8;
        float *pCurA = pA+i*16;
        float32x4x4_t vsrcA32x4x4;
        float32x2x4_t vsrcB32x2x4;

        vsrcB32x2x4 = vld1_f32_x4(pCurB);
        vsrcA32x4x4 = vld1q_f32_x4(pCurA);
        vsrcC32x2x4.val[0] = vmla_n_f32(vsrcC32x2x4.val[0], vsrcB32x2x4.val[0], vsrcA32x4x4.val[0][0]);
        ARM_LOAD_PREFETCH_64(pCurA+16);
        vsrcC32x2x4.val[1] = vmla_n_f32(vsrcC32x2x4.val[1], vsrcB32x2x4.val[0], vsrcA32x4x4.val[0][1]);
        ARM_LOAD_PREFETCH_32(pCurB+8);
        vsrcC32x2x4.val[2] = vmla_n_f32(vsrcC32x2x4.val[2], vsrcB32x2x4.val[0], vsrcA32x4x4.val[0][2]);
        vsrcC32x2x4.val[3] = vmla_n_f32(vsrcC32x2x4.val[3], vsrcB32x2x4.val[0], vsrcA32x4x4.val[0][3]);

        vsrcC32x2x4.val[0] = vmla_n_f32(vsrcC32x2x4.val[0], vsrcB32x2x4.val[1], vsrcA32x4x4.val[1][0]);
        vsrcC32x2x4.val[1] = vmla_n_f32(vsrcC32x2x4.val[1], vsrcB32x2x4.val[1], vsrcA32x4x4.val[1][1]);
        vsrcC32x2x4.val[2] = vmla_n_f32(vsrcC32x2x4.val[2], vsrcB32x2x4.val[1], vsrcA32x4x4.val[1][2]);
        vsrcC32x2x4.val[3] = vmla_n_f32(vsrcC32x2x4.val[3], vsrcB32x2x4.val[1], vsrcA32x4x4.val[1][3]);

        vsrcC32x2x4.val[0] = vmla_n_f32(vsrcC32x2x4.val[0], vsrcB32x2x4.val[2], vsrcA32x4x4.val[2][0]);
        vsrcC32x2x4.val[1] = vmla_n_f32(vsrcC32x2x4.val[1], vsrcB32x2x4.val[2], vsrcA32x4x4.val[2][1]);
        vsrcC32x2x4.val[2] = vmla_n_f32(vsrcC32x2x4.val[2], vsrcB32x2x4.val[2], vsrcA32x4x4.val[2][2]);
        vsrcC32x2x4.val[3] = vmla_n_f32(vsrcC32x2x4.val[3], vsrcB32x2x4.val[2], vsrcA32x4x4.val[2][3]);

        vsrcC32x2x4.val[0] = vmla_n_f32(vsrcC32x2x4.val[0], vsrcB32x2x4.val[3], vsrcA32x4x4.val[3][0]);
        vsrcC32x2x4.val[1] = vmla_n_f32(vsrcC32x2x4.val[1], vsrcB32x2x4.val[3], vsrcA32x4x4.val[3][1]);
        vsrcC32x2x4.val[2] = vmla_n_f32(vsrcC32x2x4.val[2], vsrcB32x2x4.val[3], vsrcA32x4x4.val[3][2]);
        vsrcC32x2x4.val[3] = vmla_n_f32(vsrcC32x2x4.val[3], vsrcB32x2x4.val[3], vsrcA32x4x4.val[3][3]);
    }

    /* A:4x2 B:2x2 C:4x2 */
    if (KHas2)
    {
        float32x4x2_t vsrcA32x4x2;
        float32x2x2_t vsrcB32x2x2;
        float *pCurB = pB+KDiv4*8;
        float *pCurA = pA+KDiv4*16;

        vsrcB32x2x2 = vld1_f32_x2(pCurB);
        vsrcA32x4x2 = vld1q_f32_x2(pCurA);
        vsrcC32x2x4.val[0] = vmla_n_f32(vsrcC32x2x4.val[0], vsrcB32x2x2.val[0], vsrcA32x4x2.val[0][0]);
        ARM_LOAD_PREFETCH_64(pCurA+16);
        vsrcC32x2x4.val[1] = vmla_n_f32(vsrcC32x2x4.val[1], vsrcB32x2x2.val[0], vsrcA32x4x2.val[0][1]);
        ARM_LOAD_PREFETCH_32(pCurB+8);
        vsrcC32x2x4.val[2] = vmla_n_f32(vsrcC32x2x4.val[2], vsrcB32x2x2.val[0], vsrcA32x4x2.val[0][2]);
        vsrcC32x2x4.val[3] = vmla_n_f32(vsrcC32x2x4.val[3], vsrcB32x2x2.val[0], vsrcA32x4x2.val[0][3]);

        vsrcC32x2x4.val[0] = vmla_n_f32(vsrcC32x2x4.val[0], vsrcB32x2x2.val[1], vsrcA32x4x2.val[1][0]);
        vsrcC32x2x4.val[1] = vmla_n_f32(vsrcC32x2x4.val[1], vsrcB32x2x2.val[1], vsrcA32x4x2.val[1][1]);
        vsrcC32x2x4.val[2] = vmla_n_f32(vsrcC32x2x4.val[2], vsrcB32x2x2.val[1], vsrcA32x4x2.val[1][2]);
        vsrcC32x2x4.val[3] = vmla_n_f32(vsrcC32x2x4.val[3], vsrcB32x2x2.val[1], vsrcA32x4x2.val[1][3]);
    }

    /* A:4x1 B:1x2 C:4x2 */
    if (KHas1)
    {
        float32x4_t vsrcA32x4;
        float32x2_t vsrcB32x2;
        float *pCurB = pB+(K-1)*2;
        float *pCurA = pA+(K-1)*4;

        vsrcB32x2 = vld1_f32(pCurB);
        vsrcA32x4 = vld1q_f32(pCurA);
        vsrcC32x2x4.val[0] = vmla_n_f32(vsrcC32x2x4.val[0], vsrcB32x2, vsrcA32x4[0]);
        vsrcC32x2x4.val[1] = vmla_n_f32(vsrcC32x2x4.val[1], vsrcB32x2, vsrcA32x4[1]);
        vsrcC32x2x4.val[2] = vmla_n_f32(vsrcC32x2x4.val[2], vsrcB32x2, vsrcA32x4[2]);
        vsrcC32x2x4.val[3] = vmla_n_f32(vsrcC32x2x4.val[3], vsrcB32x2, vsrcA32x4[3]);
    }

    if (pBasis)
    {
        /* one basis for one channel */
        float32x2_t vbasis32x2_0 = vdup_n_f32(pBasis[0]);
        float32x2_t vbasis32x2_1 = vdup_n_f32(pBasis[1]);
        float32x2_t vbasis32x2_2 = vdup_n_f32(pBasis[2]);
        float32x2_t vbasis32x2_3 = vdup_n_f32(pBasis[3]);
        vsrcC32x2x4.val[0] = vadd_f32(vsrcC32x2x4.val[0], vbasis32x2_0);
        vsrcC32x2x4.val[1] = vadd_f32(vsrcC32x2x4.val[1], vbasis32x2_1);
        vsrcC32x2x4.val[2] = vadd_f32(vsrcC32x2x4.val[2], vbasis32x2_2);
        vsrcC32x2x4.val[3] = vadd_f32(vsrcC32x2x4.val[3], vbasis32x2_3);
    }

    if (bRelu)
    {
        uint32x2_t vmask;
        float32x2_t vzero;
        vmask = veor_u32(vmask, vmask);
        vzero = vreinterpret_f32_u32(vmask);

        vmask              = vcle_f32(vsrcC32x2x4.val[0], vzero);
        vsrcC32x2x4.val[0] = vbsl_f32(vmask, vzero, vsrcC32x2x4.val[0]);

        vmask              = vcle_f32(vsrcC32x2x4.val[0], vzero);
        vsrcC32x2x4.val[1] = vbsl_f32(vmask, vzero, vsrcC32x2x4.val[0]);

        vmask              = vcle_f32(vsrcC32x2x4.val[0], vzero);
        vsrcC32x2x4.val[2] = vbsl_f32(vmask, vzero, vsrcC32x2x4.val[0]);

        vmask              = vcle_f32(vsrcC32x2x4.val[0], vzero);
        vsrcC32x2x4.val[3] = vbsl_f32(vmask, vzero, vsrcC32x2x4.val[0]);
    }
    else if (pPrelu)
    {
        uint32x2_t vmask;
        float32x4_t vscale32x4;
        float32x2_t vmul, vzero;
        vmask = veor_u32(vmask, vmask);
        vzero = vreinterpret_f32_u32(vmask);
        if (bSharedPrelu) /* all channel use same prelu */
            vscale32x4 = vdupq_n_f32(pPrelu[0]);
        else
            vscale32x4 = vld1q_f32(pPrelu);

        vmask              = vcle_f32(vsrcC32x2x4.val[0], vzero);
        vmul               = vmul_n_f32(vsrcC32x2x4.val[0], vscale32x4[0]);
        vsrcC32x2x4.val[0] = vbsl_f32(vmask, vmul, vsrcC32x2x4.val[0]);

        vmask              = vcle_f32(vsrcC32x2x4.val[1], vzero);
        vmul               = vmul_n_f32(vsrcC32x2x4.val[1], vscale32x4[1]);
        vsrcC32x2x4.val[1] = vbsl_f32(vmask, vmul, vsrcC32x2x4.val[1]);

        vmask              = vcle_f32(vsrcC32x2x4.val[2], vzero);
        vmul               = vmul_n_f32(vsrcC32x2x4.val[2], vscale32x4[2]);
        vsrcC32x2x4.val[2] = vbsl_f32(vmask, vmul, vsrcC32x2x4.val[2]);

        vmask              = vcle_f32(vsrcC32x2x4.val[3], vzero);
        vmul               = vmul_n_f32(vsrcC32x2x4.val[3], vscale32x4[3]);
        vsrcC32x2x4.val[3] = vbsl_f32(vmask, vmul, vsrcC32x2x4.val[3]);
    }

    vst1_f32(pC,     vsrcC32x2x4.val[0]);
    vst1_f32(pC+N,   vsrcC32x2x4.val[1]);
    vst1_f32(pC+2*N, vsrcC32x2x4.val[2]);
    vst1_f32(pC+3*N, vsrcC32x2x4.val[3]);
}

static void sgemm2xKx2_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t KDiv4 = K>>2;
    uint32_t KHas2 = (K>>1)&1;
    uint32_t KHas1 = K&1;

    uint32x2_t vzero;
    float32x2x2_t vsrcC32x2x2;
    vzero = veor_u32(vzero, vzero);
    vsrcC32x2x2.val[0] = vreinterpret_f32_u32(vzero);
    vsrcC32x2x2.val[1] = vsrcC32x2x2.val[0];

    /* A:2x4 B:4x2 C:2x2 */
    for (uint32_t i = 0; i < KDiv4; ++i)
    {
        float *pCurB = pB+i*8;
        float *pCurA = pA+i*8;
        float32x2x4_t vsrcA32x2x4;
        float32x2x4_t vsrcB32x2x4;

        vsrcB32x2x4 = vld1_f32_x4(pCurB);
        vsrcA32x2x4 = vld1_f32_x4(pCurA);
        vsrcC32x2x2.val[0] = vmla_n_f32(vsrcC32x2x2.val[0], vsrcB32x2x4.val[0], vsrcA32x2x4.val[0][0]);
        vsrcC32x2x2.val[1] = vmla_n_f32(vsrcC32x2x2.val[1], vsrcB32x2x4.val[0], vsrcA32x2x4.val[0][1]);
        ARM_LOAD_PREFETCH_32(pCurB);
        vsrcC32x2x2.val[0] = vmla_n_f32(vsrcC32x2x2.val[0], vsrcB32x2x4.val[1], vsrcA32x2x4.val[1][0]);
        vsrcC32x2x2.val[1] = vmla_n_f32(vsrcC32x2x2.val[1], vsrcB32x2x4.val[1], vsrcA32x2x4.val[1][1]);
        ARM_LOAD_PREFETCH_32(pCurA);
        vsrcC32x2x2.val[0] = vmla_n_f32(vsrcC32x2x2.val[0], vsrcB32x2x4.val[2], vsrcA32x2x4.val[2][0]);
        vsrcC32x2x2.val[1] = vmla_n_f32(vsrcC32x2x2.val[1], vsrcB32x2x4.val[2], vsrcA32x2x4.val[2][1]);

        vsrcC32x2x2.val[0] = vmla_n_f32(vsrcC32x2x2.val[0], vsrcB32x2x4.val[3], vsrcA32x2x4.val[3][0]);
        vsrcC32x2x2.val[1] = vmla_n_f32(vsrcC32x2x2.val[1], vsrcB32x2x4.val[3], vsrcA32x2x4.val[3][1]);
    }

    /* A:2x2 B:2x2 C:2x2 */
    if (KHas2)
    {
        float32x2x2_t vsrcA32x2x2;
        float32x2x2_t vsrcB32x2x2;
        float *pCurB = pB+KDiv4*8;
        float *pCurA = pA+KDiv4*8;

        vsrcB32x2x2 = vld1_f32_x2(pCurB);
        vsrcA32x2x2 = vld1_f32_x2(pCurA);

        vsrcC32x2x2.val[0] = vmla_n_f32(vsrcC32x2x2.val[0], vsrcB32x2x2.val[0], vsrcA32x2x2.val[0][0]);
        vsrcC32x2x2.val[1] = vmla_n_f32(vsrcC32x2x2.val[1], vsrcB32x2x2.val[0], vsrcA32x2x2.val[0][1]);
        vsrcC32x2x2.val[0] = vmla_n_f32(vsrcC32x2x2.val[0], vsrcB32x2x2.val[1], vsrcA32x2x2.val[1][0]);
        vsrcC32x2x2.val[1] = vmla_n_f32(vsrcC32x2x2.val[1], vsrcB32x2x2.val[1], vsrcA32x2x2.val[1][1]);
    }

    /* A:2x1 B:1x2 C:2x2 */
    if (KHas1)
    {
        float32x2_t vsrcA32x2;
        float32x2_t vsrcB32x2;
        float *pCurB = pB+(K-1)*2;
        float *pCurA = pA+(K-1)*2;

        vsrcB32x2 = vld1_f32(pCurB);
        vsrcA32x2 = vld1_f32(pCurA);
        vsrcC32x2x2.val[0] = vmla_n_f32(vsrcC32x2x2.val[0], vsrcB32x2, vsrcA32x2[0]);
        vsrcC32x2x2.val[1] = vmla_n_f32(vsrcC32x2x2.val[1], vsrcB32x2, vsrcA32x2[1]);
    }

    if (pBasis)
    {
        /* one basis for one channel */
        float32x2_t vbasis32x2_0 = vdup_n_f32(pBasis[0]);
        float32x2_t vbasis32x2_1 = vdup_n_f32(pBasis[1]);
        vsrcC32x2x2.val[0] = vadd_f32(vsrcC32x2x2.val[0], vbasis32x2_0);
        vsrcC32x2x2.val[1] = vadd_f32(vsrcC32x2x2.val[1], vbasis32x2_1);
    }

    if (bRelu)
    {
        uint32x2_t vmask;
        float32x2_t vzero;
        vmask = veor_u32(vmask, vmask);
        vzero = vreinterpret_f32_u32(vmask);

        vmask              = vcle_f32(vsrcC32x2x2.val[0], vzero);
        vsrcC32x2x2.val[0] = vbsl_f32(vmask, vzero, vsrcC32x2x2.val[0]);

        vmask              = vcle_f32(vsrcC32x2x2.val[0], vzero);
        vsrcC32x2x2.val[1] = vbsl_f32(vmask, vzero, vsrcC32x2x2.val[0]);
    }
    else if (pPrelu)
    {
        uint32x2_t vmask;
        float32x2_t vscale32x2, vmul, vzero;
        vmask = veor_u32(vmask, vmask);
        vzero = vreinterpret_f32_u32(vmask);
        if (bSharedPrelu) /* all channel use same prelu */
            vscale32x2 = vdup_n_f32(pPrelu[0]);
        else
            vscale32x2 = vld1_f32(pPrelu);

        vmask              = vcle_f32(vsrcC32x2x2.val[0], vzero);
        vmul               = vmul_n_f32(vsrcC32x2x2.val[0], vscale32x2[0]);
        vsrcC32x2x2.val[0] = vbsl_f32(vmask, vmul, vsrcC32x2x2.val[0]);

        vmask              = vcle_f32(vsrcC32x2x2.val[1], vzero);
        vmul               = vmul_n_f32(vsrcC32x2x2.val[1], vscale32x2[1]);
        vsrcC32x2x2.val[1] = vbsl_f32(vmask, vmul, vsrcC32x2x2.val[1]);
    }

    vst1_f32(pC,     vsrcC32x2x2.val[0]);
    vst1_f32(pC+N,   vsrcC32x2x2.val[1]);
}

static void sgemm1xKx2_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t KDiv4 = K>>2;
    uint32_t KHas2 = (K>>1)&1;
    uint32_t KHas1 = K&1;

    uint32x2_t vzero;
    float32x2_t vsrcC32x2;
    vzero = veor_u32(vzero, vzero);
    vsrcC32x2 = vreinterpret_f32_u32(vzero);

    /* A:1x4 B:4x2 C:1x2 */
    for (uint32_t i = 0; i < KDiv4; ++i)
    {
        float *pCurB = pB+i*8;
        float *pCurA = pA+i*4;
        float32x4_t vsrcA32x4;
        float32x2x4_t vsrcB32x2x4;

        vsrcB32x2x4 = vld1_f32_x4(pCurB);
        vsrcA32x4   = vld1q_f32(pCurA);

        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcB32x2x4.val[0], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_32(pCurB);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcB32x2x4.val[1], vsrcA32x4[1]);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcB32x2x4.val[2], vsrcA32x4[2]);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcB32x2x4.val[3], vsrcA32x4[3]);
    }

    /* A:1x2 B:2x2 C:1x2 */
    if (KHas2)
    {
        float32x2_t vsrcA32x2;
        float32x2x2_t vsrcB32x2x2;
        float *pCurB = pB+KDiv4*8;
        float *pCurA = pA+KDiv4*8;

        vsrcB32x2x2 = vld1_f32_x2(pCurB);
        vsrcA32x2 = vld1_f32(pCurA);

        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcB32x2x2.val[0], vsrcA32x2[0]);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcB32x2x2.val[1], vsrcA32x2[1]);
    }

    /* A:1x1 B:1x2 C:1x2 */
    if (KHas1)
    {
        float32x2_t vsrcB32x2;
        float *pCurB = pB+(K-1)*2;
        float *pCurA = pA+(K-1);

        vsrcB32x2 = vld1_f32(pCurB);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcB32x2, pCurA[0]);
    }

    if (pBasis)
    {
        /* one basis for one channel */
        float32x2_t vbasis32x2 = vdup_n_f32(pBasis[0]);
        vsrcC32x2 = vadd_f32(vsrcC32x2, vbasis32x2);
    }

    if (bRelu)
    {
        uint32x2_t vmask;
        float32x2_t vzero;
        vmask = veor_u32(vmask, vmask);
        vzero = vreinterpret_f32_u32(vmask);

        vmask     = vcle_f32(vsrcC32x2, vzero);
        vsrcC32x2 = vbsl_f32(vmask, vzero, vsrcC32x2);
    }
    else if (pPrelu)
    {
        uint32x2_t vmask;
        float32x2_t vmul, vzero;
        vmask = veor_u32(vmask, vmask);
        vzero = vreinterpret_f32_u32(vmask);

        vmask     = vcle_f32(vsrcC32x2, vzero);
        vmul      = vmul_n_f32(vsrcC32x2, pPrelu[0]);
        vsrcC32x2 = vbsl_f32(vmask, vmul, vsrcC32x2);
    }

    vst1_f32(pC,     vsrcC32x2);
}

void sgemmMxKx2_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t MDiv4, MHas2, MHas1;
    MDiv4 = M>>2;
    MHas2 = (M>>1)&1;
    MHas1 = M&1;

    for (uint32_t i = 0; i < MDiv4; ++i)
    {
        sgemm4xKx2_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 4*K;
        pC += 4*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 4;
        if (NULL != pBasis)
            pBasis += 4;
    }

    if (MHas2)
    {
        sgemm2xKx2_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 2*K;
        pC += 2*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 2;
        if (NULL != pBasis)
            pBasis += 2;
    }

    if (MHas1)
        sgemm1xKx2_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
}

/* fp32 unit sgemm block is, A:4x4  B:4x1 C:4x1 */
static void sgemm4xKx1_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t KDiv4 = K>>2;
    uint32_t KHas2 = (K>>1)&1;
    uint32_t KHas1 = K&1;

    uint32x4_t vzero;
    float32x4_t vsrcC32x4;
    vzero = veorq_u32(vzero, vzero);
    vsrcC32x4 = vreinterpretq_f32_u32(vzero);

    /* A:4x4 B:4x1 C:4x1 */
    for (uint32_t i = 0; i < KDiv4; ++i)
    {
        float *pCurB = pB+i*4;
        float *pCurA = pA+i*16;
        float32x4x4_t vsrcA32x4x4;
        float32x4_t vsrcB32x4;

        vsrcB32x4 = vld1q_f32(pCurB);
        vsrcA32x4x4 = vld1q_f32_x4(pCurA);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcA32x4x4.val[0], vsrcB32x4[0]);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcA32x4x4.val[1], vsrcB32x4[1]);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcA32x4x4.val[2], vsrcB32x4[2]);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcA32x4x4.val[3], vsrcB32x4[3]);
    }

    /* A:4x2 B:2x1 C:4x1 */
    if (KHas2)
    {
        float32x4x2_t vsrcA32x4x2;
        float32x2_t vsrcB32x2;
        float *pCurB = pB+KDiv4*8;
        float *pCurA = pA+KDiv4*16;

        vsrcB32x2 = vld1_f32(pCurB);
        vsrcA32x4x2 = vld1q_f32_x2(pCurA);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcA32x4x2.val[0], vsrcB32x2[0]);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcA32x4x2.val[1], vsrcB32x2[1]);
    }

    /* A:4x1 B:1x1 C:4x1 */
    if (KHas1)
    {
        float32x4_t vsrcA32x4;
        float *pCurB = pB+(K-1);
        float *pCurA = pA+(K-1)*4;

        vsrcA32x4 = vld1q_f32(pCurA);
        vsrcC32x4 = vmlaq_n_f32(vsrcC32x4, vsrcA32x4, *pCurB);
    }

    if (pBasis)
    {
        /* one basis for one channel */
        float32x4_t vbasis32x4 = vld1q_f32(pBasis);
        vsrcC32x4 = vaddq_f32(vsrcC32x4, vbasis32x4);
    }

    if (bRelu)
    {
        uint32x4_t vmask;
        float32x4_t vzero;
        vmask = veorq_u32(vmask, vmask);
        vzero = vreinterpretq_f32_u32(vmask);

        vmask     = vcleq_f32(vsrcC32x4, vzero);
        vsrcC32x4 = vbslq_f32(vmask, vzero, vsrcC32x4);
    }
    else if (pPrelu)
    {
        if (vsrcC32x4[0] < 0)
            vsrcC32x4[0] *= pPrelu[0];
        if (vsrcC32x4[1] < 0)
            vsrcC32x4[1] *= pPrelu[1];
        if (vsrcC32x4[2] < 0)
            vsrcC32x4[2] *= pPrelu[2];
        if (vsrcC32x4[3] < 0)
            vsrcC32x4[3] *= pPrelu[3];
    }

    *(pC + 0*N) = vsrcC32x4[0];
    *(pC + 1*N) = vsrcC32x4[1];
    *(pC + 2*N) = vsrcC32x4[2];
    *(pC + 3*N) = vsrcC32x4[3];
}

static void sgemm2xKx1_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t KDiv4 = K>>2;
    uint32_t KHas2 = (K>>1)&1;
    uint32_t KHas1 = K&1;

    uint32x2_t vzero;
    float32x2_t vsrcC32x2;
    vzero = veor_u32(vzero, vzero);
    vsrcC32x2 = vreinterpret_f32_u32(vzero);

    /* A:2x4 B:4x1 C:2x1 */
    for (uint32_t i = 0; i < KDiv4; ++i)
    {
        float *pCurB = pB+i*4;
        float *pCurA = pA+i*8;
        float32x2x4_t vsrcA32x2x4;
        float32x4_t vsrcB32x4;

        vsrcB32x4 = vld1q_f32(pCurB);
        vsrcA32x2x4 = vld1_f32_x4(pCurA);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcA32x2x4.val[0], vsrcB32x4[0]);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcA32x2x4.val[1], vsrcB32x4[1]);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcA32x2x4.val[2], vsrcB32x4[2]);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcA32x2x4.val[3], vsrcB32x4[3]);
    }

    /* A:2x2 B:2x1 C:2x1 */
    if (KHas2)
    {
        float32x2x2_t vsrcA32x2x2;
        float *pCurB = pB+KDiv4*4;
        float *pCurA = pA+KDiv4*8;

        vsrcA32x2x2 = vld1_f32_x2(pCurA);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcA32x2x2.val[0], pCurB[0]);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcA32x2x2.val[1], pCurB[1]);
    }

    /* A:2x1 B:1x1 C:2x1 */
    if (KHas1)
    {
        float32x2_t vsrcA32x2;
        float *pCurB = pB+(K-1);
        float *pCurA = pA+(K-1)*2;

        vsrcA32x2 = vld1_f32(pCurA);
        vsrcC32x2 = vmla_n_f32(vsrcC32x2, vsrcA32x2, pCurB[0]);
    }

    if (pBasis)
    {
        /* one basis for one channel */
        float32x2_t vbasis32x2 = vld1_f32(pBasis);
        vsrcC32x2 = vadd_f32(vsrcC32x2, vbasis32x2);
    }

    if (bRelu)
    {
        uint32x2_t vmask;
        float32x2_t vzero;
        vmask = veor_u32(vmask, vmask);
        vzero = vreinterpret_f32_u32(vmask);

        vmask     = vcle_f32(vsrcC32x2, vzero);
        vsrcC32x2 = vbsl_f32(vmask, vzero, vsrcC32x2);
    }
    else if (pPrelu)
    {
        if (vsrcC32x2[0] < 0)
            vsrcC32x2[0] *= pPrelu[0];
        if (vsrcC32x2[1] < 0)
            vsrcC32x2[1] *= pPrelu[1];
    }

    *(pC + 0*N) = vsrcC32x2[0];
    *(pC + 1*N) = vsrcC32x2[1];
}

static void sgemm1xKx1_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t KDiv4 = K>>2;
    uint32_t KHas2 = (K>>1)&1;
    uint32_t KHas1 = K&1;

    uint32x4_t vzero;
    float32x4_t vsrcC32x4;
    vzero = veorq_u32(vzero, vzero);
    vsrcC32x4 = vreinterpretq_f32_u32(vzero);

    /* A:1x4 B:4x1 C:1x1 */
    for (uint32_t i = 0; i < KDiv4; ++i)
    {
        float *pCurB = pB+i*4;
        float *pCurA = pA+i*4;
        float32x4_t vsrcA32x4;
        float32x4_t vsrcB32x4;

        vsrcB32x4 = vld1q_f32(pCurB);
        vsrcA32x4 = vld1q_f32(pCurA);
        vsrcC32x4 = vmlaq_f32(vsrcC32x4, vsrcA32x4, vsrcB32x4);
    }
    *pC = vsrcC32x4[0] + vsrcC32x4[1] + vsrcC32x4[2] + vsrcC32x4[3];

    /* A:1x2 B:2x1 C:1x1 */
    if (KHas2)
    {
        float *pCurB = pB+KDiv4*4;
        float *pCurA = pA+KDiv4*4;
        *pC += pCurA[0]*pCurB[0];
        *pC += pCurA[1]*pCurB[1];
    }

    /* A:1x1 B:1x1 C:1x1 */
    if (KHas1)
    {
        float *pCurB = pB+(K-1);
        float *pCurA = pA+(K-1);
        *pC += pCurA[0]*pCurB[0];
    }

    if (pBasis)
        *pC += pBasis[0];

    if (bRelu)
    {
        if (*pC < 0)
            *pC = 0;
    }
    else if (pPrelu)
    {
        if (*pC < 0)
            *pC *= pPrelu[0];
    }
}

void sgemmMxKx1_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bSharedPrelu, float *pBasis)
{
    uint32_t MDiv4, MHas2, MHas1;
    MDiv4 = M>>2;
    MHas2 = (M>>1)&1;
    MHas1 = M&1;

    for (uint32_t i = 0; i < MDiv4; ++i)
    {
        sgemm4xKx1_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 4*K;
        pC += 4*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 4;
        if (NULL != pBasis)
            pBasis += 4;
    }

    if (MHas2)
    {
        sgemm2xKx1_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
        pA += 2*K;
        pC += 2*N;
        if ((NULL != pPrelu) && (!bSharedPrelu))
            pPrelu += 2;
        if (NULL != pBasis)
            pBasis += 2;
    }

    if (MHas1)
        sgemm1xKx1_fp32(pA, pB, pC, K, N, bRelu, pPrelu, bSharedPrelu, pBasis);
}
