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
        pSrcStart = pA + j*4*K;
        pDstStart = pPackA + j*4*K;

        for (i = 0; i < KDiv4; ++i)
        {
            float32x4x4_t vsrc_32x4x4;
            vsrc_32x4x4.val[0][0] = *(pSrcStart+0*K);
            vsrc_32x4x4.val[0][1] = *(pSrcStart+1*K);
            vsrc_32x4x4.val[0][2] = *(pSrcStart+2*K);
            vsrc_32x4x4.val[0][3] = *(pSrcStart+3*K);
            pSrcStart++;

            vsrc_32x4x4.val[1][0] = *(pSrcStart+0*K);
            vsrc_32x4x4.val[1][1] = *(pSrcStart+1*K);
            vsrc_32x4x4.val[1][2] = *(pSrcStart+2*K);
            vsrc_32x4x4.val[1][3] = *(pSrcStart+3*K);
            pSrcStart++;

            vsrc_32x4x4.val[2][0] = *(pSrcStart+0*K);
            vsrc_32x4x4.val[2][1] = *(pSrcStart+1*K);
            vsrc_32x4x4.val[2][2] = *(pSrcStart+2*K);
            vsrc_32x4x4.val[2][3] = *(pSrcStart+3*K);
            pSrcStart++;

            vsrc_32x4x4.val[3][0] = *(pSrcStart+0*K);
            vsrc_32x4x4.val[3][1] = *(pSrcStart+1*K);
            vsrc_32x4x4.val[3][2] = *(pSrcStart+2*K);
            vsrc_32x4x4.val[3][3] = *(pSrcStart+3*K);
            pSrcStart++;
            vst1q_f32_x4(pDstStart, vsrc_32x4x4);
            pDstStart += 16;
        }

        if(KHas2)
        {
            float32x4x2_t vsrc_32x4x2;
            vsrc_32x4x2.val[0][0] = *(pSrcStart+0*K);
            vsrc_32x4x2.val[0][1] = *(pSrcStart+1*K);
            vsrc_32x4x2.val[0][2] = *(pSrcStart+2*K);
            vsrc_32x4x2.val[0][3] = *(pSrcStart+3*K);
            pSrcStart++;

            vsrc_32x4x2.val[1][0] = *(pSrcStart+0*K);
            vsrc_32x4x2.val[1][1] = *(pSrcStart+1*K);
            vsrc_32x4x2.val[1][2] = *(pSrcStart+2*K);
            vsrc_32x4x2.val[1][3] = *(pSrcStart+3*K);
            pSrcStart++;
            vst1q_f32_x2(pDstStart, vsrc_32x4x2);
            pDstStart += 8;
        }

        if(KHas1)
        {
            float32x4_t vsrc_32x4;
            vsrc_32x4[0] = *(pSrcStart+0*K);
            vsrc_32x4[1] = *(pSrcStart+1*K);
            vsrc_32x4[2] = *(pSrcStart+2*K);
            vsrc_32x4[3] = *(pSrcStart+3*K);
            vst1q_f32(pDstStart, vsrc_32x4);
        }
    }

    if(MHas2)
    {
        pSrcStart = pA + MDiv4*4*K;
        pDstStart = pPackA + MDiv4*4*K;

        for (i = 0; i < KDiv4; ++i)
        {
            float32x2x4_t vsrc_32x2x4;
            vsrc_32x2x4.val[0][0] = *(pSrcStart+0*K);
            vsrc_32x2x4.val[0][1] = *(pSrcStart+1*K);
            pSrcStart++;

            vsrc_32x2x4.val[1][0] = *(pSrcStart+0*K);
            vsrc_32x2x4.val[1][1] = *(pSrcStart+1*K);
            pSrcStart++;

            vsrc_32x2x4.val[2][0] = *(pSrcStart+0*K);
            vsrc_32x2x4.val[2][1] = *(pSrcStart+1*K);
            pSrcStart++;

            vsrc_32x2x4.val[3][0] = *(pSrcStart+0*K);
            vsrc_32x2x4.val[3][1] = *(pSrcStart+1*K);
            pSrcStart++;

            vst1_f32_x4(pDstStart, vsrc_32x2x4);
            pDstStart += 8;
        }

        if(KHas2)
        {
            float32x2x2_t vsrc_32x2x2;
            vsrc_32x2x2.val[0][0] = *(pSrcStart+0*K);
            vsrc_32x2x2.val[0][1] = *(pSrcStart+1*K);
            pSrcStart++;

            vsrc_32x2x2.val[1][0] = *(pSrcStart+0*K);
            vsrc_32x2x2.val[1][1] = *(pSrcStart+1*K);
            pSrcStart++;
            vst1_f32_x2(pDstStart, vsrc_32x2x2);
            pDstStart += 4;
        }

        if(KHas1)
        {
            float32x2_t vsrc_32x2;
            vsrc_32x2[0] = *(pSrcStart+0*K);
            vsrc_32x2[1] = *(pSrcStart+1*K);
            vst1_f32(pDstStart, vsrc_32x2);
        }
    }

    if (MHas1)
        memcpy(pPackA + (M-1)*K, pA + (M-1)*K, K*sizeof(*pA));
}

void tinySgemmConvPackB4x12_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    uint32_t i = 0, j = 0;
    uint32_t KDiv4, KHas2, KHas1;
    float *pSrcStart, *pDstStart;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;
    pSrcStart = pB;
    pDstStart = pPackB;

    for (i = 0; i < KDiv4; ++i)
    {
        float32x4x3_t vsrc_32x4x3_0, vsrc_32x4x3_1, vsrc_32x4x3_2, vsrc_32x4x3_3;

        vsrc_32x4x3_0 = vld1q_f32_x3(pSrcStart+0*N);
        vst1q_f32_x3(pDstStart, vsrc_32x4x3_0);
        vsrc_32x4x3_1 = vld1q_f32_x3(pSrcStart+1*N);
        vst1q_f32_x3(pDstStart + 12, vsrc_32x4x3_1);
        vsrc_32x4x3_2 = vld1q_f32_x3(pSrcStart+2*N);
        vst1q_f32_x3(pDstStart + 24, vsrc_32x4x3_2);
        vsrc_32x4x3_3 = vld1q_f32_x3(pSrcStart+3*N);
        vst1q_f32_x3(pDstStart + 36, vsrc_32x4x3_3);
        pDstStart += 48;
        pSrcStart += 4*N;
    }

    if (KHas2)
    {
        float32x4x3_t vsrc_32x4x3_0, vsrc_32x4x3_1;
        vsrc_32x4x3_0 = vld1q_f32_x3(pSrcStart+0*N);
        vst1q_f32_x3(pDstStart, vsrc_32x4x3_0);
        vsrc_32x4x3_1 = vld1q_f32_x3(pSrcStart+1*N);
        vst1q_f32_x3(pDstStart + 12, vsrc_32x4x3_1);
        pDstStart += 24;
        pSrcStart += 2*N;
    }

    if (KHas1)
    {
        float32x4x3_t vsrc_32x4x3 = vld1q_f32_x3(pSrcStart+0*N);
        vst1q_f32_x3(pDstStart, vsrc_32x4x3);
    }
}

void tinySgemmConvPackB4x4_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    uint32_t i = 0, j = 0;
    uint32_t KDiv4, KHas2, KHas1;
    float *pSrcStart, *pDstStart;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;
    pSrcStart = pB;
    pDstStart = pPackB;

    for (i = 0; i < KDiv4; ++i)
    {
        float32x4x4_t vsrc_32x4x4;
        vsrc_32x4x4.val[0] = vld1q_f32(pSrcStart+0*N);
        vsrc_32x4x4.val[1] = vld1q_f32(pSrcStart+1*N);
        vsrc_32x4x4.val[2] = vld1q_f32(pSrcStart+2*N);
        vsrc_32x4x4.val[3] = vld1q_f32(pSrcStart+3*N);
        vst1q_f32_x4(pDstStart, vsrc_32x4x4);

        pDstStart += 16;
        pSrcStart += 4*N;
    }

    if (KHas2)
    {
        float32x4x2_t vsrc_32x4x2;
        vsrc_32x4x2.val[0] = vld1q_f32(pSrcStart+0*N);
        vsrc_32x4x2.val[1] = vld1q_f32(pSrcStart+1*N);
        vst1q_f32_x2(pDstStart, vsrc_32x4x2);

        pDstStart += 8;
        pSrcStart += 2*N;
    }

    if (KHas1)
    {
        float32x4_t vsrc_32x4 = vld1q_f32(pSrcStart+0*N);
        vst1q_f32(pDstStart, vsrc_32x4);
    }
}

void tinySgemmConvPackB4x2_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    uint32_t i = 0, j = 0;
    uint32_t KDiv4, KHas2, KHas1;
    float *pSrcStart, *pDstStart;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;
    pSrcStart = pB;
    pDstStart = pPackB;

    for (i = 0; i < KDiv4; ++i)
    {
        float32x2x4_t vsrc_32x2x4;
        vsrc_32x2x4.val[0] = vld1_f32(pSrcStart+0*N);
        vsrc_32x2x4.val[1] = vld1_f32(pSrcStart+1*N);
        vsrc_32x2x4.val[2] = vld1_f32(pSrcStart+2*N);
        vsrc_32x2x4.val[3] = vld1_f32(pSrcStart+3*N);
        vst1_f32_x4(pDstStart, vsrc_32x2x4);

        pDstStart += 8;
        pSrcStart += 4*N;
    }

    if (KHas2)
    {
        float32x2x2_t vsrc_32x2x2;
        vsrc_32x2x2.val[0] = vld1_f32(pSrcStart+0*N);
        vsrc_32x2x2.val[1] = vld1_f32(pSrcStart+1*N);
        vst1_f32_x2(pDstStart, vsrc_32x2x2);

        pDstStart += 4;
        pSrcStart += 2*N;
    }

    if (KHas1)
    {
        float32x2_t vsrc_32x2 = vld1_f32(pSrcStart+0*N);
        vst1_f32(pDstStart, vsrc_32x2);
    }
}

void tinySgemmConvPackB4x12_fp32_fp32(float *pB, float *pPackB, uint32_t K, uint32_t N)
{
    uint32_t i = 0;
    uint32_t NDiv12, leftN, leftNDiv4, leftNHas2, leftNHas1;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    NDiv12 = N/12;
    leftN  = N%12;
    leftNDiv4 = leftN>>2;
    leftNHas2 = (leftN>>1)&1;
    leftNHas1 = leftN&1;

    for (i = 0; i < NDiv12; ++i)
        tinySgemmConvPackB4x12_fp32_fp32_unit(pB + i*12, pPackB + K*12, K, N);

    for (i = 0; i < leftNDiv4; ++i)
        tinySgemmConvPackB4x4_fp32_fp32_unit(pB + NDiv12*12 + i*4, pPackB + NDiv12*K*12 + i*K*4, K, N);

    if (leftNHas2)
        tinySgemmConvPackB4x2_fp32_fp32_unit(pB + NDiv12*12 + leftNDiv4*4, pPackB + NDiv12*K*12 + leftNDiv4*K*4, K, N);

    if (leftNHas1)
    {
        float *pSrc = pB+K-1;
        float *pDst = pPackB+K*(N-1);
        for (i = 0; i < K; ++i)
            *pDst++ = *(pSrc + i*N);
    }
}