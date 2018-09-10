#include <stdio.h>
#include <string.h>
#include "armNeon.h"
#include "common.h"
#include "pack.h"

void tinySgemmConvPackA4x4_fp32_fp32(float *pA, float *pPackA, uint32_t M, uint32_t K)
{
    uint32_t i = 0, j = 0;
    uint32_t MDiv4, MHas2, MHas1, KDiv4, KHas2, KHas1, alignK;
    float *pSrcStart, *pDstStart;
    
    POINTER_CHECK_NO_RET(pA);
    POINTER_CHECK_NO_RET(pPackA);

    MDiv4 = M>>2; MHas2 = (M>>1)&1; MHas1 = M&1;
    KDiv4 = K>>2; KHas2 = (K>>1)&1; KHas1 = K&1;
    alignK = alignSize(K, 4);

    for (j = 0; j < MDiv4; ++j)
    {
        pSrcStart = pA + j*4*K;
        pDstStart = pPackA + j*4*alignK;

        for (i = 0; i < KDiv4; ++i)
        {
            float32x4x4_t vsrc_32x4x4;
            vsrc_32x4x4.val[0] = vld1q_f32(pA + i*4);
            vsrc_32x4x4.val[1] = vld1q_f32(pA + i*4 + K);
            vsrc_32x4x4.val[2] = vld1q_f32(pA + i*4 + 2*K);
            vsrc_32x4x4.val[3] = vld1q_f32(pA + i*4 + 3*K);
            vst1q_f32_x4(pPackA + i*16, vsrc_32x4x4);
        }

        pSrcStart += KDiv4*4;
        pDstStart += KDiv4*16;

        if(KHas2)
        {
            float32x2x4_t vsrc_32x2x4;
            vsrc_32x2x4.val[0] = vld1_f32(pSrcStart);
            vsrc_32x2x4.val[1] = vld1_f32(pSrcStart + K);
            vsrc_32x2x4.val[2] = vld1_f32(pSrcStart + 2*K);
            vsrc_32x2x4.val[3] = vld1_f32(pSrcStart + 3*K);
            vst1_f32_x4(pDstStart, vsrc_32x2x4);

            pSrcStart += 2;
            pDstStart += 8;
        }

        if(KHas1)
        {
            pDstStart[0] = *pSrcStart;
            pDstStart[1] = *(pSrcStart + K);
            pDstStart[2] = *(pSrcStart + 2*K);
            pDstStart[3] = *(pSrcStart + 3*K);
        }
    }

    if(MHas2)
    {
        pSrcStart = pA + MDiv4*4*K;
        pDstStart = pPackA + MDiv4*4*alignK;

        for( i = 0; i < KDiv4; i++)
        {
            float32x4x2_t vsrc_32x4x2;
            vsrc_32x4x2.val[0] = vld1q_f32(pSrcStart + i*4);
            vsrc_32x4x2.val[1] = vld1q_f32(pSrcStart + i*4 + K);
            vst1q_f32_x2(pDstStart + i*8, vsrc_32x4x2);
        }

        pSrcStart += KDiv4*4;
        pDstStart += KDiv4*8;

        if(KHas2)
        {
            *(pDstStart+0) = *(pSrcStart);
            *(pDstStart+1) = *(pSrcStart + 1);
            *(pDstStart+2) = *(pSrcStart + K);
            *(pDstStart+3) = *(pSrcStart + K + 1);

            pSrcStart += 2;
            pDstStart += 4;
        }

        if(KHas1)
        {
            *(pDstStart+0) = *(pSrcStart);
            *(pDstStart+1) = *(pSrcStart + K);
        }
    }

    if (MHas1)
        memcpy(pPackA + (M-1)*alignK, pA + (M-1)*K, K*sizeof(*pA));
}

void tinySgemmConvPackA4x4_fp32_fp16(float *pA, short *pPackA, uint32_t M, uint32_t K)
{
    uint32_t i = 0, j = 0;
    uint32_t MDiv4, MHas2, MHas1, KDiv4, KHas2, KHas1, alignK;
    float *pSrcStart;
    short *pDstStart;

    POINTER_CHECK_NO_RET(pA);
    POINTER_CHECK_NO_RET(pPackA);

    MDiv4 = M>>2; MHas2 = (M>>1)&1; MHas1 = M&1;
    KDiv4 = K>>2; KHas2 = (K>>1)&1; KHas1 = K&1;
    alignK = alignSize(K*sizeof(*pPackA), MALLOC_MEM_ALIGN)/sizeof(*pPackA);

    for (j = 0; j < MDiv4; ++j)
    {
        pSrcStart = pA + j*4*K;
        pDstStart = pPackA + j*4*alignK;

        for (i = 0; i < KDiv4; ++i)
        {
            float32x4x4_t vsrc_32x4x4;
            vsrc_32x4x4.val[0] = vld1q_f32(pA + i*4);
            vsrc_32x4x4.val[1] = vld1q_f32(pA + i*4 + K);
            vsrc_32x4x4.val[2] = vld1q_f32(pA + i*4 + 2*K);
            vsrc_32x4x4.val[3] = vld1q_f32(pA + i*4 + 3*K);
            vst1q_f16_f32_x4((void*)(pPackA + i*16), &vsrc_32x4x4);
        }

        pSrcStart += KDiv4*4;
        pDstStart += KDiv4*16;

        if(KHas2)
        {
            float32x2x4_t vsrc_32x2x4;
            vsrc_32x2x4.val[0] = vld1_f32(pSrcStart);
            vsrc_32x2x4.val[1] = vld1_f32(pSrcStart + K);
            vsrc_32x2x4.val[2] = vld1_f32(pSrcStart + 2*K);
            vsrc_32x2x4.val[3] = vld1_f32(pSrcStart + 3*K);
            vst1q_f16_f32_x2((void*)(pDstStart), (float32x4x2_t *)&vsrc_32x2x4);

            pSrcStart += 2;
            pDstStart += 8;
        }

        if(KHas1)
        {
            float32x4_t vsrc_32x4;
            vsrc_32x4[0] = *pSrcStart;
            vsrc_32x4[1] = *(pSrcStart + K);
            vsrc_32x4[2] = *(pSrcStart + 2*K);
            vsrc_32x4[3] = *(pSrcStart + 3*K);
            vst1q_f16_f32((void*)pDstStart, vsrc_32x4);
        }
    }

    if(MHas2)
    {
        pSrcStart = pA + MDiv4*4*K;
        pDstStart = pPackA + MDiv4*4*alignK;

        for( i = 0; i < KDiv4; i++)
        {
            float32x4x2_t vsrc_32x4x2;
            vsrc_32x4x2.val[0] = vld1q_f32(pSrcStart + i*4);
            vsrc_32x4x2.val[1] = vld1q_f32(pSrcStart + i*4 + K);
            vst1q_f16_f32_x2((void*)(pDstStart + i*8), &vsrc_32x4x2);
        }

        pSrcStart += KDiv4*4;
        pDstStart += KDiv4*8;

        if(KHas2)
        {
            float32x4_t vsrc_32x4;
            vsrc_32x4[0] = *(pSrcStart);
            vsrc_32x4[1] = *(pSrcStart + 1);
            vsrc_32x4[2] = *(pSrcStart + K);
            vsrc_32x4[3] = *(pSrcStart + K + 1);
            vst1q_f16_f32((void*)pDstStart, vsrc_32x4);

            pSrcStart += 2;
            pDstStart += 4;
        }

        if(KHas1)
        {
            float32x4_t vsrc_32x4;
            vsrc_32x4[0] = *pSrcStart;
            vsrc_32x4[1] = *(pSrcStart + K);
            vsrc_32x4[2] = .0f; /* pad */
            vsrc_32x4[3] = .0f; /* pad */
            vst1q_f16_f32((void*)pDstStart, vsrc_32x4);
        }
    }

    if (MHas1)
    {
        float *pSrcStart = pA + (M-1)*K;
        short *pDstStart = pPackA + (M-1)*alignK;
        float32x4_t vsrc_32x4;

        for( i = 0; i < KDiv4; i++)
        {
            vsrc_32x4 = vld1q_f32(pSrcStart + i*4);
            vst1q_f16_f32((void*)(pDstStart + i*4), vsrc_32x4);
        }

        if (KHas2)
        {
            vsrc_32x4[0] = *pSrcStart;
            vsrc_32x4[1] = *(pSrcStart + K);
            vsrc_32x4[2] = .0f; /* pad */
            vsrc_32x4[3] = .0f; /* pad */
            vst1q_f16_f32((void*)pDstStart, vsrc_32x4);
        }
        else if (KHas1) /* must be else if do not change */
        {
            vsrc_32x4[0] = *pSrcStart;
            vsrc_32x4[1] = .0f; /* pad */
            vsrc_32x4[2] = .0f; /* pad */
            vsrc_32x4[3] = .0f; /* pad */
            vst1q_f16_f32((void*)pDstStart, vsrc_32x4);
        }
    }
}

void tinySgemmConvPackA8x4_fp32_fp32(void *pA, void *pPackA, uint32_t M, uint32_t K)
{
    uint32_t i = 0, j = 0;
    uint32_t MDiv8, MHas4, MHas2, MHas1, KDiv4, KHas2, KHas1;

    MDiv8 = M>>3; MHas4 = (M>>2)&1; MHas2 = (M>>1)&1; MHas1 = M&1;
    KDiv4 = K>>2; KHas2 = (K>>1)&1; KHas1 = K&1;

    POINTER_CHECK_NO_RET(pA);
    POINTER_CHECK_NO_RET(pPackA);

    for (j = 0; j < MDiv8; ++j)
    {
        for (i = 0; i < KDiv4; ++i)
        {

        }
    }
}