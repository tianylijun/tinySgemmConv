#include <stdio.h>
#include <stdlib.h>
#include "armNeon.h"
#include "config.h"
#include "tinySgemmConv.h"
#include "innerTinySgemmConv.h"

void packA(float *pA, float *pPackA, uint32_t M, uint32_t K, enum TINY_SGEMM_CONV_DATA_MODE mode)
{
    uint32_t numBlockM = M / TINY_SGEMM_BLOCK_M;
    uint32_t numBlockK = K / TINY_SGEMM_BLOCK_K;
    uint32_t leftM     = M % TINY_SGEMM_BLOCK_M;
    uint32_t leftK     = K % TINY_SGEMM_BLOCK_K;
    uint32_t MBlocks   = numBlockM + ((0 == leftM)?(0):(1));
    uint32_t KBlocks   = numBlockK + ((0 == leftK)?(0):(1));
    struct rangeInfo *pARangeLoop, *pARange;
    uint32_t dataTypeSize;

    if (TINY_SGEMM_CONV_DATA_MODE_A_FIX16_FIX16_B_FP32 == mode)
        dataTypeSize = sizeof(uint16_t);
    else if (TINY_SGEMM_CONV_DATA_MODE_A_FIX8_FIX8_B_FP32 != mode)
        dataTypeSize = sizeof(float);
    else
        dataTypeSize = sizeof(uint8_t);

    pARange = (struct rangeInfo *)calloc(MBlocks*KBlocks, sizeof(struct rangeInfo));
    if (NULL == pARange)
    {
        printf("%s at line: %d %d %d\n", "no memory", __LINE__, MBlocks, KBlocks);
        return;
    }
    pARangeLoop = pARange;

    for (uint32_t i = 0; i < MBlocks; ++i)
    {
        uint32_t blockH = TINY_SGEMM_BLOCK_M;
        uint8_t *pACur = (uint8_t*)pA + i*TINY_SGEMM_BLOCK_M*K*dataTypeSize;
        if (M < (i+1)*TINY_SGEMM_BLOCK_M)
        {
            if (0 == i) /* first */
            {
                pACur  = (uint8_t*)pA;
                blockH = M;
            }
            else        /* last */
            {
                pACur  = (uint8_t*)pA + (M - leftM)*K*dataTypeSize;
                blockH = leftM;
            }
        }

        for (uint32_t j = 0; j < KBlocks; ++j)
        {
            pARangeLoop->pStart  = (uint8_t*)pACur + j*TINY_SGEMM_BLOCK_K*dataTypeSize;
            pARangeLoop->XBlocks = KBlocks;
            pARangeLoop->YBlocks = MBlocks;
            pARangeLoop->width   = TINY_SGEMM_BLOCK_K;
            if (K < (j+1)*TINY_SGEMM_BLOCK_K)
            {
                if (0 == j) /* first */
                    pARangeLoop->width = K;
                else        /* last */
                    pARangeLoop->width = leftK;
            }
            pARangeLoop->height       = blockH;
            pARangeLoop->stride       = K;
            pARangeLoop->dataTypeSize = dataTypeSize;
            pARangeLoop++;
        }
    }

#if 0
    pARangeLoop = pARange;
    for (uint32_t i = 0; i < MBlocks; ++i)
    {
        for (uint32_t j = 0; j < KBlocks; ++j)
        {
            if ((TINY_SGEMM_BLOCK_K == pARangeLoop->width) && (TINY_SGEMM_BLOCK_M == pARangeLoop->height))
            {
                if (64 == TINY_SGEMM_BLOCK_K) /* arm64 */
                    sgemmPackA128x64(pARangeLoop->pStart, pPackA, K);
                else /* arm32 */
                    sgemmPackA128x128(pARangeLoop->pStart, pPackA, K);
            }
            else
            {
                if (8 == TINY_SGEMM_UNIT_M) /* arm64 */
                    sgemmPackA8x4(pARangeLoop->pStart, pPackA, K);
                else /* arm32 */
                    sgemmPackA4x4(pARangeLoop->pStart, pPackA, K);
            }

            pPackA += pARangeLoop->width*pARangeLoop->height;
            pARangeLoop++;
        }
    }
#endif
}

/* fp32 unit sgemm block is, A:4x4  B:4x12 C:4x12 */
void sgemm4xkx12_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t bRelu, float *pPrelu, uint32_t bPreluShare, float *pBasis)
{
    uint32_t KDiv4 = K>>2;
    uint32_t KHas2 = (K>>1)&1;
    uint32_t KHas1 = K&1;

    uint32x4_t vzero;
    float32x4x3_t vsrcC32x4x3_0, vsrcC32x4x3_1, vsrcC32x4x3_2, vsrcC32x4x3_3;   /* 12 registers */
    vzero = veorq_u32(vzero, vzero);
    vsrcC32x4x3_0.val[0] = vreinterpretq_f32_u32(vzero);
    vsrcC32x4x3_0.val[1] = vsrcC32x4x3_0.val[0];
    vsrcC32x4x3_0.val[2] = vsrcC32x4x3_0.val[0];
    vsrcC32x4x3_1 = vsrcC32x4x3_0;
    vsrcC32x4x3_2 = vsrcC32x4x3_0;
    vsrcC32x4x3_3 = vsrcC32x4x3_0;

    for (uint32_t i = 0; i < KDiv4; ++i)
    {
        float *pCurB = pB+i*48;
        float *pCurA = pA+i*16;
        float32x4_t vsrcA32x4;     /* 1 registers */
        float32x4x3_t vsrcB32x4x3; /* 3 registers */

        vsrcB32x4x3 = vld1q_f32_x3(pCurB);
        pCurB += 12;
        vsrcA32x4   = vld1q_f32(pCurA);
        pCurA += 4;
        vsrcC32x4x3_0.val[0] = vmlaq_n_f32(vsrcC32x4x3_0.val[0], vsrcB32x4x3.val[0], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_64(pCurB);
        vsrcC32x4x3_0.val[1] = vmlaq_n_f32(vsrcC32x4x3_0.val[1], vsrcB32x4x3.val[1], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_16(pCurA);
        vsrcC32x4x3_0.val[2] = vmlaq_n_f32(vsrcC32x4x3_0.val[2], vsrcB32x4x3.val[2], vsrcA32x4[0]);
        vsrcC32x4x3_1.val[0] = vmlaq_n_f32(vsrcC32x4x3_1.val[0], vsrcB32x4x3.val[0], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[1] = vmlaq_n_f32(vsrcC32x4x3_1.val[1], vsrcB32x4x3.val[1], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[2] = vmlaq_n_f32(vsrcC32x4x3_1.val[2], vsrcB32x4x3.val[2], vsrcA32x4[1]);
        vsrcC32x4x3_2.val[0] = vmlaq_n_f32(vsrcC32x4x3_2.val[0], vsrcB32x4x3.val[0], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[1] = vmlaq_n_f32(vsrcC32x4x3_2.val[1], vsrcB32x4x3.val[1], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[2] = vmlaq_n_f32(vsrcC32x4x3_2.val[2], vsrcB32x4x3.val[2], vsrcA32x4[2]);
        vsrcC32x4x3_3.val[0] = vmlaq_n_f32(vsrcC32x4x3_3.val[0], vsrcB32x4x3.val[0], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[1] = vmlaq_n_f32(vsrcC32x4x3_3.val[1], vsrcB32x4x3.val[1], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[2] = vmlaq_n_f32(vsrcC32x4x3_3.val[2], vsrcB32x4x3.val[2], vsrcA32x4[3]);

        vsrcB32x4x3 = vld1q_f32_x3(pCurB);
        pCurB += 12;
        vsrcA32x4   = vld1q_f32(pCurA);
        pCurA += 4;
        vsrcC32x4x3_0.val[0] = vmlaq_n_f32(vsrcC32x4x3_0.val[0], vsrcB32x4x3.val[0], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_64(pCurB);
        vsrcC32x4x3_0.val[1] = vmlaq_n_f32(vsrcC32x4x3_0.val[1], vsrcB32x4x3.val[1], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_16(pCurA);
        vsrcC32x4x3_0.val[2] = vmlaq_n_f32(vsrcC32x4x3_0.val[2], vsrcB32x4x3.val[2], vsrcA32x4[0]);
        vsrcC32x4x3_1.val[0] = vmlaq_n_f32(vsrcC32x4x3_1.val[0], vsrcB32x4x3.val[0], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[1] = vmlaq_n_f32(vsrcC32x4x3_1.val[1], vsrcB32x4x3.val[1], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[2] = vmlaq_n_f32(vsrcC32x4x3_1.val[2], vsrcB32x4x3.val[2], vsrcA32x4[1]);
        vsrcC32x4x3_2.val[0] = vmlaq_n_f32(vsrcC32x4x3_2.val[0], vsrcB32x4x3.val[0], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[1] = vmlaq_n_f32(vsrcC32x4x3_2.val[1], vsrcB32x4x3.val[1], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[2] = vmlaq_n_f32(vsrcC32x4x3_2.val[2], vsrcB32x4x3.val[2], vsrcA32x4[2]);
        vsrcC32x4x3_3.val[0] = vmlaq_n_f32(vsrcC32x4x3_3.val[0], vsrcB32x4x3.val[0], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[1] = vmlaq_n_f32(vsrcC32x4x3_3.val[1], vsrcB32x4x3.val[1], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[2] = vmlaq_n_f32(vsrcC32x4x3_3.val[2], vsrcB32x4x3.val[2], vsrcA32x4[3]);

        vsrcB32x4x3 = vld1q_f32_x3(pCurB);
        pCurB += 12;
        vsrcA32x4   = vld1q_f32(pCurA);
        pCurA += 4;
        vsrcC32x4x3_0.val[0] = vmlaq_n_f32(vsrcC32x4x3_0.val[0], vsrcB32x4x3.val[0], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_64(pCurB);
        vsrcC32x4x3_0.val[1] = vmlaq_n_f32(vsrcC32x4x3_0.val[1], vsrcB32x4x3.val[1], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_16(pCurA);
        vsrcC32x4x3_0.val[2] = vmlaq_n_f32(vsrcC32x4x3_0.val[2], vsrcB32x4x3.val[2], vsrcA32x4[0]);
        vsrcC32x4x3_1.val[0] = vmlaq_n_f32(vsrcC32x4x3_1.val[0], vsrcB32x4x3.val[0], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[1] = vmlaq_n_f32(vsrcC32x4x3_1.val[1], vsrcB32x4x3.val[1], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[2] = vmlaq_n_f32(vsrcC32x4x3_1.val[2], vsrcB32x4x3.val[2], vsrcA32x4[1]);
        vsrcC32x4x3_2.val[0] = vmlaq_n_f32(vsrcC32x4x3_2.val[0], vsrcB32x4x3.val[0], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[1] = vmlaq_n_f32(vsrcC32x4x3_2.val[1], vsrcB32x4x3.val[1], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[2] = vmlaq_n_f32(vsrcC32x4x3_2.val[2], vsrcB32x4x3.val[2], vsrcA32x4[2]);
        vsrcC32x4x3_3.val[0] = vmlaq_n_f32(vsrcC32x4x3_3.val[0], vsrcB32x4x3.val[0], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[1] = vmlaq_n_f32(vsrcC32x4x3_3.val[1], vsrcB32x4x3.val[1], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[2] = vmlaq_n_f32(vsrcC32x4x3_3.val[2], vsrcB32x4x3.val[2], vsrcA32x4[3]);

        vsrcB32x4x3 = vld1q_f32_x3(pCurB);
        pCurB += 12;
        vsrcA32x4   = vld1q_f32(pCurA);
        pCurA += 4;
        vsrcC32x4x3_0.val[0] = vmlaq_n_f32(vsrcC32x4x3_0.val[0], vsrcB32x4x3.val[0], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_64(pCurB);
        vsrcC32x4x3_0.val[1] = vmlaq_n_f32(vsrcC32x4x3_0.val[1], vsrcB32x4x3.val[1], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_16(pCurA);
        vsrcC32x4x3_0.val[2] = vmlaq_n_f32(vsrcC32x4x3_0.val[2], vsrcB32x4x3.val[2], vsrcA32x4[0]);
        vsrcC32x4x3_1.val[0] = vmlaq_n_f32(vsrcC32x4x3_1.val[0], vsrcB32x4x3.val[0], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[1] = vmlaq_n_f32(vsrcC32x4x3_1.val[1], vsrcB32x4x3.val[1], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[2] = vmlaq_n_f32(vsrcC32x4x3_1.val[2], vsrcB32x4x3.val[2], vsrcA32x4[1]);
        vsrcC32x4x3_2.val[0] = vmlaq_n_f32(vsrcC32x4x3_2.val[0], vsrcB32x4x3.val[0], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[1] = vmlaq_n_f32(vsrcC32x4x3_2.val[1], vsrcB32x4x3.val[1], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[2] = vmlaq_n_f32(vsrcC32x4x3_2.val[2], vsrcB32x4x3.val[2], vsrcA32x4[2]);
        vsrcC32x4x3_3.val[0] = vmlaq_n_f32(vsrcC32x4x3_3.val[0], vsrcB32x4x3.val[0], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[1] = vmlaq_n_f32(vsrcC32x4x3_3.val[1], vsrcB32x4x3.val[1], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[2] = vmlaq_n_f32(vsrcC32x4x3_3.val[2], vsrcB32x4x3.val[2], vsrcA32x4[3]);
    }

    /* A:4x2 B:2x8 C:4x8 */
    if (KHas2)
    {
        float *pCurB = pB+KDiv4*48;
        float *pCurA = pA+KDiv4*16;
        float32x4_t vsrcA32x4;     /* 1 registers */
        float32x4x3_t vsrcB32x4x3; /* 3 registers */

        vsrcB32x4x3 = vld1q_f32_x3(pCurB);
        pCurB += 12;
        vsrcA32x4   = vld1q_f32(pCurA);
        pCurA += 4;
        vsrcC32x4x3_0.val[0] = vmlaq_n_f32(vsrcC32x4x3_0.val[0], vsrcB32x4x3.val[0], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_64(pCurB);
        vsrcC32x4x3_0.val[1] = vmlaq_n_f32(vsrcC32x4x3_0.val[1], vsrcB32x4x3.val[1], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_16(pCurA);
        vsrcC32x4x3_0.val[2] = vmlaq_n_f32(vsrcC32x4x3_0.val[2], vsrcB32x4x3.val[2], vsrcA32x4[0]);
        vsrcC32x4x3_1.val[0] = vmlaq_n_f32(vsrcC32x4x3_1.val[0], vsrcB32x4x3.val[0], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[1] = vmlaq_n_f32(vsrcC32x4x3_1.val[1], vsrcB32x4x3.val[1], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[2] = vmlaq_n_f32(vsrcC32x4x3_1.val[2], vsrcB32x4x3.val[2], vsrcA32x4[1]);
        vsrcC32x4x3_2.val[0] = vmlaq_n_f32(vsrcC32x4x3_2.val[0], vsrcB32x4x3.val[0], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[1] = vmlaq_n_f32(vsrcC32x4x3_2.val[1], vsrcB32x4x3.val[1], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[2] = vmlaq_n_f32(vsrcC32x4x3_2.val[2], vsrcB32x4x3.val[2], vsrcA32x4[2]);
        vsrcC32x4x3_3.val[0] = vmlaq_n_f32(vsrcC32x4x3_3.val[0], vsrcB32x4x3.val[0], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[1] = vmlaq_n_f32(vsrcC32x4x3_3.val[1], vsrcB32x4x3.val[1], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[2] = vmlaq_n_f32(vsrcC32x4x3_3.val[2], vsrcB32x4x3.val[2], vsrcA32x4[3]);

        vsrcB32x4x3 = vld1q_f32_x3(pCurB);
        pCurB += 12;
        vsrcA32x4   = vld1q_f32(pCurA);
        pCurA += 4;
        vsrcC32x4x3_0.val[0] = vmlaq_n_f32(vsrcC32x4x3_0.val[0], vsrcB32x4x3.val[0], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_64(pCurB);
        vsrcC32x4x3_0.val[1] = vmlaq_n_f32(vsrcC32x4x3_0.val[1], vsrcB32x4x3.val[1], vsrcA32x4[0]);
        ARM_LOAD_PREFETCH_16(pCurA);
        vsrcC32x4x3_0.val[2] = vmlaq_n_f32(vsrcC32x4x3_0.val[2], vsrcB32x4x3.val[2], vsrcA32x4[0]);
        vsrcC32x4x3_1.val[0] = vmlaq_n_f32(vsrcC32x4x3_1.val[0], vsrcB32x4x3.val[0], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[1] = vmlaq_n_f32(vsrcC32x4x3_1.val[1], vsrcB32x4x3.val[1], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[2] = vmlaq_n_f32(vsrcC32x4x3_1.val[2], vsrcB32x4x3.val[2], vsrcA32x4[1]);
        vsrcC32x4x3_2.val[0] = vmlaq_n_f32(vsrcC32x4x3_2.val[0], vsrcB32x4x3.val[0], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[1] = vmlaq_n_f32(vsrcC32x4x3_2.val[1], vsrcB32x4x3.val[1], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[2] = vmlaq_n_f32(vsrcC32x4x3_2.val[2], vsrcB32x4x3.val[2], vsrcA32x4[2]);
        vsrcC32x4x3_3.val[0] = vmlaq_n_f32(vsrcC32x4x3_3.val[0], vsrcB32x4x3.val[0], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[1] = vmlaq_n_f32(vsrcC32x4x3_3.val[1], vsrcB32x4x3.val[1], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[2] = vmlaq_n_f32(vsrcC32x4x3_3.val[2], vsrcB32x4x3.val[2], vsrcA32x4[3]);
    }

    /* A:4x1 B:1x12 C:4x12 */
    if (KHas1)
    {
        float *pCurB = pB+(K-1)*12;
        float *pCurA = pA+(K-1)*4;
        float32x4_t vsrcA32x4;     /* 1 registers */
        float32x4x3_t vsrcB32x4x3; /* 3 registers */

        vsrcB32x4x3 = vld1q_f32_x3(pCurB);
        vsrcA32x4   = vld1q_f32(pCurA);
        vsrcC32x4x3_0.val[0] = vmlaq_n_f32(vsrcC32x4x3_0.val[0], vsrcB32x4x3.val[0], vsrcA32x4[0]);
        vsrcC32x4x3_0.val[1] = vmlaq_n_f32(vsrcC32x4x3_0.val[1], vsrcB32x4x3.val[1], vsrcA32x4[0]);
        vsrcC32x4x3_0.val[2] = vmlaq_n_f32(vsrcC32x4x3_0.val[2], vsrcB32x4x3.val[2], vsrcA32x4[0]);
        vsrcC32x4x3_1.val[0] = vmlaq_n_f32(vsrcC32x4x3_1.val[0], vsrcB32x4x3.val[0], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[1] = vmlaq_n_f32(vsrcC32x4x3_1.val[1], vsrcB32x4x3.val[1], vsrcA32x4[1]);
        vsrcC32x4x3_1.val[2] = vmlaq_n_f32(vsrcC32x4x3_1.val[2], vsrcB32x4x3.val[2], vsrcA32x4[1]);
        vsrcC32x4x3_2.val[0] = vmlaq_n_f32(vsrcC32x4x3_2.val[0], vsrcB32x4x3.val[0], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[1] = vmlaq_n_f32(vsrcC32x4x3_2.val[1], vsrcB32x4x3.val[1], vsrcA32x4[2]);
        vsrcC32x4x3_2.val[2] = vmlaq_n_f32(vsrcC32x4x3_2.val[2], vsrcB32x4x3.val[2], vsrcA32x4[2]);
        vsrcC32x4x3_3.val[0] = vmlaq_n_f32(vsrcC32x4x3_3.val[0], vsrcB32x4x3.val[0], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[1] = vmlaq_n_f32(vsrcC32x4x3_3.val[1], vsrcB32x4x3.val[1], vsrcA32x4[3]);
        vsrcC32x4x3_3.val[2] = vmlaq_n_f32(vsrcC32x4x3_3.val[2], vsrcB32x4x3.val[2], vsrcA32x4[3]);
    }

    if (pBasis)
    {
        /* one basis for one channel */
        float32x4_t vbasis32x4_0 = vdupq_n_f32(pBasis[0]);
        float32x4_t vbasis32x4_1 = vdupq_n_f32(pBasis[1]);
        float32x4_t vbasis32x4_2 = vdupq_n_f32(pBasis[2]);
        float32x4_t vbasis32x4_3 = vdupq_n_f32(pBasis[3]);
        vsrcC32x4x3_0.val[0] = vaddq_f32(vsrcC32x4x3_0.val[0], vbasis32x4_0);
        vsrcC32x4x3_0.val[1] = vaddq_f32(vsrcC32x4x3_0.val[1], vbasis32x4_0);
        vsrcC32x4x3_0.val[2] = vaddq_f32(vsrcC32x4x3_0.val[2], vbasis32x4_0);
        vsrcC32x4x3_1.val[0] = vaddq_f32(vsrcC32x4x3_1.val[0], vbasis32x4_1);
        vsrcC32x4x3_1.val[1] = vaddq_f32(vsrcC32x4x3_1.val[1], vbasis32x4_1);
        vsrcC32x4x3_1.val[2] = vaddq_f32(vsrcC32x4x3_1.val[2], vbasis32x4_1);
        vsrcC32x4x3_2.val[0] = vaddq_f32(vsrcC32x4x3_2.val[0], vbasis32x4_2);
        vsrcC32x4x3_2.val[1] = vaddq_f32(vsrcC32x4x3_2.val[1], vbasis32x4_2);
        vsrcC32x4x3_2.val[2] = vaddq_f32(vsrcC32x4x3_2.val[2], vbasis32x4_2);
        vsrcC32x4x3_3.val[0] = vaddq_f32(vsrcC32x4x3_3.val[0], vbasis32x4_3);
        vsrcC32x4x3_3.val[1] = vaddq_f32(vsrcC32x4x3_3.val[1], vbasis32x4_3);
        vsrcC32x4x3_3.val[2] = vaddq_f32(vsrcC32x4x3_3.val[2], vbasis32x4_3);
    }

    if (bRelu)
    {
        uint32x4_t vmask;
        float32x4_t vzero;
        vmask = veorq_u32(vmask, vmask);
        vzero = vreinterpretq_f32_u32(vmask);

        vmask                = vcleq_f32(vsrcC32x4x3_0.val[0], vzero);
        vsrcC32x4x3_0.val[0] = vbslq_f32(vmask, vzero, vsrcC32x4x3_0.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x3_0.val[1], vzero);
        vsrcC32x4x3_0.val[1] = vbslq_f32(vmask, vzero, vsrcC32x4x3_0.val[1]);
        vmask                = vcleq_f32(vsrcC32x4x3_0.val[2], vzero);
        vsrcC32x4x3_0.val[2] = vbslq_f32(vmask, vzero, vsrcC32x4x3_0.val[2]);

        vmask                = vcleq_f32(vsrcC32x4x3_1.val[0], vzero);
        vsrcC32x4x3_1.val[0] = vbslq_f32(vmask, vzero, vsrcC32x4x3_1.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x3_1.val[1], vzero);
        vsrcC32x4x3_1.val[1] = vbslq_f32(vmask, vzero, vsrcC32x4x3_1.val[1]);
        vmask                = vcleq_f32(vsrcC32x4x3_1.val[2], vzero);
        vsrcC32x4x3_1.val[2] = vbslq_f32(vmask, vzero, vsrcC32x4x3_1.val[2]);

        vmask                = vcleq_f32(vsrcC32x4x3_2.val[0], vzero);
        vsrcC32x4x3_2.val[0] = vbslq_f32(vmask, vzero, vsrcC32x4x3_2.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x3_2.val[1], vzero);
        vsrcC32x4x3_2.val[1] = vbslq_f32(vmask, vzero, vsrcC32x4x3_2.val[1]);
        vmask                = vcleq_f32(vsrcC32x4x3_2.val[2], vzero);
        vsrcC32x4x3_2.val[2] = vbslq_f32(vmask, vzero, vsrcC32x4x3_2.val[2]);

        vmask                = vcleq_f32(vsrcC32x4x3_3.val[0], vzero);
        vsrcC32x4x3_3.val[0] = vbslq_f32(vmask, vzero, vsrcC32x4x3_3.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x3_3.val[1], vzero);
        vsrcC32x4x3_3.val[1] = vbslq_f32(vmask, vzero, vsrcC32x4x3_3.val[1]);
        vmask                = vcleq_f32(vsrcC32x4x3_3.val[2], vzero);
        vsrcC32x4x3_3.val[2] = vbslq_f32(vmask, vzero, vsrcC32x4x3_3.val[2]);
    }
    else if (pPrelu)
    {
        uint32x4_t vmask;
        float32x4_t vscale32x4, vmul, vzero;
        vmask = veorq_u32(vmask, vmask);
        vzero = vreinterpretq_f32_u32(vmask);
        if (bPreluShare) /* all channel use same prelu */
            vscale32x4 = vdupq_n_f32(pPrelu[0]);
        else
            vscale32x4 = vld1q_f32(pPrelu);

        vmask                = vcleq_f32(vsrcC32x4x3_0.val[0], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_0.val[0], vscale32x4[0]);
        vsrcC32x4x3_0.val[0] = vbslq_f32(vmask, vmul, vsrcC32x4x3_0.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x3_0.val[1], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_0.val[1], vscale32x4[0]);
        vsrcC32x4x3_0.val[1] = vbslq_f32(vmask, vmul, vsrcC32x4x3_0.val[1]);
        vmask                = vcleq_f32(vsrcC32x4x3_0.val[2], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_0.val[2], vscale32x4[0]);
        vsrcC32x4x3_0.val[2] = vbslq_f32(vmask, vmul, vsrcC32x4x3_0.val[2]);

        vmask                = vcleq_f32(vsrcC32x4x3_1.val[0], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_1.val[0], vscale32x4[1]);
        vsrcC32x4x3_1.val[0] = vbslq_f32(vmask, vmul, vsrcC32x4x3_1.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x3_1.val[1], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_1.val[1], vscale32x4[1]);
        vsrcC32x4x3_1.val[1] = vbslq_f32(vmask, vmul, vsrcC32x4x3_1.val[1]);
        vmask                = vcleq_f32(vsrcC32x4x3_1.val[2], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_1.val[2], vscale32x4[1]);
        vsrcC32x4x3_1.val[2] = vbslq_f32(vmask, vmul, vsrcC32x4x3_1.val[2]);

        vmask                = vcleq_f32(vsrcC32x4x3_2.val[0], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_2.val[0], vscale32x4[2]);
        vsrcC32x4x3_2.val[0] = vbslq_f32(vmask, vmul, vsrcC32x4x3_2.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x3_2.val[1], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_2.val[1], vscale32x4[2]);
        vsrcC32x4x3_2.val[1] = vbslq_f32(vmask, vmul, vsrcC32x4x3_2.val[1]);
        vmask                = vcleq_f32(vsrcC32x4x3_2.val[2], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_2.val[2], vscale32x4[2]);
        vsrcC32x4x3_2.val[2] = vbslq_f32(vmask, vmul, vsrcC32x4x3_2.val[2]);

        vmask                = vcleq_f32(vsrcC32x4x3_3.val[0], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_3.val[0], vscale32x4[3]);
        vsrcC32x4x3_3.val[0] = vbslq_f32(vmask, vmul, vsrcC32x4x3_3.val[0]);
        vmask                = vcleq_f32(vsrcC32x4x3_3.val[1], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_3.val[1], vscale32x4[3]);
        vsrcC32x4x3_3.val[1] = vbslq_f32(vmask, vmul, vsrcC32x4x3_3.val[1]);
        vmask                = vcleq_f32(vsrcC32x4x3_3.val[2], vzero);
        vmul                 = vmulq_n_f32(vsrcC32x4x3_3.val[2], vscale32x4[3]);
        vsrcC32x4x3_3.val[2] = vbslq_f32(vmask, vmul, vsrcC32x4x3_3.val[2]);
    }

    vst1q_f32_x3(pC,     vsrcC32x4x3_0);
    vst1q_f32_x3(pC+N,   vsrcC32x4x3_1);
    vst1q_f32_x3(pC+2*N, vsrcC32x4x3_2);
    vst1q_f32_x3(pC+3*N, vsrcC32x4x3_3);
}
