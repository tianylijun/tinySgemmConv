#ifndef __TINYSGEMMCONV_SGEMM_H
#define __TINYSGEMMCONV_SGEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void sgemmMxKx24_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx16_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx12_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx8_fp32 (float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx4_fp32 (float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx2_fp32 (float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx1_fp32 (float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t bRelu, float *pPrelu, uint32_t bPreluShare, float *pBasis);

#ifdef __cplusplus
}
#endif

#endif