#ifndef __TINYSGEMMCONV_PACK_H
#define __TINYSGEMMCONV_PACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void tinySgemmConvPackB4x12_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N);
void tinySgemmConvPackA4x4_fp32_fp32(float *pA, float *pPackA, uint32_t M, uint32_t K);
void tinySgemmConvPackB4x4_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N);
void tinySgemmConvPackB4x2_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N);
void tinySgemmConvPackB4x12_fp32_fp32(float *pB, float *pPackB, uint32_t K, uint32_t N);

#ifdef __cplusplus
}
#endif

#endif