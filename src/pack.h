#ifndef __TINYSGEMMCONV_PACK_H
#define __TINYSGEMMCONV_PACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void tinySgemmConvPackB4x24_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N);
void tinySgemmConvPackB4x12_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N);
void tinySgemmConvPackBLeft_fp32_fp32(float *pB, float *pPackB, uint32_t K, uint32_t leftN);
void tinySgemmConvPackA4x4_fp32_fp32(float *pA, float *pPackA, uint32_t M, uint32_t K);

#ifdef __cplusplus
}
#endif

#endif