#ifndef __TINYSGEMMCONV_PACK_H
#define __TINYSGEMMCONV_PACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void tinySgemmConvPackA4x4(void *pA, void *pPackA, uint32_t M, uint32_t K, enum TINY_SGEMM_CONV_DATA_MODE mode);
void tinySgemmConvPackA8x4(void *pA, void *pPackA, uint32_t M, uint32_t K, enum TINY_SGEMM_CONV_DATA_MODE mode);

#ifdef __cplusplus
}
#endif

#endif