#ifndef __TINYMATRIXMUL_H
#define __TINYMATRIXMUL_H

#include <stdint.h>
#include <stdbool.h>

#ifndef MAX_CORE_NUMBER
#define MAX_CORE_NUMBER (32U)
#endif

#ifndef THREAD_STACK_SIZE
#define THREAD_STACK_SIZE (16*1024)
#endif

#ifndef MAX_MSGPOOL_NUM
#define MAX_MSGPOOL_NUM (512U)
#endif

enum TINY_SGEMM_CONV_DATA_MODE
{
    TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP32_B_FP32,
    TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP16_B_FP32,
    TINY_SGEMM_CONV_DATA_MODE_A_FIX16_FIX16_B_FP32,
    TINY_SGEMM_CONV_DATA_MODE_A_FIX8_FIX8_B_FP32
};

#ifdef __cplusplus
extern "C" {
#endif

int tinySgemmConvInit(uint32_t num_threads, int32_t stack_size, uint32_t (*affinity)[MAX_CORE_NUMBER], void **pCtx);
int tinySgemmConv(void *pCtx,
	              void *pA, void *pB, void *pC,
	              uint32_t M, uint32_t N, uint32_t K,
	              float *pBasis, bool bRelu, float *pRelu, bool bSharedPrelu,
	              enum TINY_SGEMM_CONV_DATA_MODE mode);
int tinySgemmConvDeinit(void *pCtx);

#ifdef __cplusplus
}
#endif

#endif
