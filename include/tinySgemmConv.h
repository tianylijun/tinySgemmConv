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

#ifndef __TINYSGEMM_CONV_H
#define __TINYSGEMM_CONV_H

#include <stdint.h>
#include <stdbool.h>

#ifndef MAX_CORE_NUMBER
#define MAX_CORE_NUMBER (32U)
#endif

#ifndef THREAD_STACK_SIZE
#define THREAD_STACK_SIZE (16*1024)
#endif

#ifndef MAX_MSGPOOL_NUM
#define MAX_MSGPOOL_NUM (1024U)
#endif

enum TINY_SGEMM_CONV_DATA_MODE
{
    TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP32,
    TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP16,
    TINY_SGEMM_CONV_DATA_MODE_A_FIX16_FIX16,
    TINY_SGEMM_CONV_DATA_MODE_A_FIX8_FIX8
};

#ifdef __cplusplus
extern "C" {
#endif

int tinySgemmConvInit(uint32_t num_threads, int32_t stack_size, uint32_t (*affinity)[MAX_CORE_NUMBER], void **pCtx);

/* pCtx param is return by  tinySgemmConvInit */
void* tinySgemmConvCreateInstance(void *pCtx, void *pWeight,
                                  uint32_t inChannels,  uint32_t inputH, uint32_t inputW,
                                  uint32_t outChannels, uint32_t kernelH, uint32_t kernelW,
                                  uint32_t padH, uint32_t padW,
                                  uint32_t strideH, uint32_t strideW,
                                  uint32_t dilateH, uint32_t dilateW,
                                  enum TINY_SGEMM_CONV_DATA_MODE mode);
/* pInstance param is return by  tinySgemmConvCreateInstance */
int tinySgemmConv(void *pInstance,
                  void *pInput, void *pOutPut,
                  float *pBasis, bool bRelu, float *pPrelu, bool bSharedPrelu,
                  float (*int8Scale)[3], enum TINY_SGEMM_CONV_DATA_MODE mode);

int tinySgemmConvReleaseInstance(void *pInstance);
int tinySgemmConvDeinit(void *pCtx);

#ifdef __cplusplus
}
#endif

#endif
