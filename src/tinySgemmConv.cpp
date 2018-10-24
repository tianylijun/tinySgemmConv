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
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
#include <assert.h>
#include <math.h>
#include <tinySgemmConv.h>
#include "list.h"
#include "common.h"
#include "pack.h"
#include "innerTinySgemmConv.h"
#include "thread_server.h"
#include "messageQueue.h"

static inline uint32_t getMaxFreqAccordToAffinity(uint32_t affinity, uint32_t *coresMaxFreq)
{
    uint32_t maxFreq = 0;
    POINTER_CHECK(coresMaxFreq, 0);
    for (uint32_t i = 0; i < MAX_CORE_NUMBER; ++i)
    {
        if (affinity & (1U<<i))
            maxFreq = T_MAX(maxFreq, coresMaxFreq[i]);
    }
    return maxFreq;
}

int tinySgemmConvInit
(
    uint32_t num_threads,
    int32_t stack_size,
    uint32_t (*affinity)[MAX_CORE_NUMBER],
    bool bindBigCore,
    void **pCtx
)
{
    int32_t ret = 0, threadsStatusCheck = 1;
    uint32_t availCores = 0;
    struct thread_info *pThreadInfo = NULL;
    pthread_attr_t attr;
    struct tinySgemmConvCtx *pCtxInner = NULL;
    uint32_t coresMaxFreq[MAX_CORE_NUMBER] = {0};
    uint32_t maxFreq = 0, bigCoreMask = 0, bigCoreNum = 0;
    bool biglittlecore = false;
    struct msg *pMsgPool = NULL;

    POINTER_CHECK(pCtx, -1);

    if (-1 != stack_size)
    {
        stack_size = T_MAX(stack_size, PTHREAD_STACK_MIN);
        ret = pthread_attr_init(&attr);
        if (ret != 0)
        {
            printf("%s, %d\n", "pthread_attr_init", ret);
            return -2;
        }
        ret = pthread_attr_setstacksize(&attr, stack_size);
        if (ret != 0)
        {
            printf("%s %d %d\n", "pthread_attr_setstacksize", stack_size, ret);
            pthread_attr_destroy(&attr);
            return -3;
        }
    }

    num_threads = T_MIN(num_threads, MAX_CORE_NUMBER);
    availCores  = getAvaiableCoresMaxFreq(&coresMaxFreq, &maxFreq);
    if (num_threads > availCores)
    {
        printf("Warning: num_threads adjust from %d to %d, as availCores limit\n", num_threads, availCores);
        num_threads = availCores;
    }

    for (uint32_t i = 0; i < availCores; ++i)
    {
        if (maxFreq != coresMaxFreq[i])
        {
            biglittlecore = true;
        }
        else
        {
            bigCoreNum++;
            bigCoreMask |= 1U<<i;
        }
    }
    printf("num_threads:%d, availCores:%d, maxFreq:%d, biglittlecore:%s, bigCoreMask:%#x, bigCoreNum:%d\n",
           num_threads, availCores, maxFreq, biglittlecore?"Yes":"No", bigCoreMask, bigCoreNum);
    for (uint32_t i = 0; i < availCores; ++i)
        printf("[%d] max freq: %d\n", i, coresMaxFreq[i]);

    pThreadInfo = (struct thread_info*)calloc(num_threads, sizeof(struct thread_info));
    if (NULL == pThreadInfo)
    {
        printf("%s, %d\n", "calloc thread info failed", num_threads);
        pthread_attr_destroy(&attr);
        return -4;
    }

    pCtxInner = (struct tinySgemmConvCtx *)calloc(1, sizeof(struct tinySgemmConvCtx));
    if (NULL == pCtxInner)
    {
        printf("%s, %d\n", "pthread_attr_destroy failed", ret);
        pthread_attr_destroy(&attr);
        free(pThreadInfo);
        return -5;
    }

    INIT_LIST_HEAD(&pCtxInner->bigCoreThreads);
    INIT_LIST_HEAD(&pCtxInner->littleCoreThreads);

    for (uint32_t i = 0; i < num_threads; i++)
    {
        pThreadInfo[i].index    = i;
        pThreadInfo[i].affinity = 0xffffffff;

        if (NULL != affinity) /* user define bind core */
        {
            pThreadInfo[i].affinity = affinity[0][i];
            //printf("[%d/%d] affinity: %#x\n", i,num_threads, affinity[0][i]);
            pThreadInfo[i].maxFrequence = getMaxFreqAccordToAffinity(pThreadInfo[i].affinity, coresMaxFreq);
        }
        else /* bind all */
        {
            if (bindBigCore)
                pThreadInfo[i].affinity = bigCoreMask;
            pThreadInfo[i].maxFrequence = maxFreq;
        }

        //printf("max fre: %d, affinity:%#x\n", pThreadInfo[i].maxFrequence, pThreadInfo[i].affinity);
        if (maxFreq == pThreadInfo[i].maxFrequence)
        {
            pThreadInfo[i].bigCore = 1;
            list_add_tail(&pThreadInfo[i].biglittlecorelist, &pCtxInner->bigCoreThreads);
        }
        else
        {
            pThreadInfo[i].bigCore = 0;
            list_add_tail(&pThreadInfo[i].biglittlecorelist, &pCtxInner->littleCoreThreads);
            //printf("[%d] little core\n", i);
        }
        if (-1 != stack_size)
            ret = pthread_create(&pThreadInfo[i].thread_id, NULL,
                                 &sgemm_thread_process, &pThreadInfo[i]);
        else
            ret = pthread_create(&pThreadInfo[i].thread_id, &attr,
                                 &sgemm_thread_process, &pThreadInfo[i]);
        if (0 != ret)
        {
            printf("%s, %d, %d\n", "pthread_create failed", ret, i);
            pthread_attr_destroy(&attr);
            free(pThreadInfo);
            free(pCtxInner);
            return -6;
        }
    }

    /* wait all thread init ok */
    while(threadsStatusCheck)
    {
        usleep(2000);
        uint32_t status = 0;
        for (uint32_t i = 0; i < num_threads; i++)
        {
            if (pThreadInfo[i].status)
                status++;
        }
        if (num_threads == status)
            threadsStatusCheck = 0;
    }

    if (-1 != stack_size)
    {
        ret = pthread_attr_destroy(&attr);
        if (ret != 0)
        {
            printf("%s, %d\n", "pthread_attr_destroy failed", ret);
            free(pThreadInfo);
            free(pCtxInner);
            return -7;
        }
    }

    INIT_LIST_HEAD(&pCtxInner->msgPoolList);
    ret = pthread_mutex_init(&pCtxInner->msgPoolLock, NULL);
    if (0 != ret)
    {
        printf("%s, %d\n", "pthread_mutex_init(msLock) failed", ret);
        free(pThreadInfo);
        free(pCtxInner);
        return -8;
    }
    ret = pthread_mutex_init(&pCtxInner->threadLock, NULL);
    if (0 != ret)
    {
        printf("%s, %d\n", "pthread_mutex_init(msLock) failed", ret);
        pthread_mutex_destroy(&pCtxInner->msgPoolLock);
        free(pThreadInfo);
        free(pCtxInner);
        return -8;
    }
    pMsgPool = msgPoolInit(pCtxInner, MAX_MSGPOOL_NUM);
    if (NULL == pMsgPool)
    {
        printf("%s, %d\n", "msg pool malloc failed", MAX_MSGPOOL_NUM * (uint32_t)sizeof(struct msg));
        pthread_mutex_destroy(&pCtxInner->threadLock);
        pthread_mutex_destroy(&pCtxInner->msgPoolLock);
        free(pThreadInfo);
        free(pCtxInner);
        return -9;
    }

    pCtxInner->num_threads = num_threads;
    pCtxInner->pThreadInfo = pThreadInfo;
    pCtxInner->pMsgPool    = pMsgPool;
    *pCtx = pCtxInner;
    return ret;
}

/* do pack weight & im2col B buffer malloc */
void* tinySgemmConvCreateInstance(void *pCtx, void *pWeight,
                                  uint32_t inChannels,  uint32_t inputH, uint32_t inputW,
                                  uint32_t outChannels, uint32_t kernelH, uint32_t kernelW,
                                  uint32_t padH, uint32_t padW,
                                  uint32_t strideH, uint32_t strideW,
                                  uint32_t dilateH, uint32_t dilateW,
                                  bool tf_pad,
                                  enum TINY_SGEMM_CONV_DATA_MODE mode)
{
    uint32_t i, packBTypeSize, packATypeSize, packBSize;
    uint8_t *pBIm2col, *pPackA, *pPackB;
    struct tinySgemmInstance *psgemmInstance;
    enum SGEMM_DataType packADataType, packBDataType;
    uint32_t outputW = (inputW + 2*padW - kernelW)/strideW + 1;
    uint32_t outputH = (inputH + 2*padH - kernelH)/strideH + 1;
    uint32_t M = outChannels;
    uint32_t N = outputH*outputW;
    uint32_t K = inChannels*kernelH*kernelW;
    bool pad_only_bottom = false, pad_only_right = false;
    int padding_bottom = 0, padding_right = 0;
    struct tinySgemmConvCtx *pCtxInner = (struct tinySgemmConvCtx *)pCtx;

    POINTER_CHECK(pCtx, NULL);
    POINTER_CHECK(pWeight, NULL);

    psgemmInstance = (struct tinySgemmInstance*)calloc(1, sizeof(struct tinySgemmInstance));
    POINTER_CHECK(psgemmInstance, NULL);

    if (tf_pad) /* TF SAME */
    {
        int pad_all_height, pad_all_width, padding_top, padding_left;

        outputW  = ceil((float)inputW / (float)strideW);
        outputH = ceil((float)inputH / (float)strideH);
        N = outputH*outputW;

        pad_all_height = (outputH - 1) * strideH + kernelH - inputH;
        padding_top = int(pad_all_height / 2.0);
        padding_bottom = pad_all_height - padding_top;

        pad_all_width = (outputW - 1) * strideW + kernelW - inputW;
        padding_left = int(pad_all_width / 2.0);
        padding_right = pad_all_width - padding_left;

        pad_only_bottom = padding_top == 0?true:false;
        pad_only_right = padding_left == 0?true:false;
        printf("TF conv pad: [%d %d %d %d]\n", padding_left, padding_right, padding_top, padding_bottom);
    }

    switch(mode)
    {
    case TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP16:
        packATypeSize = sizeof(uint16_t);
        packBTypeSize = sizeof(uint16_t);
        packADataType = FLOAT16_TYPE;
        packBDataType = FLOAT16_TYPE;
        break;
    case TINY_SGEMM_CONV_DATA_MODE_A_FIX16_FIX16:
        packATypeSize = sizeof(uint16_t);
        packBTypeSize = sizeof(uint16_t);
        packADataType = INT16_TYPE;
        packBDataType = INT16_TYPE;
        break;
    case TINY_SGEMM_CONV_DATA_MODE_A_FIX8_FIX8:
        packATypeSize = sizeof(uint8_t);
        packBTypeSize = sizeof(uint8_t);
        packADataType = INT8_TYPE;
        packBDataType = INT8_TYPE;
        break;
    default:
        packATypeSize = sizeof(float);
        packBTypeSize = sizeof(float);
        packADataType = FLOAT32_TYPE;
        packBDataType = FLOAT32_TYPE;
        break;
    }

    if (kernelW == 1 && kernelH == 1 && strideH == 1 && strideW == 1 && padH == 0 && padW == 0)
    {
        pBIm2col      = NULL;
        packBTypeSize = sizeof(float);
    }
    else
    {
        /* we do data narrow during im2col stage for not 1x1 case */
        pBIm2col = (uint8_t *)tinySgemmMalloc(K*N*packBTypeSize);
        if (NULL == pBIm2col)
        {
            printf("im2col B buffer malloc failed\n");
            free(psgemmInstance);
            return NULL;
        }
    }

    packBSize = alignSize(K*TINY_SGEMM_UNIT_N*packBTypeSize, MALLOC_MEM_ALIGN);
    /* packB(num_threads) + packA */
    pPackB = (uint8_t *)tinySgemmMalloc(pCtxInner->num_threads*packBSize + M*K*packATypeSize);
    if (NULL == pPackB)
    {
        printf("packB + packA buffer malloc failed\n");
        if (kernelW != 1 || kernelH != 1 || strideH != 1 || strideW != 1 || padH != 0 || padW != 0)
            tinySgemmFree(pBIm2col);
        free(psgemmInstance);
        return NULL;
    }

    pPackA = (uint8_t *)pPackB + pCtxInner->num_threads*packBSize;
    switch(packADataType)
    {
    case FLOAT32_TYPE:
        tinySgemmConvPackA4x4_fp32_fp32((float*)pWeight, (float*)pPackA, M, K);
        break;
    case FLOAT16_TYPE:
        //tinySgemmConvPackA4x4_fp32_fp16((float*)pWeight, (float*)pPackA, M, K);
        break;
    case INT16_TYPE:
        break;
    case INT8_TYPE:
        break;
    }

    psgemmInstance->M                  = M;
    psgemmInstance->N                  = N;
    psgemmInstance->K                  = K;
    psgemmInstance->inChannels         = inChannels;
    psgemmInstance->inputH             = inputH;
    psgemmInstance->inputW             = inputW;
    psgemmInstance->outChannels        = outChannels;
    psgemmInstance->kernelH            = kernelH;
    psgemmInstance->kernelW            = kernelW;
    if (tf_pad)
    {
        psgemmInstance->padH           = padding_bottom;
        psgemmInstance->padW           = pad_only_right;
    }
    else
    {
        psgemmInstance->padH           = padH;
        psgemmInstance->padW           = padW;
    }
    psgemmInstance->pad_only_bottom    = pad_only_bottom;
    psgemmInstance->pad_only_right     = pad_only_right;
    psgemmInstance->strideH            = strideH;
    psgemmInstance->strideW            = strideW;
    psgemmInstance->dilateH            = dilateH;
    psgemmInstance->dilateW            = dilateW;
    psgemmInstance->pPackA             = pPackA;
    psgemmInstance->pBIm2col           = pBIm2col;
    assert(pCtxInner->num_threads <= MAX_CORE_NUMBER);
    for (i = 0; i < pCtxInner->num_threads; ++i)
        psgemmInstance->pPackB[i]      = (uint8_t *)pPackB + i*packBSize;
    psgemmInstance->packATypeSize      = packATypeSize;
    psgemmInstance->packBTypeSize      = packBTypeSize;
    psgemmInstance->packADataType      = packADataType;
    psgemmInstance->packBDataType      = packBDataType;
    psgemmInstance->pCtx               = pCtxInner;

    return (void*)psgemmInstance;
}

int tinySgemmConvReleaseInstance(void *pInstance)
{
    struct tinySgemmInstance *pInnerInstance = (struct tinySgemmInstance *)pInstance;
    POINTER_CHECK(pInnerInstance, -1);
    /* NULL pointer check inner */
    tinySgemmFree(pInnerInstance->pBIm2col);
    tinySgemmFree(pInnerInstance->pPackB[0]);
    free(pInnerInstance);
    return 0;
}

int tinySgemmConvProcess(void *pInstance,
                         float *pInput, float *pOutput,
                         float *pBasis, enum TINY_SGEMM_RELU_TYPE reluType, float *pPrelu, bool bSharedPrelu,
                         float (*int8Scale)[3],
                         enum TINY_SGEMM_CONV_DATA_MODE mode)
{
    uint32_t i, N, packBTypeSize, blockN, leftN;
    struct list_head jobsQueue;
    enum SGEMM_DataType packBDataType, packADataType;
    struct tinySgemmConvCtx *pCtxInner;
    struct tinySgemmInstance *psgemmInstance = (struct tinySgemmInstance *)pInstance;

    if (NULL == pInstance || NULL == pInput || NULL == pOutput)
    {
        printf("%s, %p %p %p\n", "NULL pointer", pInstance, pInput, pOutput);
        return -1;
    }

    pCtxInner = psgemmInstance->pCtx;
    POINTER_CHECK(pCtxInner, -2);

    packBTypeSize = psgemmInstance->packBTypeSize;
    packADataType = psgemmInstance->packADataType;
    packBDataType = psgemmInstance->packBDataType;

    N      = psgemmInstance->N;
    blockN = N/TINY_SGEMM_UNIT_N;
    leftN  = N%TINY_SGEMM_UNIT_N;

    INIT_LIST_HEAD(&jobsQueue);
    //printf("Im2col start\n");
    TIME_STAMP_BEG(begIm2col);

    /* 1x1 not need do im2col for input */
    if (NULL == psgemmInstance->pBIm2col)
    {
        psgemmInstance->pBIm2col = (uint8_t *)pInput;
        packBDataType            = FLOAT32_TYPE;
        packBTypeSize            = sizeof(float);
    }
    else
    {
        uint32_t inputChannelSize = psgemmInstance->inputH*psgemmInstance->inputW;

        for (i = 0; i < psgemmInstance->inChannels; ++i)
        {
            struct thread_info *pThreadInfo = NULL;
            if (!list_empty(&pCtxInner->bigCoreThreads))
                pThreadInfo = getMostFreeThread(pCtxInner, &pCtxInner->bigCoreThreads, MSG_CMD_IM2COL);
            if ((NULL == pThreadInfo) && !list_empty(&pCtxInner->littleCoreThreads))
                pThreadInfo = getMostFreeThread(pCtxInner, &pCtxInner->littleCoreThreads, MSG_CMD_IM2COL);
            assert(NULL != pThreadInfo);
            struct msg *pMsg                  = fetchMsg(pCtxInner);
            assert(NULL != pMsg);
            pMsg->pThreadInfo                 = pThreadInfo;
            pMsg->cmd                         = MSG_CMD_IM2COL;
            pMsg->JobInfo.im2colInfo.kernelH  = psgemmInstance->kernelH;
            pMsg->JobInfo.im2colInfo.kernelW  = psgemmInstance->kernelW;
            pMsg->JobInfo.im2colInfo.strideH  = psgemmInstance->strideH;
            pMsg->JobInfo.im2colInfo.strideW  = psgemmInstance->strideW;
            pMsg->JobInfo.im2colInfo.padH     = psgemmInstance->padH;
            pMsg->JobInfo.im2colInfo.padW     = psgemmInstance->padW;
            pMsg->JobInfo.im2colInfo.dilateH  = psgemmInstance->dilateH;
            pMsg->JobInfo.im2colInfo.dilateW  = psgemmInstance->dilateW;
            pMsg->JobInfo.im2colInfo.height   = psgemmInstance->inputH;
            pMsg->JobInfo.im2colInfo.width    = psgemmInstance->inputW;
            pMsg->JobInfo.im2colInfo.outType  = packBDataType;
            pMsg->JobInfo.im2colInfo.pB       = pInput + i*inputChannelSize;
            pMsg->JobInfo.im2colInfo.pad_only_bottom = psgemmInstance->pad_only_bottom;
            pMsg->JobInfo.im2colInfo.pad_only_right = psgemmInstance->pad_only_right;
            pMsg->JobInfo.im2colInfo.pBIm2col = psgemmInstance->pBIm2col + i*psgemmInstance->kernelH*psgemmInstance->kernelW*N*packBTypeSize;

            sendMsg(pMsg);
            list_add_tail(&pMsg->listJobsQueue, &jobsQueue);
        }
        waitForJobsDone(pCtxInner, &jobsQueue);
    }
    TIME_STAMP_END(begIm2col, endIm2col, "im2col");
    assert(list_empty(jobsQueue));

    //printf("- blockN: %d leftN: %d N: %d TINY_SGEMM_UNIT_N: %d -\n", blockN, leftN, N, TINY_SGEMM_UNIT_N);
    TIME_STAMP_BEG(begSgemm);

    for (i = 0; i < blockN; ++i)
    {
        struct thread_info *pThreadInfo = NULL;
        if (!list_empty(&pCtxInner->bigCoreThreads))
            pThreadInfo = getMostFreeThread(pCtxInner, &pCtxInner->bigCoreThreads, MSG_CMD_SGEMM);
        if ((NULL == pThreadInfo) && !list_empty(&pCtxInner->littleCoreThreads))
            pThreadInfo = getMostFreeThread(pCtxInner, &pCtxInner->littleCoreThreads, MSG_CMD_SGEMM);
        assert(NULL != pThreadInfo);
        struct msg *pMsg                      = fetchMsg(pCtxInner);
        assert(NULL != pMsg);
        pMsg->pThreadInfo                     = pThreadInfo;
        pMsg->cmd                             = MSG_CMD_SGEMM;
        pMsg->JobInfo.sgemmInfo.M             = psgemmInstance->M;
        pMsg->JobInfo.sgemmInfo.N             = psgemmInstance->N;
        pMsg->JobInfo.sgemmInfo.K             = psgemmInstance->K;
        pMsg->JobInfo.sgemmInfo.n             = TINY_SGEMM_UNIT_N;
        pMsg->JobInfo.sgemmInfo.pA            = psgemmInstance->pPackA;
        pMsg->JobInfo.sgemmInfo.pBIm2col      = (uint8_t *)psgemmInstance->pBIm2col + i*TINY_SGEMM_UNIT_N*packBTypeSize;
        pMsg->JobInfo.sgemmInfo.pC            = pOutput + i*TINY_SGEMM_UNIT_N;
        pMsg->JobInfo.sgemmInfo.pPackB        = psgemmInstance->pPackB[pThreadInfo->index];
        pMsg->JobInfo.sgemmInfo.pBasis        = pBasis;
        pMsg->JobInfo.sgemmInfo.reluType      = reluType;
        pMsg->JobInfo.sgemmInfo.pPrelu        = pPrelu;
        pMsg->JobInfo.sgemmInfo.bSharedPrelu  = bSharedPrelu;
        pMsg->JobInfo.sgemmInfo.int8Scale     = int8Scale;
        pMsg->JobInfo.sgemmInfo.packADataType = packADataType;
        pMsg->JobInfo.sgemmInfo.packBDataType = packBDataType;

        sendMsg(pMsg);
        list_add_tail(&pMsg->listJobsQueue, &jobsQueue);
    }

    if (0 != leftN)
    {
        struct thread_info *pThreadInfo = NULL;
        if (!list_empty(&pCtxInner->littleCoreThreads))
            pThreadInfo = getMostFreeThread(pCtxInner, &pCtxInner->littleCoreThreads, MSG_CMD_SGEMM);
        if ((NULL == pThreadInfo) && !list_empty(&pCtxInner->bigCoreThreads))
            pThreadInfo = getMostFreeThread(pCtxInner, &pCtxInner->bigCoreThreads, MSG_CMD_SGEMM);
        assert(NULL != pThreadInfo);
        struct msg *pMsg                      = fetchMsg(pCtxInner);
        pMsg->pThreadInfo                     = pThreadInfo;
        pMsg->cmd                             = MSG_CMD_SGEMM;
        pMsg->JobInfo.sgemmInfo.M             = psgemmInstance->M;
        pMsg->JobInfo.sgemmInfo.N             = psgemmInstance->N;
        pMsg->JobInfo.sgemmInfo.K             = psgemmInstance->K;
        pMsg->JobInfo.sgemmInfo.n             = leftN;
        pMsg->JobInfo.sgemmInfo.pA            = psgemmInstance->pPackA;
        pMsg->JobInfo.sgemmInfo.pBIm2col      = (uint8_t *)psgemmInstance->pBIm2col + blockN*TINY_SGEMM_UNIT_N*packBTypeSize;
        pMsg->JobInfo.sgemmInfo.pC            = pOutput + blockN*TINY_SGEMM_UNIT_N;
        pMsg->JobInfo.sgemmInfo.pPackB        = psgemmInstance->pPackB[pThreadInfo->index];
        pMsg->JobInfo.sgemmInfo.pBasis        = pBasis;
        pMsg->JobInfo.sgemmInfo.reluType      = reluType;
        pMsg->JobInfo.sgemmInfo.pPrelu        = pPrelu;
        pMsg->JobInfo.sgemmInfo.bSharedPrelu  = bSharedPrelu;
        pMsg->JobInfo.sgemmInfo.int8Scale     = int8Scale;
        pMsg->JobInfo.sgemmInfo.packADataType = packADataType;
        pMsg->JobInfo.sgemmInfo.packBDataType = packBDataType;
        sendMsg(pMsg);
        list_add_tail(&pMsg->listJobsQueue, &jobsQueue);
    }

    waitForJobsDone(pCtxInner, &jobsQueue);

    TIME_STAMP_END(begSgemm, endSgemm, "SGEMM");
    return 0;
}

int tinySgemmConvDeinit(void *pCtx)
{
    struct list_head jobsQueue;
    struct tinySgemmConvCtx *pCtxInner = (struct tinySgemmConvCtx *)pCtx;
    POINTER_CHECK(pCtxInner, -1);

    /* send exit cmd msg to each thread */
    INIT_LIST_HEAD(&jobsQueue);
    for (uint32_t i = 0; i < pCtxInner->num_threads; i++)
    {
        struct msg *pMsg = (struct msg *)fetchMsg(pCtxInner);
        assert(NULL != pMsg);
        pMsg->cmd         = MSG_CMD_EXIT;
        pMsg->pThreadInfo = &pCtxInner->pThreadInfo[i];
        sendMsg(pMsg);
        list_add_tail(&pMsg->listJobsQueue, &jobsQueue);
    }
    waitForJobsDone(pCtxInner, &jobsQueue);

    for (uint32_t i = 0; i < pCtxInner->num_threads; i++)
    {
        pthread_join(pCtxInner->pThreadInfo[i].thread_id, NULL);
        pthread_mutex_destroy(&pCtxInner->pThreadInfo[i].msgQueueLock);
        pthread_cond_destroy(&pCtxInner->pThreadInfo[i].msgQueueNoEmpty);
    }
    pthread_mutex_destroy(&pCtxInner->msgPoolLock);
    pthread_mutex_destroy(&pCtxInner->threadLock);
    free(pCtxInner->pThreadInfo);
    msgPoolDeInit(pCtxInner);
    free(pCtxInner);
    return 0;
}