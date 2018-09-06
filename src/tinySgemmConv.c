#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
#include <assert.h>
#include <tinySgemmConv.h>
#include "config.h"
#include "common.h"
#include "innerTinySgemmConv.h"
#include "thread_server.h"

int tinySgemmConvInit(uint32_t num_threads, int32_t stack_size, uint32_t (*affinity)[MAX_CORE_NUMBER], void **pCtx)
{
    int32_t ret = 0;
    uint32_t availCores = 0, i = 0;
    struct thread_info *pThreadInfo = NULL;
    pthread_attr_t attr;
    struct tinySgemmConvCtx *pCtxInner = NULL;
    uint32_t coreFreq[MAX_CORE_NUMBER];
    uint32_t maxFreq;
    struct msg *pMsg = NULL;

    if (NULL == pCtx)
        return -1;

    /* create memory pool */
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

    printf("num_threads:%d\n", num_threads);
    num_threads = T_MIN(num_threads, MAX_CORE_NUMBER);
    availCores  = getAvaiableCoresMaxFreq(&coreFreq, &maxFreq);
    num_threads = T_MIN(num_threads, availCores);
    printf("availCores :%d\n", availCores);

    pThreadInfo = calloc(num_threads, sizeof(struct thread_info));
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

    for (i = 0; i < num_threads; i++)
    {
        pThreadInfo[i].sgemmCtx = (void *)pCtxInner;
        pThreadInfo[i].maxFrequence = coreFreq[i];
        if (maxFreq == pThreadInfo[i].maxFrequence)
            pThreadInfo[i].bigCore = 1;
        pThreadInfo[i].index = i;
        if (NULL != affinity)
            pThreadInfo[i].affinity = affinity[0][i];
        else
            pThreadInfo[i].affinity = -1;
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
            return -7;
        }
    }

    if (-1 != stack_size)
    {
        ret = pthread_attr_destroy(&attr);
        if (ret != 0)
        {
            printf("%s, %d\n", "pthread_attr_destroy failed", ret);
            free(pThreadInfo);
            free(pCtxInner);
            return -8;
        }
    }

    INIT_LIST_HEAD(&pCtxInner->msgHeadFree);
    ret = pthread_mutex_init(&pCtxInner->msgLock, NULL);
    if (0 != ret)
    {
        printf("%s, %d\n", "pthread_mutex_init(msLock) failed", ret);
        free(pThreadInfo);
        free(pCtxInner);
        return -9;
    }

    pMsg = (struct msg *)calloc(MAX_MSGPOOL_NUM, sizeof(struct msg));
    if (NULL == pMsg)
    {
        printf("%s, %d\n", "msg pool malloc failed", MAX_MSGPOOL_NUM * (uint32_t)sizeof(struct msg));
        pthread_mutex_destroy(&pCtxInner->msgLock);
        free(pThreadInfo);
        free(pCtxInner);
        return -10;
    }

    for (uint32_t i = 0; i < MAX_MSGPOOL_NUM; ++i)
        list_add_tail(&pMsg[i].listFree, &pCtxInner->msgHeadFree);

    pCtxInner->num_threads = num_threads;
    pCtxInner->pThreadInfo = pThreadInfo;
    pCtxInner->pMsgPool    = pMsg;
    *pCtx = pCtxInner;
    return ret;
}

static void returnMsg(struct tinySgemmConvCtx *pCtx, struct msg *pMsg)
{
    pthread_mutex_lock(&pCtx->msgLock);
    list_add_tail(&pMsg->listFree, &pCtx->msgHeadFree);
    pthread_mutex_unlock(&pCtx->msgLock);
}

static struct msg * fetchMsg(struct tinySgemmConvCtx *pCtx)
{
    int ret;
    struct msg *pMsg = NULL;
    pthread_mutex_lock(&pCtx->msgLock);
    if (list_empty(&pCtx->msgHeadFree))
    {
        pthread_mutex_unlock(&pCtx->msgLock);
        return NULL;
    }
    pMsg = list_first_entry(&pCtx->msgHeadFree, struct msg, listFree);
    list_del(&pMsg->listFree);
    pthread_mutex_unlock(&pCtx->msgLock);
    assert(NULL != pMsg);
    pMsg->status = MSG_STATUS_IDEL;
    ret = pthread_mutex_init(&pMsg->lock, NULL);
    if (0 != ret)
    {
        printf("%s %d, %d\n", "msg pthread_mutex_init failed at line", __LINE__, ret);
        returnMsg(pCtx, pMsg);
        return NULL;
    }
    ret = pthread_cond_init(&pMsg->jobDoneCondition, NULL);
    if (0 != ret)
    {
        printf("%s %d, %d\n", "msg pthread_cond_init failed at line", __LINE__, ret);
        pthread_mutex_destroy(&pMsg->lock);
        returnMsg(pCtx, pMsg);
        return NULL;
    }
    return pMsg;
}

static inline struct rangeInfo *range(struct rangeInfo *pRange, uint32_t i, uint32_t j)
{
    if ((NULL != pRange) && (i < pRange->YBlocks) && (j < pRange->XBlocks))
        return pRange + i * pRange->XBlocks + j;
    else
        return NULL;
}

static struct matricRangeInfo* createMatircRange(void *pA, void *pB, void *pC,
        uint32_t M, uint32_t N, uint32_t K,
        float *pBasis, bool bRelu, float *pPRelu,
        bool bSharedPrelu, enum TINY_SGEMM_CONV_DATA_MODE mode)
{
    struct matricRangeInfo *pMatricRangeInfo;
    uint8_t *pACur, *pBCur, *pCCur;
    struct rangeInfo *pARange, *pBRange, *pCRange;
    uint32_t numBlockM = M / TINY_SGEMM_BLOCK_M;
    uint32_t numBlockN = N / TINY_SGEMM_BLOCK_N;
    uint32_t numBlockK = K / TINY_SGEMM_BLOCK_K;
    uint32_t leftM = M % TINY_SGEMM_BLOCK_M;
    uint32_t leftN = N % TINY_SGEMM_BLOCK_N;
    uint32_t leftK = K % TINY_SGEMM_BLOCK_K;
    uint32_t mBlocks = numBlockM + ((0 == leftM)?(0):(1));
    uint32_t nBlocks = numBlockN + ((0 == leftN)?(0):(1));
    uint32_t kBlocks = numBlockK + ((0 == leftK)?(0):(1));
    uint32_t dataTypeSize;

    pMatricRangeInfo = (struct matricRangeInfo *)calloc(1, (mBlocks*kBlocks + kBlocks*nBlocks + mBlocks*nBlocks)*sizeof(struct rangeInfo) + sizeof(struct matricRangeInfo));
    if (NULL == pMatricRangeInfo)
    {
        printf("%s at line: %d %d %d %d\n", "no memory", __LINE__, mBlocks, nBlocks, kBlocks);
        return NULL;
    }
    pARange = (struct rangeInfo *)(pMatricRangeInfo + 1);
    pBRange = pARange + mBlocks*kBlocks;
    pCRange = pBRange + kBlocks*nBlocks;
    pMatricRangeInfo->pARange = pARange;
    pMatricRangeInfo->pBRange = pBRange;
    pMatricRangeInfo->pCRange = pCRange;

    if (TINY_SGEMM_CONV_DATA_MODE_A_FIX16_FIX16_B_FP32 == mode)
        dataTypeSize = sizeof(uint16_t);
    else if (TINY_SGEMM_CONV_DATA_MODE_A_FIX8_FIX8_B_FP32 != mode)
        dataTypeSize = sizeof(float);
    else
        dataTypeSize = sizeof(uint8_t);

    /* split A */
    for (uint32_t i = 0; i < mBlocks; ++i)
    {
        uint32_t blockH;
        if (M < (i+1)*TINY_SGEMM_BLOCK_M)
        {
            if (0 == i) /* first */
            {
                pACur  = (uint8_t*)pA;
                blockH = M;
            }
            else /* last */
            {
                pACur  = (uint8_t*)pA + (M - leftM)*K*dataTypeSize;
                blockH = leftM;
            }
        }
        else
        {
            pACur  = (uint8_t*)pA + i*TINY_SGEMM_BLOCK_M*K*dataTypeSize;
            blockH = TINY_SGEMM_BLOCK_M;
        }

        for (uint32_t j = 0; j < kBlocks; ++j)
        {
            pARange->pStart  = (uint8_t*)pACur + j*TINY_SGEMM_BLOCK_K*dataTypeSize;
            pARange->XBlocks = kBlocks;
            pARange->YBlocks = mBlocks;
            if (K < (j+1)*TINY_SGEMM_BLOCK_K)
            {
                if (0 == j) /* first */
                    pARange->width = K;
                else /* last */
                    pARange->width = K - leftK;
            }
            else
                pARange->width    = TINY_SGEMM_BLOCK_K;
            pARange->height       = blockH;
            pARange->stride       = K;
            pARange->dataTypeSize = dataTypeSize;
            pARange++;
        }
    }

    /* split B */
    dataTypeSize = sizeof(float);
    for (uint32_t i = 0; i < kBlocks; ++i)
    {
        uint32_t blockH;
        if (K < (i+1)*TINY_SGEMM_BLOCK_K)
        {
            if (0 == i) /* first */
            {
                pBCur  = (uint8_t*)pB;
                blockH = K;
            }
            else /* last */
            {
                pBCur  = (uint8_t*)pB + (K - leftK)*N*dataTypeSize;
                blockH = leftK;
            }
        }
        else
        {
            pBCur  = (uint8_t*)pB + i*TINY_SGEMM_BLOCK_K*N*dataTypeSize;
            blockH = TINY_SGEMM_BLOCK_K;
        }

        for (uint32_t j = 0; j < nBlocks; ++j)
        {
            pBRange->pStart  = (uint8_t*)pBCur + j*TINY_SGEMM_BLOCK_N*dataTypeSize;
            pBRange->XBlocks = nBlocks;
            pBRange->YBlocks = kBlocks;
            if (N < (j+1)*TINY_SGEMM_BLOCK_N)
            {
                if (0 == j) /* first */
                    pBRange->width = N;
                else /* last */
                    pBRange->width = N - leftN;
            }
            else
                pBRange->width    = TINY_SGEMM_BLOCK_N;
            pBRange->height       = blockH;
            pBRange->stride       = N;
            pBRange->dataTypeSize = dataTypeSize;
            pBRange++;
        }
    }

    /* split C */
    dataTypeSize = sizeof(float);
    for (uint32_t i = 0; i < mBlocks; ++i)
    {
        uint32_t blockH;
        if (M < (i+1)*TINY_SGEMM_BLOCK_M)
        {
            if (0 == i) /* first */
            {
                pCCur  = (uint8_t*)pC;
                blockH = M;
            }
            else /* last */
            {
                pCCur  = (uint8_t*)pC + (M - leftM)*N*dataTypeSize;
                blockH = leftM;
            }
        }
        else
        {
            pCCur = (uint8_t*)pC + i*TINY_SGEMM_BLOCK_M*N*dataTypeSize;
            blockH = TINY_SGEMM_BLOCK_M;
        }

        for (uint32_t j = 0; j < nBlocks; ++j)
        {
            pCRange->pStart  = (uint8_t*)pCCur + j*TINY_SGEMM_BLOCK_N*dataTypeSize;
            pCRange->XBlocks = nBlocks;
            pCRange->YBlocks = mBlocks;
            pCRange->runOnlittleCore = 0;
            if (N < (j+1)*TINY_SGEMM_BLOCK_N)
            {
                if (0 == j) /* first */
                    pCRange->width = N;
                else /* last */
                {
                    pCRange->width = N - leftN;
                    /* small range can be assigned to little core */
                    if (pCRange->width*2 < TINY_SGEMM_BLOCK_N)
                        pCRange->runOnlittleCore = 1;
                }
            }
            else
                pCRange->width    = TINY_SGEMM_BLOCK_N;
            pCRange->height       = blockH;
            pCRange->stride       = N;
            pCRange->dataTypeSize = dataTypeSize;
            pCRange->pBasis       = NULL;
            if (pBasis)
            {
                pCRange->pBasis   = pBasis;
                pBasis            += blockH;
            }
            pCRange->bRelu        = bRelu;
            pCRange->bSharedPrelu = bSharedPrelu;
            pCRange->pPRelu       = NULL;
            if (pPRelu)
            {
                pCRange->pPRelu   = pPRelu;
                if (!bSharedPrelu)
                    pPRelu        += blockH;
            }

            pCRange++;
        }
    }

    return pMatricRangeInfo;
}

static void tinySgemmConvPackA(void *pA, void *pPackA, uint32_t M, uint32_t K, enum TINY_SGEMM_CONV_DATA_MODE mode)
{

}

/* do pack weight & im2col B buffer malloc */
void* tinySgemmConvCreateInstance(void *pCtx, void *pWeight,
                                  uint32_t inChannels,  uint32_t inputH, uint32_t inputW,
                                  uint32_t outChannels, uint32_t kernelH, uint32_t kernelW,
                                  uint32_t padH, uint32_t padW,
                                  uint32_t strideH, uint32_t strideW,
                                  uint32_t dilateH, uint32_t dilateW,
                                  enum TINY_SGEMM_CONV_DATA_MODE mode)
{
    uint32_t outputW = (inputW + 2*padW - kernelW)/strideW + 1;
    uint32_t outputH = (inputH + 2*padH - kernelH)/strideH + 1;
    uint32_t M = outChannels;
    uint32_t N = outputH*outputW;
    uint32_t K = inChannels*kernelH*kernelW;
    uint32_t packBTypeSize, packATypeSize, packBPerThreadSize;
    void *pIm2colB, *pPackA, *pPackBPerThread;
    struct tinySgemmInstance *psgemmInstance;
    struct tinySgemmConvCtx *pCtxInner = (struct tinySgemmConvCtx *)pCtx;

    /* TODO: support dilate conv */
    (void)dilateH;
    (void)dilateW;
    assert(NULL != pWeight);

    psgemmInstance = (struct tinySgemmInstance*)calloc(1, sizeof(struct tinySgemmInstance));
    if (NULL == psgemmInstance)
    {
        printf("%s\n", "tinySgemmInstance malloc failed");
        return NULL;
    }

    /* Im2col B buffer */
    if (kernelW == 1 && kernelH == 1 && strideH == 1 && strideW == 1 && padH == 0 && padW == 0)
        pIm2colB = NULL;
    else
    {
        pIm2colB = (void *)tinySgemmMalloc(K*N*sizeof(float));
        if (NULL == pIm2colB)
        {
            printf("%s\n", "im2col buffer null");
            free(psgemmInstance);
            return NULL;
        }
    }

    /* thread block pack B */
    switch(mode)
    {
    case TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP16_B_FP32:
        packBTypeSize = sizeof(uint16_t);
        packATypeSize = sizeof(uint16_t);
        break;
    case TINY_SGEMM_CONV_DATA_MODE_A_FIX16_FIX16_B_FP32:
        packBTypeSize = sizeof(uint16_t);
        packATypeSize = sizeof(uint16_t);
        break;
    case TINY_SGEMM_CONV_DATA_MODE_A_FIX8_FIX8_B_FP32:
        packBTypeSize = sizeof(uint8_t);
        packATypeSize = sizeof(uint8_t);
        break;
    default:
        packBTypeSize = sizeof(float);
        packATypeSize = sizeof(float);
        break;
    }

    /* malloc packB & packA buffer */
    packBPerThreadSize = alignSize(TINY_SGEMM_BLOCK_K*TINY_SGEMM_BLOCK_N*packBTypeSize, MALLOC_MEM_ALIGN);
    pPackBPerThread = (void *)tinySgemmMalloc(pCtxInner->num_threads*packBPerThreadSize + M*K*packATypeSize);
    if (NULL == pPackBPerThread)
    {
        printf("%s\n", "packB buffer malloc failed");
        tinySgemmFree(pIm2colB); /* NULL pointer check inner */
        free(psgemmInstance);
        return NULL;
    }

    /* do packA */
    pPackA = (uint8_t *)pPackBPerThread + pCtxInner->num_threads*packBPerThreadSize;
    tinySgemmConvPackA(pWeight, pPackA, M, K, mode);

    psgemmInstance->M                  = M;
    psgemmInstance->N                  = N;
    psgemmInstance->K                  = K;
    psgemmInstance->pPackA             = pPackA;
    psgemmInstance->pIm2colB           = pIm2colB;
    psgemmInstance->pPackBPerThread    = pPackBPerThread;
    psgemmInstance->packBPerThreadSize = packBPerThreadSize;
    psgemmInstance->mode               = mode;
    psgemmInstance->pCtx               = pCtx;

    return (void*)psgemmInstance;
}

int tinySgemmConvReleaseInstance(void *pInstance)
{
    struct tinySgemmInstance *pInnerInstance = (struct tinySgemmInstance *)pInstance;
    if (NULL == pInnerInstance)
    {
        printf("%s\n", "NULL pointer in release instance");
        return -1;
    }

    tinySgemmFree(pInnerInstance->pIm2colB); /* NULL pointer check inner */
    tinySgemmFree(pInnerInstance->pPackBPerThread);
    free(pInnerInstance);
    return 0;
}

static void waitForJobsDone(struct tinySgemmConvCtx *pCtxInner, struct list_head *workQueue)
{
    uint32_t totalJobs = list_avail(workQueue);
    while(!list_empty(workQueue))
    {
        struct msg *pMsg = list_first_entry(workQueue, struct msg, listWork);
        assert(NULL != pMsg);
        pthread_mutex_lock(&pMsg->lock);
        while (MSG_STATUS_DONE != pMsg->status)
            pthread_cond_wait(&pMsg->jobDoneCondition, &pMsg->lock);
        list_del(&pMsg->listWork);
        pthread_mutex_unlock(&pMsg->lock);
#define PRT_JOB_TIME
#ifdef PRT_JOB_TIME
        printf("[%d][%d] %d/%d job done in %lu us\n", pMsg->cmd, pMsg->coreId, pMsg->idx, totalJobs, pMsg->end-pMsg->beg);
#endif
        returnMsg(pCtxInner, pMsg);
    }
}

int tinySgemmConv(void *pInstance,
                  void *pInput, void *pOutput,
                  float *pBasis, bool bRelu, float *pPRelu, bool bSharedPrelu,
                  enum TINY_SGEMM_CONV_DATA_MODE mode)
{
    int ret = 0;
    uint32_t littlecore_ids[MAX_CORE_NUMBER];
    uint32_t bigcore_ids[MAX_CORE_NUMBER];
    uint32_t bigCoreNum = 0, littleCoreNum = 0;
    uint32_t bigCoreIdx = 0, littleCoreIdx = 0;
    struct matricRangeInfo* pMarticRangeInfo;
    struct tinySgemmInstance *psgemmInstance = (struct tinySgemmInstance *)pInstance;
    struct tinySgemmConvCtx *pCtxInner;
    struct list_head workQueue;
    void *pB;
    uint32_t M, N, K;

    if (NULL == psgemmInstance)
    {
        printf("%s\n", "NULL Instance pointer");
        return -1;
    }

    M = psgemmInstance->M;
    N = psgemmInstance->N;
    K = psgemmInstance->K;

    pCtxInner = psgemmInstance->pCtx;
    INIT_LIST_HEAD(&workQueue);
    for (uint32_t i = 0; i < pCtxInner->num_threads; ++i)
    {
        if (pCtxInner->pThreadInfo[i].bigCore)
        {
            if (bigCoreNum <= (MAX_CORE_NUMBER-1))
                bigcore_ids[bigCoreNum++] = i;
            else
            {
                printf("%s\n", "bigcore idx overflow");
                return -2;
            }
        }
        else
        {
            if (littleCoreNum <= (MAX_CORE_NUMBER-1))
                littlecore_ids[littleCoreNum++] = i;
            else
            {
                printf("%s\n", "littlecore idx overflow");
                return -3;
            }
        }
    }

    if (NULL != psgemmInstance->pIm2colB)
    {
        /* TODO: do im2col mutithread */

        pB = psgemmInstance->pIm2colB;
    }
    else
        pB = pInput;

    /* wait for all im2col jobs done */
    waitForJobsDone(pCtxInner, &workQueue);

    pMarticRangeInfo = createMatircRange(psgemmInstance->pPackA, pB, pOutput, M, N, K, pBasis, bRelu, pPRelu, bSharedPrelu, mode);
    if (NULL == pMarticRangeInfo)
    {
        printf("%s\n", "matricRangeInfo create failed");
        return -4;
    }

    /* assign range to threads */
    for (uint32_t i = 0; i < pMarticRangeInfo->pCRange->YBlocks; ++i)
    {
        for (uint32_t j = 0; j < pMarticRangeInfo->pCRange->XBlocks; ++j)
        {
            uint32_t coreId = 0;
            struct msg *pMsg  = fetchMsg(pCtxInner);
            assert(NULL != pMsg);
            pMsg->cmd         = THREAD_CMD_SGEMM_WORK;
            pMsg->idx         = list_avail(&workQueue);
            pMsg->pWorkCRange = pMarticRangeInfo->pCRange + i*pMarticRangeInfo->pCRange->XBlocks + j;
            pMsg->pPackBPerThread = (uint8_t*)psgemmInstance->pPackBPerThread + coreId*psgemmInstance->packBPerThreadSize;
            if (pMarticRangeInfo->pCRange->runOnlittleCore && (littleCoreNum > 0))
            {
                coreId        = littlecore_ids[littleCoreIdx];
                littleCoreIdx = (littleCoreIdx + 1) % littleCoreNum;
            }
            else
            {
                coreId        = bigcore_ids[bigCoreIdx];
                bigCoreIdx    = (bigCoreIdx + 1) % bigCoreNum;
            }
            pMsg->coreId = coreId;
            sendMsg(&pCtxInner->pThreadInfo[coreId], pMsg);
            list_add_tail(&pMsg->listWork, &workQueue);
        }
    }

    /* wait for all sgemm jobs done */
    waitForJobsDone(pCtxInner, &workQueue);
    free(pMarticRangeInfo);
    return ret;
}

int tinySgemmConvDeinit(void *pCtx)
{
    struct list_head workQueue;
    struct tinySgemmConvCtx *pCtxInner = (struct tinySgemmConvCtx *)pCtx;
    if (NULL == pCtxInner)
    {
        printf("%s, deinit failed\n", "NULL pCtx");
        return -1;
    }
    INIT_LIST_HEAD(&workQueue);
    for (uint32_t i = 0; i < pCtxInner->num_threads; i++)
    {
        struct msg *pMsg = (struct msg *)fetchMsg(pCtxInner);
        assert(NULL != pMsg);
        pMsg->cmd = THREAD_CMD_EXIT;
        sendMsg(&pCtxInner->pThreadInfo[i], pMsg);
        list_add_tail(&pMsg->listWork, &workQueue);
    }
    waitForJobsDone(pCtxInner, &workQueue);
    for (uint32_t i = 0; i < pCtxInner->num_threads; i++)
    {
        pthread_join(pCtxInner->pThreadInfo[i].thread_id, NULL);
        pthread_mutex_destroy(&pCtxInner->pThreadInfo[i].queue_lock);
        pthread_cond_destroy (&pCtxInner->pThreadInfo[i].noempty);
    }
    free(pCtxInner->pMsgPool);
    free(pCtxInner->pThreadInfo);
    free(pCtxInner);
    return 0;
}