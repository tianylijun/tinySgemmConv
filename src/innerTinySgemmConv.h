#ifndef __INNERTINYMATRIXMUL_H
#define __INNERTINYMATRIXMUL_H

#include <pthread.h>
#include <stdbool.h>
#include "list.h"

struct rangeInfo
{
    uint8_t *pStart;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t dataTypeSize;
    uint32_t XBlocks;
    uint32_t YBlocks;
    uint32_t runOnlittleCore;
    float *pBasis;
    bool bRelu;
    float *pPRelu;
    bool bSharedPrelu;
    struct list_head list;
};

struct matricRangeInfo
{
    struct rangeInfo *pARange;
    struct rangeInfo *pBRange;
    struct rangeInfo *pCRange;
};

struct thread_info
{
    uint32_t index;
    uint32_t maxFrequence;
    uint32_t bigCore;
    pthread_t thread_id;
    pthread_mutex_t msgQueueLock;
    struct list_head msgQueueList;
    pthread_cond_t msgQueueNoEmpty;
    uint32_t affinity;
    void *sgemmCtx;
    struct list_head biglittlecorelist;
    uint64_t sgemmJobsDoneNum;
    uint64_t im2colJobsDoneNum;
};

struct tinySgemmConvCtx
{
    uint32_t num_threads;
    struct thread_info *pThreadInfo;
    pthread_mutex_t msgPoolLock;
    pthread_mutex_t threadLock;
    struct msg *pMsgPool;
    struct list_head msgPoolList;
    struct list_head bigCoreThreads;
    struct list_head littleCoreThreads;
};

struct tinySgemmInstance
{
    uint8_t *pPackA;
    void *pIm2colB;
    uint8_t *pPackBPerThread[MAX_CORE_NUMBER];
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t packBPerThreadSize;
    enum TINY_SGEMM_CONV_DATA_MODE mode;
    struct tinySgemmConvCtx *pCtx;
};

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif