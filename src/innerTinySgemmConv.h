#ifndef __INNERTINYMATRIXMUL_H
#define __INNERTINYMATRIXMUL_H

#include <pthread.h>
#include <stdbool.h>
#include "list.h"

enum MSG_STATUS
{
    MSG_STATUS_IDEL,
    MSG_STATUS_BUSY,
    MSG_STATUS_DONE
};

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

enum THREAD_CMD
{
    THREAD_CMD_EXIT,
    THREAD_CMD_SGEMM_WORK,
    THREAD_CMD_IM2COL_WORK
};

struct thread_info
{
    uint32_t index;
    uint32_t maxFrequence;
    uint32_t bigCore;
    pthread_t thread_id;
    pthread_mutex_t queue_lock;
    struct list_head msgHead;
    pthread_cond_t noempty;
    uint32_t affinity;
    void *sgemmCtx;
    struct list_head biglittlecorelist;
    uint64_t jobsDoneNum;
};

struct msg
{
    uint32_t cmd;
    uint32_t sequenceId;
    enum MSG_STATUS status;
    struct thread_info *pThreadInfo;
    struct rangeInfo *pWorkCRange;
    void *pPackBPerThread;
    pthread_mutex_t lock;
    pthread_cond_t jobDoneCondition;
    struct list_head listThread;
    struct list_head listWork;
    struct list_head listFree;
    uint64_t beg;
    uint64_t end;
};

struct tinySgemmConvCtx
{
    uint32_t num_threads;
    struct thread_info *pThreadInfo;
    pthread_mutex_t msgLock;
    struct msg *pMsgPool;
    struct list_head msgHeadFree;
    struct list_head bigCoreThreads;
    struct list_head littleCoreThreads;
};

struct tinySgemmInstance
{
    void *pPackA;
    void *pIm2colB;
    void *pPackBPerThread;
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