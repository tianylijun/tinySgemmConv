#ifndef __INNERTINYMATRIXMUL_H
#define __INNERTINYMATRIXMUL_H

#include <pthread.h>
#include <stdbool.h>
#include "list.h"

enum RANGE_STATUS
{
    RANGE_STATUS_IDEL,
    RANGE_STATUS_BUSY,
    RANGE_STATUS_DONE
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
    enum RANGE_STATUS status;
    pthread_mutex_t lock;
    pthread_cond_t jobDoneCondition;
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
    THREAD_CMD_WORK
};

struct msg
{
    uint32_t cmd;
    struct rangeInfo *pWorkCRange;
    struct list_head list;
    struct list_head listCtx;
    struct list_head listWorkRange;
};

struct thread_info
{
    uint32_t index;
    uint32_t maxFrequence;
    uint32_t bigCore;
    pthread_t thread_id;
    pthread_mutex_t queue_lock;
    struct list_head head;
    pthread_cond_t noempty;
    int32_t affinity;
    void *sgemmCtx;
    void *pPackB;
};

struct tinySgemmConvCtx
{
    uint32_t num_threads;
    struct thread_info *pThreadInfo;
    pthread_mutex_t msgLock;
    struct list_head msgHeadFree;
    struct list_head msgHeadBusy;
};

#ifdef __cplusplus
extern "C" {
#endif

void returnMsg(struct tinySgemmConvCtx *pCtx, struct msg *pMsg);

#ifdef __cplusplus
}
#endif

#endif