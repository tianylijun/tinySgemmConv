#ifndef __MESSAGEQUEUE_H
#define __MESSAGEQUEUE_H

#include <stdint.h>
#include <pthread.h>
#include "list.h"
#include "innerTinySgemmConv.h"

enum MSG_STATUS
{
    MSG_STATUS_IDEL,
    MSG_STATUS_BUSY,
    MSG_STATUS_DONE
};

struct msg
{
    uint32_t cmd;
    uint64_t sequenceId;
    enum MSG_STATUS status;
    struct thread_info *pThreadInfo;
    struct rangeInfo *pWorkCRange;
    void *pPackBPerThread;
    pthread_mutex_t lock;
    pthread_cond_t jobDoneCondition;
    struct list_head listMsgQueue;
    struct list_head listJobsQueue;
    struct list_head listMsgPool;
    uint64_t timeStampBeg;
    uint64_t timeStampEnd;
};

#ifdef __cplusplus
extern "C" {
#endif

struct msg *msgPoolInit(struct tinySgemmConvCtx *pCtx, uint32_t maxNumber);
int msgPoolDeInit(struct tinySgemmConvCtx *pCtx);
void returnMsg(struct tinySgemmConvCtx *pCtx, struct msg *pMsg);
struct msg * fetchMsg(struct tinySgemmConvCtx *pCtx);
void sendMsg(struct msg *pMsg);
struct msg * rcvMsg(struct thread_info *pThreadInfo);

#ifdef __cplusplus
}
#endif

#endif