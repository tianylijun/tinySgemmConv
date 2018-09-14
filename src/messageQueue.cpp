#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sched.h>
#include <pthread.h>
#include "common.h"
#include "list.h"
#include "messageQueue.h"
#include "innerTinySgemmConv.h"

static uint64_t msgSequence = 0;

struct msg *msgPoolInit(struct tinySgemmConvCtx *pCtx, uint32_t maxNumber)
{
    struct msg *pMsg;
    POINTER_CHECK(pCtx, NULL);
    pMsg = (struct msg *)calloc(maxNumber, sizeof(struct msg));
    POINTER_CHECK(pMsg, NULL);

    for (uint32_t i = 0; i < maxNumber; ++i)
        list_add_tail(&pMsg[i].listMsgPool, &pCtx->msgPoolList);

    return pMsg;
}

int msgPoolDeInit(struct tinySgemmConvCtx *pCtx)
{
    POINTER_CHECK(pCtx, -1);
    /* clear msg pool */
    INIT_LIST_HEAD(&pCtx->msgPoolList);
    assert(NULL != pCtx->pMsgPool);
    free(pCtx->pMsgPool);
    return 0;
}

struct msg *fetchMsg(struct tinySgemmConvCtx *pCtx)
{
    struct msg *pMsg;
    POINTER_CHECK(pCtx, NULL);
    pthread_mutex_lock(&pCtx->msgPoolLock);
    if (list_empty(&pCtx->msgPoolList))
    {
        pthread_mutex_unlock(&pCtx->msgPoolLock);
        return NULL;
    }
    pMsg = list_first_entry(&pCtx->msgPoolList, struct msg, listMsgPool);
    list_del(&pMsg->listMsgPool);
    pMsg->sequenceId = ++msgSequence;
    pMsg->status     = MSG_STATUS_IDEL;
    pthread_mutex_unlock(&pCtx->msgPoolLock);
    pthread_mutex_init(&pMsg->lock, NULL);
    pthread_cond_init(&pMsg->jobDoneCondition, NULL);
    return pMsg;
}

void returnMsg(struct tinySgemmConvCtx *pCtx, struct msg *pMsg)
{
    POINTER_CHECK_NO_RET(pCtx);
    POINTER_CHECK_NO_RET(pMsg);
    pthread_mutex_lock(&pCtx->msgPoolLock);
    list_add_tail(&pMsg->listMsgPool, &pCtx->msgPoolList);
    pthread_mutex_unlock(&pCtx->msgPoolLock);
}

void sendMsg(struct msg *pMsg)
{
    POINTER_CHECK_NO_RET(pMsg);
    POINTER_CHECK_NO_RET(pMsg->pThreadInfo);
    pthread_mutex_lock(&pMsg->pThreadInfo->msgQueueLock);
    list_add_tail(&pMsg->listMsgQueue, &pMsg->pThreadInfo->msgQueueList);
    pthread_cond_signal(&pMsg->pThreadInfo->msgQueueNoEmpty);
    pthread_mutex_unlock(&pMsg->pThreadInfo->msgQueueLock);
}

struct msg *rcvMsg(struct thread_info *pThreadInfo)
{
    struct msg *pFirstMsg;
    POINTER_CHECK(pThreadInfo, NULL);
    pthread_mutex_lock(&pThreadInfo->msgQueueLock);
    while (list_empty(&pThreadInfo->msgQueueList))
        pthread_cond_wait(&pThreadInfo->msgQueueNoEmpty, &pThreadInfo->msgQueueLock);
    pFirstMsg = list_first_entry(&pThreadInfo->msgQueueList, struct msg, listMsgQueue);
    list_del(&pFirstMsg->listMsgQueue);
    pthread_mutex_unlock(&pThreadInfo->msgQueueLock);
    return pFirstMsg;
}