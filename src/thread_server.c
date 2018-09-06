#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#include "common.h"
#include "thread_server.h"
#include "list.h"
#include "jobs.h"

void sendMsg(struct thread_info *pThreadInfo, struct msg *pMsg)
{
    pthread_mutex_lock(&pThreadInfo->queue_lock);
    list_add_tail(&pMsg->listThread, &pThreadInfo->msgHead);
    pthread_cond_signal(&pThreadInfo->noempty);
    pthread_mutex_unlock(&pThreadInfo->queue_lock);
}

static struct msg * rcvMsg(struct thread_info *pThreadInfo)
{
    struct msg *pFirstMsg = NULL;
    pthread_mutex_lock(&pThreadInfo->queue_lock);
    while (list_empty(&pThreadInfo->msgHead))
        pthread_cond_wait(&pThreadInfo->noempty, &pThreadInfo->queue_lock);
    pFirstMsg = list_first_entry(&pThreadInfo->msgHead, struct msg, listThread);
    list_del(&pFirstMsg->listThread);
    pthread_mutex_unlock(&pThreadInfo->queue_lock);
    return pFirstMsg;
}

void * sgemm_thread_process(void *args)
{
    int ret = -1, i = 0, availCores = 0, realWorkCores = 0, deadloop = 1;
    cpu_set_t mask;
    struct thread_info *pThreadInfo = (struct thread_info *)args;
    struct tinySgemmConvCtx *pCtx = NULL;
    if (NULL == pThreadInfo)
        return NULL;
    pCtx =  (struct tinySgemmConvCtx *)pThreadInfo->sgemmCtx;
    if (NULL == pCtx)
        return NULL;
    availCores = sysconf(_SC_NPROCESSORS_CONF);
    realWorkCores = T_MIN(availCores, 32);

    if (0 != pThreadInfo->affinity)
    {
        uint32_t availMask = 0, flag = 0;
        CPU_ZERO(&mask);
        ret = sched_getaffinity(0, sizeof(mask), &mask);
        if (0 != ret)
        {
            printf("%s, %d\n", "sched_getaffinity failed", ret);
            return NULL;
        }
        for(i = 0; i < realWorkCores; i++)
            if (CPU_ISSET(i, &mask))
                availMask |= (1U<<i);
        CPU_ZERO(&mask);
        for (int k = 0; k < realWorkCores; ++k)
        {
            if ((pThreadInfo->affinity & (1U<<k)) && (availMask & (1U<<k)))
            {
                flag = 1;
                CPU_SET(k, &mask);
            }
        }
        if (flag)
        {
            ret = sched_setaffinity(0, sizeof(mask), &mask);
            if(0 != ret)
            {
                printf("%s, %d\n", "sched_setaffinity failed", ret);
                return NULL;
            }
        }
        else
            printf("\033[43mWarning:: skip set thread %d affinity(0x%x), as core not avaiable\n", pThreadInfo->index, pThreadInfo->affinity);

#define DEBUG_AFFINETY
#ifdef DEBUG_AFFINETY
        CPU_ZERO(&mask);
        ret = sched_getaffinity(0, sizeof(mask), &mask);
        if (0 != ret)
        {
            printf("%s, %d\n", "sched_getaffinity failed", ret);
            return NULL;
        }
        printf("thread %d can run at core [", pThreadInfo->index);
        for(i = 0; i < realWorkCores; i++)
        {
            if (CPU_ISSET(i, &mask))
                printf(" %d", i);
        }
        printf(" ]\n");
#endif
    }

    INIT_LIST_HEAD(&pThreadInfo->msgHead);
    ret = pthread_mutex_init(&pThreadInfo->queue_lock, NULL);
    if (0 != ret)
    {
        printf("%s, %d\n", "pthread_mutex_init failed", ret);
        return NULL;
    }
    ret = pthread_cond_init(&pThreadInfo->noempty, NULL);
    if (0 != ret)
    {
        printf("%s, %d\n", "pthread_cond_init failed", ret);
        pthread_mutex_destroy(&pThreadInfo->queue_lock);
        return NULL;
    }

    printf("thread %d start\n", pThreadInfo->index);

    while(deadloop)
    {
        struct msg *pMsg = rcvMsg(pThreadInfo);
        WMB;

        pMsg->status = MSG_STATUS_BUSY;
        pMsg->beg = timestamp();
        switch(pMsg->cmd)
        {
        case THREAD_CMD_SGEMM_WORK:
            /* call sgemm top finish the job */

            break;
        case THREAD_CMD_IM2COL_WORK:
            /* call inm2col to finish the job */

            break;
        case THREAD_CMD_EXIT:
            deadloop = 0;
            break;
        }
        pMsg->end = timestamp();
        pthread_mutex_lock(&pMsg->lock);
        pMsg->status = MSG_STATUS_DONE;
        pthread_cond_signal(&pMsg->jobDoneCondition);
        pthread_mutex_unlock(&pMsg->lock);
        WMB;
        printf("process msg, %d\n", pMsg->cmd);
    }

    printf("thread %d exit\n", pThreadInfo->index);
    return NULL;
}
