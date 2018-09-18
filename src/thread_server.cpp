#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sched.h>
#include <pthread.h>
#include "common.h"
#include "thread_server.h"
#include "list.h"
#include "jobs.h"
#include "messageQueue.h"

#ifdef DEBUG_AFFINETY
static void showAffinty(struct thread_info *pThreadInfo)
{
    int ret = -1;
    uint32_t availCores = 0, realWorkCores = 0, i;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    availCores = sysconf(_SC_NPROCESSORS_CONF);
    realWorkCores = T_MIN(availCores, 32);
    ret = sched_getaffinity(0, sizeof(mask), &mask);
    if (0 != ret)
    {
        printf("%s, %d\n", "sched_getaffinity failed", ret);
        return;
    }
    printf("thread %d can run at core [", pThreadInfo->index);
    for(i = 0; i < realWorkCores; i++)
    {
        if (CPU_ISSET(i, &mask))
            printf(" %d", i);
    }
    printf(" ]\n");
}
#endif

uint32_t getAvaiableCoresMaxFreq(uint32_t (*coreMaxFreqs)[MAX_CORE_NUMBER], uint32_t *maxFreq)
{
    uint32_t i = 0, availCores = 0;
    cpu_set_t mask;
    uint32_t maxCores = sysconf(_SC_NPROCESSORS_CONF);
    uint32_t max_freq_khz = 0;
    char path[256];

    CPU_ZERO(&mask);
    if (0 != sched_getaffinity(0, sizeof(mask), &mask))
    {
        printf("%s line %d\n", "sched_getaffinity failed", __LINE__);
        return 0;
    }

    for(i = 0; i < maxCores; i++)
    {
        if (CPU_ISSET(i, &mask))
        {
            uint32_t cur_max_freq_khz = 0;
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
            FILE* fp = fopen(path, "rb");
            if (NULL != fp)
            {
                fscanf(fp, "%d", &cur_max_freq_khz);
                fclose(fp);
            }
            if ((NULL != coreMaxFreqs) && (availCores < MAX_CORE_NUMBER))
                coreMaxFreqs[0][availCores] = cur_max_freq_khz;
            max_freq_khz = T_MAX(max_freq_khz, cur_max_freq_khz);
            availCores++;
        }
    }
    if (NULL != maxFreq)
        *maxFreq = max_freq_khz;
    return availCores;
}

void waitForJobsDone(struct tinySgemmConvCtx *pCtx, struct list_head *workQueue)
{
    assert(NULL != pCtx);
    assert(NULL != workQueue);
    while (!list_empty(workQueue))
    {
        struct msg *pMsg = list_first_entry(workQueue, struct msg, listJobsQueue);
        assert(NULL != pMsg);
        pthread_mutex_lock(&pMsg->lock);
        while (MSG_STATUS_DONE != pMsg->status)
            pthread_cond_wait(&pMsg->jobDoneCondition, &pMsg->lock);
        list_del(&pMsg->listJobsQueue);
        pthread_mutex_unlock(&pMsg->lock);
#ifdef HREAD_STASTIC_INFO_ENABLE
        printf("[%d][%d] msg: %llu, done in %llu us\n", pMsg->cmd, pMsg->pThreadInfo->index, pMsg->sequenceId, pMsg->end - pMsg->beg);
#endif
        returnMsg(pCtx, pMsg);
    }
}

struct thread_info *getMinJobsNumThread(struct tinySgemmConvCtx *pCtx, struct list_head *pHead, enum MSG_CMD cmd)
{
    struct thread_info *pThreadInfoRet = NULL;
    uint64_t minJobsDoneNum = 0xffffffffffffffff;
    struct list_head *pos;
    assert(NULL != pHead);

    pthread_mutex_lock(&pCtx->threadLock);
    list_for_each(pos, pHead)
    {
        struct thread_info *pThreadInfo = list_entry(pos, struct thread_info, biglittlecorelist);
        if (MSG_CMD_SGEMM == cmd)
        {
            if (pThreadInfo->sgemmJobsDoneNum < minJobsDoneNum)
            {
                minJobsDoneNum = pThreadInfo->sgemmJobsDoneNum;
                pThreadInfoRet = pThreadInfo;
            }
        }
        else if (MSG_CMD_IM2COL == cmd)
        {
            if (pThreadInfo->im2colJobsDoneNum < minJobsDoneNum)
            {
                minJobsDoneNum = pThreadInfo->im2colJobsDoneNum;
                pThreadInfoRet = pThreadInfo;
            }
        }
    }

    if (MSG_CMD_SGEMM == cmd)
        pThreadInfoRet->sgemmJobsDoneNum++;
    else if (MSG_CMD_IM2COL == cmd)
        pThreadInfoRet->im2colJobsDoneNum++;
    pthread_mutex_unlock(&pCtx->threadLock);
    return pThreadInfoRet;
}

void * sgemm_thread_process(void *args)
{
    int ret = -1;
    cpu_set_t mask;
    uint32_t i = 0, availCores = 0, realWorkCores = 0, deadloop = 1;
    struct thread_info *pThreadInfo = (struct thread_info *)args;
    POINTER_CHECK(pThreadInfo, NULL);
    availCores = sysconf(_SC_NPROCESSORS_CONF);
    realWorkCores = T_MIN(availCores, MAX_CORE_NUMBER);

    if (0xffffffff != pThreadInfo->affinity)
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
        for (uint32_t k = 0; k < realWorkCores; ++k)
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
#ifdef DEBUG_AFFINETY
        showAffinty(pThreadInfo);
#endif
    }

    INIT_LIST_HEAD(&pThreadInfo->msgQueueList);
    ret = pthread_mutex_init(&pThreadInfo->msgQueueLock, NULL);
    if (0 != ret)
    {
        printf("%s, %d\n", "pthread_mutex_init failed", ret);
        return NULL;
    }
    ret = pthread_cond_init(&pThreadInfo->msgQueueNoEmpty, NULL);
    if (0 != ret)
    {
        printf("%s, %d\n", "pthread_cond_init failed", ret);
        pthread_mutex_destroy(&pThreadInfo->msgQueueLock);
        return NULL;
    }
    printf("thread %d start\n", pThreadInfo->index);

    while(deadloop)
    {
        struct msg *pMsg = rcvMsg(pThreadInfo);
        WMB;

        pMsg->status = MSG_STATUS_BUSY;
        pMsg->timeStampBeg = timestamp();
        switch(pMsg->cmd)
        {
        case MSG_CMD_SGEMM:
            /* call sgemm top finish the job */
            if (TINY_SGEMM_UNIT_N == pMsg->JobInfo.sgemmInfo.n)
            {
                /* do  TINY_SGEMM_UNIT_M * K * TINY_SGEMM_UNIT_N */
                /* packB K*TINY_SGEMM_UNIT_N */

                /* do sgemm */
            }
            else
            {
                /* do  TINY_SGEMM_UNIT_M * K * leftN */
                uint32_t leftN = pMsg->JobInfo.sgemmInfo.n;
                uint32_t leftNDiv8 = leftN>>3;
                uint32_t leftNHas4 = (leftN>>2)&1;
                uint32_t leftNHas2 = (leftN>>1)&1;
                uint32_t leftNHas1 = leftN&1;

                /* packB K*leftN */

                /* do sgemm */
                for (uint32_t i = 0; i < leftNDiv8; ++i)
                {
                    /* code */
                }

                if (leftNHas4)
                {

                }

                if (leftNHas2)
                {
                    /* code */
                }

                if (leftNHas1)
                {
                    /* code */
                }
            }
            break;
        case MSG_CMD_IM2COL:
            /* call inm2col to finish the job */

            break;
        case MSG_CMD_EXIT:
            deadloop = 0;
            /* clear msg queue */
            INIT_LIST_HEAD(&pThreadInfo->msgQueueList);
            break;
        }
        pMsg->timeStampEnd = timestamp();
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
