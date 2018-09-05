#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#include "common.h"
#include "thread_server.h"
#include "list.h"

uint32_t getAvaiableCoresMaxFreq(uint32_t (*coreMaxFreqs)[MAX_CORE_NUMBER], uint32_t *maxFreq)
{
    uint32_t i = 0, availCores = 0;
    cpu_set_t mask;
    uint32_t maxCores = sysconf(_SC_NPROCESSORS_CONF);
    uint32_t max_freq_khz = 0;

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
            char path[256];
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

void sendMsg(struct thread_info *pInfo, struct msg *pMsg)
{
	pthread_mutex_lock(&pInfo->queue_lock);
	list_add_tail(&pMsg->list, &pInfo->head);
	pthread_cond_signal(&pInfo->noempty);
	pthread_mutex_unlock(&pInfo->queue_lock);
}

static struct msg * rcvMsg(struct thread_info *pInfo)
{
	struct msg *pFirstMsg = NULL;
	pthread_mutex_lock(&pInfo->queue_lock);
	while (list_empty(&pInfo->head))
   		pthread_cond_wait(&pInfo->noempty, &pInfo->queue_lock);
    pFirstMsg = list_first_entry(&pInfo->head, struct msg, list);
    list_del(&pFirstMsg->list);
	pthread_mutex_unlock(&pInfo->queue_lock);
	return pFirstMsg;
}

void * sgemm_thread_process(void *args)
{
	int ret = -1, i = 0, availCores = 0, realWorkCores = 0;
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

    INIT_LIST_HEAD(&pThreadInfo->head);
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

    while(1)
    {
        struct msg *pMsg = rcvMsg(pThreadInfo);
        if (THREAD_CMD_EXIT == pMsg->cmd)
        {
            returnMsg(pCtx, pMsg);
            break;
        }
        WMB;

        pMsg->pWorkCRange->status = RANGE_STATUS_BUSY;







        pthread_mutex_lock(&pMsg->pWorkCRange->lock);
        pMsg->pWorkCRange->status = RANGE_STATUS_DONE;
        pthread_cond_signal(&pMsg->pWorkCRange->jobDoneCondition);
        pthread_mutex_unlock(&pMsg->pWorkCRange->lock);

        WMB;
        returnMsg(pCtx, pMsg);
        printf("process msg, %d\n", pMsg->cmd);
    }

    printf("thread %d exit\n", pThreadInfo->index);
    return NULL;
}
