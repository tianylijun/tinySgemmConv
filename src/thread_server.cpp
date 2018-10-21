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
#include <assert.h>
#include <sched.h>
#include <pthread.h>
#include "common.h"
#include "thread_server.h"
#include "list.h"
#include "im2col.h"
#include "messageQueue.h"
#include "sgemm.h"
#include "pack.h"

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

void waitForJobsDone(struct tinySgemmConvCtx *pCtx, struct list_head *jobsQueue)
{
    assert(NULL != pCtx);
    assert(NULL != jobsQueue);
    while (!list_empty(jobsQueue))
    {
        struct msg *pMsg = list_first_entry(jobsQueue, struct msg, listJobsQueue);
        assert(NULL != pMsg);
        pthread_mutex_lock(&pMsg->lock);
        while (MSG_STATUS_DONE != pMsg->status)
            pthread_cond_wait(&pMsg->jobDoneCondition, &pMsg->lock);
        list_del(&pMsg->listJobsQueue);
        pthread_mutex_unlock(&pMsg->lock);
#ifdef HREAD_STASTIC_INFO_ENABLE
        printf("[%d][%d] msg: %llu, done in %llu us\n", pMsg->cmd, pMsg->pThreadInfo->index, pMsg->sequenceId, pMsg->timeStampEnd - pMsg->timeStampBeg);
#endif
        returnMsg(pCtx, pMsg);
    }
}

#ifdef MOSTFREE_JOBS_NUM
static struct thread_info *getMinJobsNumThread(struct tinySgemmConvCtx *pCtx, struct list_head *pHead, enum MSG_CMD cmd)
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

#else

static struct thread_info *getMinTimeThread(struct tinySgemmConvCtx *pCtx, struct list_head *pHead, enum MSG_CMD cmd)
{
    struct thread_info *pThreadInfoRet = NULL;
    uint64_t minTIme = 0xffffffffffffffff;
    struct list_head *pos;
    assert(NULL != pHead);

    (void)cmd;
    pthread_mutex_lock(&pCtx->threadLock);
    list_for_each(pos, pHead)
    {
        struct thread_info *pThreadInfo = list_entry(pos, struct thread_info, biglittlecorelist);
        //uint64_t curThreadTime = pThreadInfo->totalMsgTime[cmd];
        uint64_t curThreadTime = pThreadInfo->totalMsgTime[0];
        if (curThreadTime < minTIme)
        {
            minTIme = curThreadTime;
            pThreadInfoRet = pThreadInfo;
        }
    }
    pthread_mutex_unlock(&pCtx->threadLock);
    return pThreadInfoRet;
}
#endif

struct thread_info *getMostFreeThread(struct tinySgemmConvCtx *pCtx, struct list_head *pHead, enum MSG_CMD cmd)
{
#ifdef MOSTFREE_JOBS_NUM
    return getMinJobsNumThread(pCtx, pHead, cmd);
#else
    return getMinTimeThread(pCtx, pHead, cmd);
#endif
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
            printf("Warning:: skip set thread %d affinity(0x%x), as core not avaiable\n", pThreadInfo->index, pThreadInfo->affinity);
    }

#ifdef DEBUG_AFFINETY
    showAffinty(pThreadInfo);
#endif

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
    pThreadInfo->status = 1;

    while(deadloop)
    {
        struct msg *pMsg = rcvMsg(pThreadInfo);
        assert(NULL != pMsg);
        pMsg->status = MSG_STATUS_BUSY;
        pMsg->timeStampBeg = timestamp();
        //printf("[%llu] thread [%d][%d] rcvMsg %s\n", pMsg->sequenceId, pThreadInfo->index, list_number(&pThreadInfo->msgQueueList), MSG2STR(pMsg->cmd));
        switch(pMsg->cmd)
        {
        case MSG_CMD_SGEMM:
            /* call sgemm finish the job */
            if (TINY_SGEMM_UNIT_N == pMsg->JobInfo.sgemmInfo.n)
            {
                /* do TINY_SGEMM_UNIT_M * K * TINY_SGEMM_UNIT_N */
                if (FLOAT32_TYPE == pMsg->JobInfo.sgemmInfo.packADataType &&
                        FLOAT32_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType)
                {
                    /* packB K*TINY_SGEMM_UNIT_N */
                    tinySgemmConvPackBUnitN_fp32_fp32((float *)pMsg->JobInfo.sgemmInfo.pBIm2col, (float *)pMsg->JobInfo.sgemmInfo.pPackB, pMsg->JobInfo.sgemmInfo.K, pMsg->JobInfo.sgemmInfo.N);

                    /* do sgemm */
#ifdef __aarch64__
                    sgemmMxKx24_fp32 ((float *)pMsg->JobInfo.sgemmInfo.pA,
                                      (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                                      pMsg->JobInfo.sgemmInfo.pC,
                                      pMsg->JobInfo.sgemmInfo.M,
                                      pMsg->JobInfo.sgemmInfo.N,
                                      pMsg->JobInfo.sgemmInfo.K,
                                      (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                                      pMsg->JobInfo.sgemmInfo.pPrelu,
                                      pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                                      pMsg->JobInfo.sgemmInfo.pBasis);
#else
                    sgemmMxKx12_fp32 ((float *)pMsg->JobInfo.sgemmInfo.pA,
                                      (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                                      pMsg->JobInfo.sgemmInfo.pC,
                                      pMsg->JobInfo.sgemmInfo.M,
                                      pMsg->JobInfo.sgemmInfo.N,
                                      pMsg->JobInfo.sgemmInfo.K,
                                      (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                                      pMsg->JobInfo.sgemmInfo.pPrelu,
                                      pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                                      pMsg->JobInfo.sgemmInfo.pBasis);
#endif
                }
                else if (FLOAT16_TYPE == pMsg->JobInfo.sgemmInfo.packADataType && FLOAT16_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType)
                {
                    /* code */
                }
                else if (INT16_TYPE == pMsg->JobInfo.sgemmInfo.packADataType && INT16_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType)
                {
                    /* code */
                }
                else if (INT8_TYPE == pMsg->JobInfo.sgemmInfo.packADataType && INT8_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType)
                {
                    /* code */
                }
            }
            else
            {
                uint32_t NHas8, NHas4, NHas2, NHas1, K, N;
#ifdef __aarch64__
                uint32_t NHas16;
#endif
                N      = pMsg->JobInfo.sgemmInfo.N;
                K      = pMsg->JobInfo.sgemmInfo.K;
#ifdef __aarch64__
                NHas16 = (N>>4)&1;
#endif
                NHas8  = (N>>3)&1;
                NHas4  = (N>>2)&1;
                NHas2  = (N>>1)&1;
                NHas1  = N&1;

                /* do TINY_SGEMM_UNIT_M * K * leftN */
                if (FLOAT32_TYPE == pMsg->JobInfo.sgemmInfo.packADataType &&
                        FLOAT32_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType)
                {
                    /* packB K*leftN */
                    tinySgemmConvPackBLeftN_fp32_fp32((float *)pMsg->JobInfo.sgemmInfo.pBIm2col, (float *)pMsg->JobInfo.sgemmInfo.pPackB, pMsg->JobInfo.sgemmInfo.K, pMsg->JobInfo.sgemmInfo.N);
#ifdef __aarch64__
                    /* do sgemm */
                    if (NHas16)
                    {
                        sgemmMxKx16_fp32 ((float *)pMsg->JobInfo.sgemmInfo.pA,
                                          (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                                          pMsg->JobInfo.sgemmInfo.pC,
                                          pMsg->JobInfo.sgemmInfo.M,
                                          pMsg->JobInfo.sgemmInfo.N,
                                          pMsg->JobInfo.sgemmInfo.K,
                                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                                          pMsg->JobInfo.sgemmInfo.pPrelu,
                                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                                          pMsg->JobInfo.sgemmInfo.pBasis);
                        pMsg->JobInfo.sgemmInfo.pPackB = (uint8_t *)((float *)pMsg->JobInfo.sgemmInfo.pPackB + 16*K);
                        pMsg->JobInfo.sgemmInfo.pC += 16;
                    }
#endif
                    if (NHas8)
                    {
                        sgemmMxKx8_fp32 ((float *)pMsg->JobInfo.sgemmInfo.pA,
                                         (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                                         pMsg->JobInfo.sgemmInfo.pC,
                                         pMsg->JobInfo.sgemmInfo.M,
                                         pMsg->JobInfo.sgemmInfo.N,
                                         pMsg->JobInfo.sgemmInfo.K,
                                         (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                                         pMsg->JobInfo.sgemmInfo.pPrelu,
                                         pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                                         pMsg->JobInfo.sgemmInfo.pBasis);
                        pMsg->JobInfo.sgemmInfo.pPackB = (uint8_t *)((float *)pMsg->JobInfo.sgemmInfo.pPackB + 8*K);
                        pMsg->JobInfo.sgemmInfo.pC += 8;
                    }

                    if (NHas4)
                    {
                        sgemmMxKx4_fp32  ((float *)pMsg->JobInfo.sgemmInfo.pA,
                                          (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                                          pMsg->JobInfo.sgemmInfo.pC,
                                          pMsg->JobInfo.sgemmInfo.M,
                                          pMsg->JobInfo.sgemmInfo.N,
                                          pMsg->JobInfo.sgemmInfo.K,
                                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                                          pMsg->JobInfo.sgemmInfo.pPrelu,
                                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                                          pMsg->JobInfo.sgemmInfo.pBasis);
                        pMsg->JobInfo.sgemmInfo.pPackB = (uint8_t *)((float *)pMsg->JobInfo.sgemmInfo.pPackB + 4*K);
                        pMsg->JobInfo.sgemmInfo.pC += 4;
                    }

                    if (NHas2)
                    {
                        sgemmMxKx2_fp32  ((float *)pMsg->JobInfo.sgemmInfo.pA,
                                          (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                                          pMsg->JobInfo.sgemmInfo.pC,
                                          pMsg->JobInfo.sgemmInfo.M,
                                          pMsg->JobInfo.sgemmInfo.N,
                                          pMsg->JobInfo.sgemmInfo.K,
                                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                                          pMsg->JobInfo.sgemmInfo.pPrelu,
                                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                                          pMsg->JobInfo.sgemmInfo.pBasis);
                        pMsg->JobInfo.sgemmInfo.pPackB = (uint8_t *)((float *)pMsg->JobInfo.sgemmInfo.pPackB + 2*K);
                        pMsg->JobInfo.sgemmInfo.pC += 2;
                    }

                    if (NHas1)
                    {
                        sgemmMxKx1_fp32  ((float *)pMsg->JobInfo.sgemmInfo.pA,
                                          (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                                          pMsg->JobInfo.sgemmInfo.pC,
                                          pMsg->JobInfo.sgemmInfo.M,
                                          pMsg->JobInfo.sgemmInfo.N,
                                          pMsg->JobInfo.sgemmInfo.K,
                                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                                          pMsg->JobInfo.sgemmInfo.pPrelu,
                                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                                          pMsg->JobInfo.sgemmInfo.pBasis);
                    }
                }
                else if (FLOAT16_TYPE == pMsg->JobInfo.sgemmInfo.packADataType && FLOAT16_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType)
                {
                    /* code */
                }
                else if (INT16_TYPE == pMsg->JobInfo.sgemmInfo.packADataType && INT16_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType)
                {
                    /* code */
                }
                else if (INT8_TYPE == pMsg->JobInfo.sgemmInfo.packADataType && INT8_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType)
                {
                    /* code */
                }
            }
            break;
        case MSG_CMD_IM2COL:
            if (FLOAT32_TYPE == pMsg->JobInfo.im2colInfo.outType)
            {
                im2col_channel_fp32_fp32 (pMsg->JobInfo.im2colInfo.pB,      (float *)pMsg->JobInfo.im2colInfo.pBIm2col,
                                          pMsg->JobInfo.im2colInfo.height,  pMsg->JobInfo.im2colInfo.width,
                                          pMsg->JobInfo.im2colInfo.kernelH, pMsg->JobInfo.im2colInfo.kernelW,
                                          pMsg->JobInfo.im2colInfo.padH,    pMsg->JobInfo.im2colInfo.padW,
                                          pMsg->JobInfo.im2colInfo.strideH, pMsg->JobInfo.im2colInfo.strideW,
                                          pMsg->JobInfo.im2colInfo.dilateH, pMsg->JobInfo.im2colInfo.dilateW,
                                          pMsg->JobInfo.im2colInfo.pad_only_bottom, pMsg->JobInfo.im2colInfo.pad_only_right);
            }
            else
            {
                printf("im2col data type %d not supported, %s %d\n", pMsg->JobInfo.im2colInfo.outType, __FILE__, __LINE__);
            }
            break;
        case MSG_CMD_EXIT:
            deadloop = 0;
            /* clear msg queue */
            INIT_LIST_HEAD(&pThreadInfo->msgQueueList);
            break;
        default:
            printf("Err: msg %s not process\n", MSG2STR(pMsg->cmd));
            break;
        }

        pMsg->timeStampEnd = timestamp();
        switch(pMsg->cmd)
        {
        case MSG_CMD_SGEMM:
        case MSG_CMD_IM2COL:
            //pThreadInfo->totalMsgTime[pMsg->cmd] += pMsg->timeStampEnd - pMsg->timeStampBeg;
            pThreadInfo->totalMsgTime[0] += pMsg->timeStampEnd - pMsg->timeStampBeg;
            break;
        default:
            break;
        }
        pthread_mutex_lock(&pMsg->lock);
        pMsg->status = MSG_STATUS_DONE;
        pthread_cond_signal(&pMsg->jobDoneCondition);
        pthread_mutex_unlock(&pMsg->lock);

        //printf("[%llu] thread [%d] process msg %s \n", pMsg->sequenceId, pThreadInfo->index, MSG2STR(pMsg->cmd));
    }

    printf("thread %d exit\n", pThreadInfo->index);
    return NULL;
}
