#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
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
    void *pPackB = NULL;

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
	if (pThreadInfo == NULL)
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
		pThreadInfo[i].pPackB = tinySgemmMalloc(TINY_SGEMM_BLOCK_K * TINY_SGEMM_BLOCK_N * sizeof(float));
		if (NULL == pThreadInfo[i].pPackB)
		{
			printf("%s, %d\n", "tinySgemmMalloc packB failed", i);
		    pthread_attr_destroy(&attr);
	        free(pThreadInfo);
	        free(pCtxInner);
		    return -6;
		}
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
            free(pPackB);
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
            free(pPackB);
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
        free(pPackB);
        return -9;
    }

	pMsg = (struct msg *)calloc(MAX_MSGPOOL_NUM, sizeof(struct msg));
    if (NULL == pMsg)
    {
        printf("%s, %d\n", "msg pool malloc failed", MAX_MSGPOOL_NUM * (uint32_t)sizeof(struct msg));
		pthread_mutex_destroy(&pCtxInner->msgLock);
        free(pThreadInfo);
        free(pCtxInner);
        free(pPackB);
        return -10;
    }

	for (uint32_t i = 0; i < MAX_MSGPOOL_NUM; ++i)
		list_add_tail(&pMsg[i].listCtx, &pCtxInner->msgHeadFree);

    pCtxInner->num_threads = num_threads;
    pCtxInner->pThreadInfo = pThreadInfo;
	*pCtx = pCtxInner;
    return ret;
}

static struct msg * fetchMsg(struct tinySgemmConvCtx *pCtx)
{
	struct msg *pMsg = NULL;
	pthread_mutex_lock(&pCtx->msgLock);
	if (list_empty(&pCtx->msgHeadFree))
	{
	    pthread_mutex_unlock(&pCtx->msgLock);
		return NULL;
	}
    pMsg = list_first_entry(&pCtx->msgHeadFree, struct msg, list);
    list_del(&pMsg->listCtx);
	pthread_mutex_unlock(&pCtx->msgLock);
	return pMsg;
}

void returnMsg(struct tinySgemmConvCtx *pCtx, struct msg *pMsg)
{
	pthread_mutex_lock(&pCtx->msgLock);
	list_add_tail(&pMsg->listCtx, &pCtx->msgHeadFree);
	pthread_mutex_unlock(&pCtx->msgLock);
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
    int ret;
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
			pCRange->status = RANGE_STATUS_IDEL;
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

		    ret = pthread_mutex_init(&pCRange->lock, NULL);
		    if (0 != ret)
		    {
		        printf("%s %d, %d\n", "pthread_mutex_init failed at line", __LINE__, ret);
		        free(pMatricRangeInfo);
		        return NULL;
		    }
		    ret = pthread_cond_init(&pCRange->jobDoneCondition, NULL);
		    if (0 != ret)
		    {
		        printf("%s %d, %d\n", "pthread_cond_init failed at line", __LINE__, ret);
		        pthread_mutex_destroy(&pCRange->lock);
				free(pMatricRangeInfo);
		        return NULL;
		    }

    		pCRange++;
    	}
    }

    return pMatricRangeInfo;
}

int tinySgemmConv(void *pCtx, 
    	          void *pA, void *pB, void *pC, 
    	          uint32_t M, uint32_t N, uint32_t K, 
    	          float *pBasis, bool bRelu, float *pPRelu, bool bSharedPrelu,
    	          enum TINY_SGEMM_CONV_DATA_MODE mode)
{
    int ret = 0;
    uint32_t coreId = 0;
    uint32_t littlecore_ids[MAX_CORE_NUMBER];
    uint32_t bigcore_ids[MAX_CORE_NUMBER];
    uint32_t bigCoreNum = 0, littleCoreNum = 0;
    uint32_t bigCoreIdx = 0, littleCoreIdx = 0;
    struct matricRangeInfo* pMarticRangeInfo;
    struct tinySgemmConvCtx *pCtxInner = (struct tinySgemmConvCtx *)pCtx;
    struct list_head waitlist;

    INIT_LIST_HEAD(&waitlist);
    for (uint32_t i = 0; i < pCtxInner->num_threads; ++i)
    {
    	if (pCtxInner->pThreadInfo[i].bigCore)
        {
            if (bigCoreNum <= (MAX_CORE_NUMBER-1))
                bigcore_ids[bigCoreNum++] = i;
            else
            {
    		    printf("%s\n", "bigcore idx overflow");
                return -1;
            }
        }
    	else
        {
            if (littleCoreNum <= (MAX_CORE_NUMBER-1))
    		    littlecore_ids[littleCoreNum++] = i;
            else
            {
                printf("%s\n", "littlecore idx overflow");
                return -2;
            }
        }
    }

    pMarticRangeInfo = createMatircRange(pA, pB, pC, M, N, K, pBasis, bRelu, pPRelu, bSharedPrelu, mode);
    if (NULL == pMarticRangeInfo)
    {
        printf("%s\n", "matricRangeInfo create failed");
        return -3;
    }

    /* assign range to threads */
    for (uint32_t i = 0; i < pMarticRangeInfo->pCRange->YBlocks; ++i)
    {
    	for (uint32_t j = 0; j < pMarticRangeInfo->pCRange->XBlocks; ++j)
    	{
    	    struct msg *pMsg  = fetchMsg(pCtxInner);
            if (NULL == pMsg)
            {
                printf("%s\n", "msg pool used out");
                free(pMarticRangeInfo);
                return -4;
            }
    	    pMsg->cmd         = THREAD_CMD_WORK;
    	    pMsg->pWorkCRange = pMarticRangeInfo->pCRange + i*pMarticRangeInfo->pCRange->XBlocks + j;
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
    	    sendMsg(&pCtxInner->pThreadInfo[coreId], pMsg);
            list_add_tail(&pMsg->pWorkCRange->list, &waitlist);
    	}
    }

    /* wait for all job done */
    while(!list_empty(&waitlist))
	{
        struct rangeInfo *pCRange = list_first_entry(&waitlist, struct rangeInfo, list);
		pthread_mutex_lock(&pCRange->lock);
		while (RANGE_STATUS_DONE != pCRange->status)
	   		pthread_cond_wait(&pCRange->jobDoneCondition, &pCRange->lock);
		pthread_mutex_unlock(&pCRange->lock);
        list_del(&pCRange->list);
	}

    free(pMarticRangeInfo);
    return ret;
}

int tinySgemmConvDeinit(void *pCtx)
{
    struct tinySgemmConvCtx *pCtxInner = (struct tinySgemmConvCtx *)pCtx;
	if (NULL == pCtxInner)
	{
	    printf("%s, deinit failed\n", "NULL pCtx");
		return -1;
	}

	for (uint32_t i = 0; i < pCtxInner->num_threads; i++)
	{
	    struct msg *pMsg = (struct msg *)fetchMsg(pCtxInner);
	    if (NULL != pMsg)
	    {
		    pMsg->cmd = THREAD_CMD_EXIT;
		    sendMsg(&pCtxInner->pThreadInfo[i], pMsg);
	    }
	    else
	    {
	    	printf("%s, %d\n", "thread exit cmd failed", i);
	    	return -2;
	    }
	}

	for (uint32_t i = 0; i < pCtxInner->num_threads; i++)
	{
	    pthread_join(pCtxInner->pThreadInfo[i].thread_id, NULL);
		pthread_mutex_destroy(&pCtxInner->pThreadInfo[i].queue_lock);
		pthread_cond_destroy (&pCtxInner->pThreadInfo[i].noempty);
		free(pCtxInner->pThreadInfo[i].pPackB);
	}

	free(pCtxInner->pThreadInfo);
	free(pCtxInner);
	return 0;
}