#ifndef TCNN_THREADSERVER_H_
#define TCNN_THREADSERVER_H_

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include "list.h"
#include "tinySgemmConv.h"
#include "innerTinySgemmConv.h"
#include "messageQueue.h"

uint32_t getAvaiableCoresMaxFreq(uint32_t (*coreMaxFreqs)[MAX_CORE_NUMBER], uint32_t *maxFreq);
void waitForJobsDone(struct tinySgemmConvCtx *pCtx, struct list_head *workQueue);
struct thread_info *getMinJobsNumThread(struct tinySgemmConvCtx *pCtx, struct list_head *pHead, enum MSG_CMD cmd);
void *sgemm_thread_process(void *args);

#endif