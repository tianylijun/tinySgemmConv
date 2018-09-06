#ifndef TCNN_THREADSERVER_H_
#define TCNN_THREADSERVER_H_

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include "list.h"
#include "tinySgemmConv.h"
#include "innerTinySgemmConv.h"

void sendMsg(struct thread_info *pThreadInfo, struct msg *pMsg);
void *sgemm_thread_process(void *args);

#endif