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

#ifndef __TINYSGEMM_MESSAGEQUEUE_H
#define __TINYSGEMM_MESSAGEQUEUE_H

#include <stdint.h>
#include <pthread.h>
#include "list.h"
#include "tinySgemmConv.h"

enum MSG_STATUS
{
    MSG_STATUS_IDEL,
    MSG_STATUS_BUSY,
    MSG_STATUS_DONE
};

enum MSG_CMD
{
    MSG_CMD_EXIT,
    MSG_CMD_SGEMM,
    MSG_CMD_IM2COL,

    MSG_CMD_END
};

struct MSG_STR
{
    enum MSG_CMD cmd;
    const char *desc;
};

enum SGEMM_DataType
{
    FLOAT32_TYPE,
    FLOAT16_TYPE,
    INT16_TYPE,
    INT8_TYPE
};

struct sgemmJobInfo
{
    uint8_t *pA;
    uint8_t *pBIm2col;
    float *pC;
    uint8_t *pPackB;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t n;
    float *pBasis;
    enum TINY_SGEMM_RELU_TYPE reluType;
    float *pPrelu;
    bool bSharedPrelu;
    float (*int8Scale)[3];
    enum SGEMM_DataType packADataType;
    enum SGEMM_DataType packBDataType;
};

struct im2colJobInfo
{
    float *pB;
    uint8_t *pBIm2col;
    uint32_t kernelW;
    uint32_t kernelH;
    uint32_t strideW;
    uint32_t strideH;
    uint32_t padW;
    uint32_t padH;
    uint32_t dilateW;
    uint32_t dilateH;
    uint32_t height;
    uint32_t width;
    enum SGEMM_DataType outType;
};

struct msg
{
    enum MSG_CMD cmd;
    uint64_t sequenceId;
    enum MSG_STATUS status;
    struct thread_info *pThreadInfo;
    union
    {
        struct sgemmJobInfo sgemmInfo;
        struct im2colJobInfo im2colInfo;
    } JobInfo;
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

const char *MSG2STR(enum MSG_CMD cmd);
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