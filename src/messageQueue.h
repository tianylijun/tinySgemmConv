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

enum MSG_CMD
{
    MSG_CMD_EXIT,
    MSG_CMD_SGEMM,
    MSG_CMD_IM2COL
};

enum DataType
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
    enum DataType inAType;
    enum DataType inBType;
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
    enum DataType outType;
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