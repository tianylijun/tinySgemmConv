#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <tinySgemmConv.h>
#include <sys/time.h>

int main(int argc, char const *argv[])
{
    int ret = 0, i = 1, loopCnt = 1, num_threads = 4;
    uint32_t inChannels = 3, inputW = 300, inputH = 300, kernelW = 3, kernelH = 3, padW = 0, padH = 0, strideW = 1, strideH = 1, outChannels = 128, dilateW = 1, dilateH = 1, outputW, outputH, M, N, K;
    void *pCtx, *psgemmInstance;
    uint32_t affinity[MAX_CORE_NUMBER] = {1<<0, 1<<1, 1<<2, 1<<3};
    struct timeval beg, end;

    if (argc > 1)  inChannels   = atoi(argv[1]);
    if (argc > 2)  inputW       = atoi(argv[2]);
    if (argc > 3)  inputH       = atoi(argv[3]);
    if (argc > 4)  kernelW      = atoi(argv[4]);
    if (argc > 5)  kernelH      = atoi(argv[5]);
    if (argc > 6)  padW         = atoi(argv[6]);
    if (argc > 7)  padH         = atoi(argv[7]);
    if (argc > 8)  strideW      = atoi(argv[8]);
    if (argc > 9)  strideH      = atoi(argv[9]);
    if (argc > 10) outChannels  = atoi(argv[10]);

    outputW = (inputW + padW*2 - kernelW)/strideW + 1;
    outputH = (inputH + padH*2 - kernelH)/strideH + 1;

    M = outChannels;
    K = inChannels*kernelW*kernelH;
    N = outputW*outputH;

    printf("MNK:[%d %d %d] in: [%d %d %d] K:[%d %d] pad:[%d %d] stride:[%d %d] out:[%d %d %d]\n",
           M, N, K,
           inChannels, inputW, inputH,
           kernelW, kernelH,
           padW, padH,
           strideW, strideH,
           outChannels, outputW, outputH);

    float *pWeight = malloc(M*K*sizeof(float));
    float *pInput  = malloc(inputW*inputH*sizeof(float));
    float *pOutput = malloc(M*N*sizeof(float));
    if (NULL == pWeight || NULL == pInput || NULL == pOutput)
    {
        printf("%s\n", "malloc failed");
        return -1;
    }

    for (i = 0; i < M * K; i++)           pWeight[i] = rand()/10000.0f;
    for (i = 0; i < inputW * inputH; i++) pInput[i]  = rand()/10000.0f;

    ret =  tinySgemmConvInit(num_threads, THREAD_STACK_SIZE, &affinity, &pCtx);
    psgemmInstance = tinySgemmConvCreateInstance(pCtx,
                     pWeight,
                     inChannels,  inputH, inputW,
                     outChannels, kernelH, kernelW,
                     padH, padW,
                     strideH, strideW,
                     dilateH, dilateW,
                     TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP32);

    gettimeofday(&beg, NULL);

    for (i = 0; i < loopCnt; ++i)
        ret = tinySgemmConvProcess(psgemmInstance, pInput, pOutput,
                                   NULL, false, NULL, false, NULL,
                                   TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP32);

    gettimeofday(&end, NULL);
    printf("\ntime: %ld ms, avg time : %.3f ms, loop: %d threads: %d\n", (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/1000, (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/(1000.0*loopCnt), loopCnt, num_threads);

    ret = tinySgemmConvReleaseInstance(psgemmInstance);
    ret = tinySgemmConvDeinit(pCtx);

    if (pWeight)
        free(pWeight);
    if (pInput)
        free(pInput);
    if (pOutput)
        free(pOutput);
    return 0;
}