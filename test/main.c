#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <tinySgemmConv.h>
#include <sys/time.h>
#include <math.h>
#include <arm_neon.h>

#ifndef MAX
#define MAX(a,b) ((a)>(b))?(a):(b)
#endif
#ifndef MIN
#define MIN(a,b) ((a)<(b))?(a):(b)
#endif

void conv3x3s1_neon(float *input, int inch, int h, int w, int inChannelSize, float *output, int outch, int outh, int outw, int outChannelSize, const float* kernel, const float* bias, unsigned num_threads);
void makeborder(float *dst, float *src, unsigned channels, unsigned w, unsigned h, unsigned padw, unsigned padh, unsigned channelAlignSize, float val, unsigned num_threads);
void padChannelBuffer(float *dst, float *src, unsigned channelSize, unsigned channelPad, unsigned channels, unsigned num_threads);
void padChannelBufferInv(float *dst, float *src, unsigned channelSize, unsigned channelPad, unsigned channels, unsigned num_threads);
unsigned alignSize(unsigned sz, int n);

static void relu_padchannel(float *input, float *output, uint32_t output_channels, uint32_t channelSize, uint32_t padOutChannel, unsigned num_threads)
{
    int size = channelSize;
    int inSize = size + padOutChannel;
    float32x4_t vzerof32x4 = vdupq_n_f32(0.f);

    #pragma omp parallel for num_threads(num_threads)
    for (int q=0; q<output_channels; q++)
    {
        const float* inPtr = input + q*inSize;
        float* outPtr = output + q*size;
        int i = 0;
#ifdef __ARM_NEON
        for (; i < size - 4; i += 4)
        {
            float32x4_t vsrcf32x4 = vld1q_f32(inPtr + i);
            uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4, vzerof32x4);
            vsrcf32x4 = vbslq_f32(vmasku32x4, vzerof32x4, vsrcf32x4);
            vst1q_f32(&outPtr[i], vsrcf32x4);
        }
#endif
        for (; i<size; i++)
        {
            if (inPtr[i] < 0)
                outPtr[i] = 0;
            else
                outPtr[i] = inPtr[i];
        }
    }
}

static void relu6_padchannel(float *input, float *output, uint32_t output_channels, uint32_t channelSize, uint32_t padOutChannel, unsigned num_threads)
{
    int size = channelSize;
    int inSize = size + padOutChannel;
    float32x4_t zero = vdupq_n_f32(0.f);
    float32x4_t six = vdupq_n_f32(6.0f);

    #pragma omp parallel for num_threads(num_threads)
    for (int q=0; q<output_channels; q++)
    {
        const float* inPtr = input + q*inSize;
        float* outPtr = output + q*size;
        int i = 0;
#ifdef __ARM_NEON
        for (; i < size - 4; i += 4)
        {
            float32x4_t vinput = vld1q_f32(inPtr + i);
            vinput = vmaxq_f32(vinput, zero);
            vinput = vminq_f32(vinput, six);
            vst1q_f32(outPtr + i, vinput);
        }
#endif
        for (; i<size; i++)
            outPtr[i] = MIN(MAX(inPtr[i], 0.0f), 6.0f);
    }
}


static void showResult(float *pOut, uint32_t data_size)
{
    for(int i = 0 ; i < data_size && i < 64; i++)
    {
        if ((0 != i)&& (0 == i % 16))
            printf("\n");
        printf("%f, ", pOut[i]);
    }
    printf("\n----------------------------------------\n");
}

int main(int argc, char const *argv[])
{
    int ret = 0, i = 1, j = 0, outLoopCnt = 1, loopCnt = 10, num_threads = 4;
    uint32_t inChannels = 3, inputW = 300, inputH = 300, kernelW = 3, kernelH = 3, padW = 0, padH = 0, strideW = 1, strideH = 1, outChannels = 128, dilateW = 1, dilateH = 1, outputW, outputH, M, N, K;
    void *pCtx, *psgemmInstance;
    enum TINY_SGEMM_RELU_TYPE reluType = TINY_SGEMM_RELU_TYPE_NORELU;
    bool fuse_relu = true, fuse_relu6 = false;
    //uint32_t affinity[MAX_CORE_NUMBER] = {0xf0, 0xf0, 0xf0, 0xf};
    uint32_t affinity[MAX_CORE_NUMBER] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
    struct timeval beg, end;

    if (argc > 1)  num_threads  = atoi(argv[i++]);
    if (argc > 2)  inChannels   = atoi(argv[i++]);
    if (argc > 3)  inputW       = atoi(argv[i++]);
    if (argc > 4)  inputH       = atoi(argv[i++]);
    if (argc > 5)  kernelW      = atoi(argv[i++]);
    if (argc > 6)  kernelH      = atoi(argv[i++]);
    if (argc > 7)  padW         = atoi(argv[i++]);
    if (argc > 8)  padH         = atoi(argv[i++]);
    if (argc > 9)  strideW      = atoi(argv[i++]);
    if (argc > 10) strideH      = atoi(argv[i++]);
    if (argc > 11) outChannels  = atoi(argv[i++]);

    if (fuse_relu)
        reluType = TINY_SGEMM_RELU_TYPE_RELU;
    else if (fuse_relu6)
        reluType = TINY_SGEMM_RELU_TYPE_RELU6;

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
    float *pInput  = malloc(inChannels*inputW*inputH*sizeof(float));
    float *pOutput = malloc(2*M*N*sizeof(float));
    if (NULL == pWeight || NULL == pInput || NULL == pOutput)
    {
        printf("%s\n", "malloc failed");
        return -1;
    }

    for (i = 0; i < M * K; i++)                    pWeight[i] = (powf(-1, (rand()%2))*(rand()%1000))*1.0f;
    for (i = 0; i < inChannels*inputW*inputH; i++) pInput[i]  = (powf(-1, (rand()%2))*(rand()%1000))*1.0f;

#if 1
    gettimeofday(&beg, NULL);
    int padInputSize  = alignSize((inputH+2*padH)*(inputW+2*padW), 16) - (inputH+2*padH)*(inputW+2*padW);
    int padOutChannel = alignSize(outputH*outputW, 16) - outputH*outputW;
    float *align_input = NULL, *align_output = NULL;
    printf("[%d %d]\n", padInputSize, padOutChannel);
    if (0 != (padW + padH) || 0 != padInputSize)
        align_input = malloc(sizeof(float) * inChannels * alignSize((inputH+2*padH)*(inputW+2*padW),   16));
    if (0 != padOutChannel)
        align_output = malloc(sizeof(float) * outChannels * alignSize(outputH*outputW, 16));

    for (i = 0; i < loopCnt; ++i)
    {
        if (0 != (padW + padH))
            makeborder(align_input, pInput, inChannels, inputW, inputH, padW, padH, 16, .0f, num_threads);
        else
        {
            if (padInputSize) padChannelBuffer(align_input, pInput, inputH*inputW, padInputSize, inChannels, num_threads);
            else align_input = pInput;
        }
        if (0 == padOutChannel) align_output = pOutput + M*N;

        conv3x3s1_neon(align_input,  inChannels,  inputH+2*padH,  inputW+2*padW,  (inputH+2*padH)*(inputW+2*padW)+padInputSize,
                       align_output, outChannels, outputH, outputW, outputH*outputW + padOutChannel,
                       pWeight, NULL, num_threads);

        if (fuse_relu)
            relu_padchannel(align_output, pOutput + M*N, M, N, padOutChannel, num_threads);
        else if (fuse_relu6)
            relu6_padchannel(align_output, pOutput + M*N, M, N, padOutChannel, num_threads);
        else if (padOutChannel)
            padChannelBufferInv(pOutput + M*N, align_output, outputH*outputW, padOutChannel, outChannels, num_threads);
    }
    gettimeofday(&end, NULL);
    printf("[Ref] time: %ld ms, avg time : %.3f ms, loop: %d threads: %d\n\n", (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/1000, (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/(1000.0*loopCnt), loopCnt, num_threads);
    showResult(pOutput + M*N, M*N);
    if (0 != (padW + padH) || 0 != padInputSize)
        free(align_input);
    if (align_output)
        free(align_output);
    printf("direct end\n");
#endif

    for (j = 0; j < outLoopCnt; ++j)
    {
        ret =  tinySgemmConvInit(num_threads, THREAD_STACK_SIZE, &affinity, true, &pCtx);
        printf("Init ok\n");
        psgemmInstance = tinySgemmConvCreateInstance(pCtx,
                         pWeight,
                         inChannels,  inputH, inputW,
                         outChannels, kernelH, kernelW,
                         padH, padW,
                         strideH, strideW,
                         dilateH, dilateW,
                         false,
                         TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP32);
        printf("Instance create ok\n");

        gettimeofday(&beg, NULL);

        for (i = 0; i < loopCnt; ++i)
            ret = tinySgemmConvProcess(psgemmInstance, pInput, pOutput,
                                       NULL, reluType, NULL, false, NULL,
                                       TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP32);

        gettimeofday(&end, NULL);
        ret = tinySgemmConvReleaseInstance(psgemmInstance);
        ret = tinySgemmConvDeinit(pCtx);
        printf("[%02d/%02d] time: %ld ms, avg time : %.3f ms, loop: %d threads: %d\n\n", j, outLoopCnt, (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/1000, (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/(1000.0*loopCnt), loopCnt, num_threads);
    }
    showResult(pOutput, M*N);

    int sameFlag = 1;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            //if (fabs(fabs(*(pOutput + i * N + j)) - fabs(*(pOutput + M*N + i * N + j)))/fabs(*(pOutput + i * N + j)) > 0.1f)
            if (*(pOutput + i * N + j) != *(pOutput + M*N + i * N + j))
            {
                printf("[%d, %d] %f %f\n", i, j, *(pOutput + i * N + j), *(pOutput + M*N + i * N + j));
                //printf("[%d, %d] %f %f\n", i, j + 1, *(pOutput + i * N + j + 1), *(pOutput + M*N + i * N + j + 1));
                //printf("[%d, %d] %f %f\n", i, j + 2, *(pOutput + i * N + j + 2), *(pOutput + M*N + i * N + j + 2));
                //printf("[%d, %d] %f %f\n", i, j + 3, *(pOutput + i * N + j + 3), *(pOutput + M*N + i * N + j + 3));
                sameFlag = 0;
                break;
            }
        }
        if (0 == sameFlag)
            break;
    }

    if (pWeight)
        free(pWeight);
    if (pInput)
        free(pInput);
    if (pOutput)
        free(pOutput);
    printf("compare %s\n",(sameFlag)?"same":"diff");
    return 0;
}
