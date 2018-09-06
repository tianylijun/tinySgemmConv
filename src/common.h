#ifndef TCNN_COMMON_H_
#define TCNN_COMMON_H_

#include <pthread.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __ARM_NEON
#include "armNeon.h"
#endif

#include "tinySgemmConv.h"

// the alignment of all the allocated buffers
#define MALLOC_MEM_ALIGN    16

#define T_MIN(a, b) (((a)<(b))?(a):(b))
#define T_MAX(a, b) (((a)>(b))?(a):(b))

#if defined(ARMV5) || defined(ARMV6)
#define MB
#define WMB
#else
#define MB   __asm__ __volatile__ ("dmb  ish" : : : "memory")
#define WMB  __asm__ __volatile__ ("dmb  ishst" : : : "memory")
#endif

static inline unsigned alignSize(unsigned sz, int n)
{
    return (sz + n-1) & -n;
}

static inline unsigned long timestamp(void)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (unsigned long)tv.tv_sec * 1000000ul + tv.tv_usec;
}

#ifdef __cplusplus
extern "C" {
#endif

void* tinySgemmMalloc(uint32_t size);
void tinySgemmFree(void* ptr);
uint32_t getAvaiableCoresMaxFreq(uint32_t (*coreMaxFreqs)[MAX_CORE_NUMBER], uint32_t *maxFreq);

#ifdef __cplusplus
}
#endif

#endif
