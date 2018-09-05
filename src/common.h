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

static inline unsigned long long rpcc(void)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (unsigned long long)tv.tv_sec * 1000000000ull + tv.tv_usec * 1000;
}

#ifdef __cplusplus
extern "C" {
#endif

void* tinySgemmMalloc(uint32_t size);
void tinySgemmFree(void* ptr);

#ifdef __cplusplus
}
#endif

#endif
