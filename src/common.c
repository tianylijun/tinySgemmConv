#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#include "common.h"

// n Alignment size that must be a power of two
static inline void* alignPtr(void* ptr, int n)
{
    return (void*)(((size_t)ptr + n-1) & -n);
}

void* tinySgemmMalloc(unsigned size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_MEM_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_MEM_ALIGN);
    adata[-1] = udata;
    return adata;
}

void tinySgemmFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

uint32_t getAvaiableCoresMaxFreq(uint32_t (*coreMaxFreqs)[MAX_CORE_NUMBER], uint32_t *maxFreq)
{
    uint32_t i = 0, availCores = 0;
    cpu_set_t mask;
    uint32_t maxCores = sysconf(_SC_NPROCESSORS_CONF);
    uint32_t max_freq_khz = 0;

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
            char path[256];
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