#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
//#define __USE_GNU
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
    unsigned char** adata = (unsigned char**)alignPtr((unsigned char**)udata + 1, MALLOC_MEM_ALIGN);
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
