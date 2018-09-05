#include <stdlib.h>
#include <malloc.h>
#include "common.h"

// the alignment of all the allocated buffers
#define MALLOC_MEM_ALIGN    16

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
