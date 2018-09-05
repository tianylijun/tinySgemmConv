#include <stdio.h>
#include <unistd.h>
#include <tinySgemmConv.h>

int main(int argc, char const *argv[])
{
	int ret = 0;
	void *pCtx;
	uint32_t affinity[MAX_CORE_NUMBER] = {1<<4, 1<<5, 1<<3, 1<<0 | 1<<2};
	ret =  tinySgemmConvInit(8, THREAD_STACK_SIZE, &affinity, &pCtx);

	sleep(2);

	ret = tinySgemmConvDeinit(pCtx);
	return 0;
}