#ifndef __TINYSGEMMCONV_CONFIG_H
#define __TINYSGEMMCONV_CONFIG_H

#define THREAD_STASTIC_INFO_ENABLE
#define DEBUG_AFFINETY

#ifdef __aarch64__

#define TINY_SGEMM_UNIT_M (8)
#define TINY_SGEMM_UNIT_K (4)
#define TINY_SGEMM_UNIT_N (16)

#ifndef TINY_SGEMM_BLOCK_M
#define TINY_SGEMM_BLOCK_M (128)
#endif

#ifndef TINY_SGEMM_BLOCK_N
#define TINY_SGEMM_BLOCK_N (256)
#endif

#ifndef TINY_SGEMM_BLOCK_K
#define TINY_SGEMM_BLOCK_K (64)
#endif

#else /* arm32 */

#define TINY_SGEMM_UNIT_M (4)
#define TINY_SGEMM_UNIT_K (4)
#define TINY_SGEMM_UNIT_N (12)

#ifndef TINY_SGEMM_BLOCK_M
#define TINY_SGEMM_BLOCK_M (128)
#endif

#ifndef TINY_SGEMM_BLOCK_N
#define TINY_SGEMM_BLOCK_N (256)
#endif

#ifndef TINY_SGEMM_BLOCK_K
#define TINY_SGEMM_BLOCK_K (128)
#endif

#endif

#endif /* HEAD */