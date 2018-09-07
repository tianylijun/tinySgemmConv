#ifndef __TINYSGEMM_CONFIG_H
#define __TINYSGEMM_CONFIG_H

#define THREAD_STASTIC_INFO_ENABLE
#define DEBUG_AFFINETY

#ifndef TINY_SGEMM_BLOCK_M
#define TINY_SGEMM_BLOCK_M (128)
#endif

#ifndef TINY_SGEMM_BLOCK_N
#define TINY_SGEMM_BLOCK_N (256)
#endif

#ifndef TINY_SGEMM_BLOCK_K
#define TINY_SGEMM_BLOCK_K (128)
#endif

#ifdef __aarch64__

#ifdef USE_8X16
#define TINY_SGEMM_UNIT_M (8)
#else
#define TINY_SGEMM_UNIT_M (4)
#endif
#define TINY_SGEMM_UNIT_K (4)
#define TINY_SGEMM_UNIT_N (16)

#else /* arm32 */

#define TINY_SGEMM_UNIT_M (4)
#define TINY_SGEMM_UNIT_K (4)
#define TINY_SGEMM_UNIT_N (8)

#endif

#endif