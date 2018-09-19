#ifndef __TINYSGEMMCONV_CONFIG_H
#define __TINYSGEMMCONV_CONFIG_H

#define THREAD_STASTIC_INFO_ENABLE
#define DEBUG_AFFINETY

#define TINY_SGEMM_UNIT_M (4)
#define TINY_SGEMM_UNIT_K (4)

#ifdef __aarch64__

#define TINY_SGEMM_UNIT_N (24)

#else /* arm32 */

#define TINY_SGEMM_UNIT_N (12)

#endif

#endif /* HEAD */