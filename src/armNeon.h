#ifndef TCNN_ARMNEON_H_
#define TCNN_ARMNEON_H_

#include <arm_neon.h>

#ifndef __clang__
static inline void vst1q_u32_x4(uint32_t *pDst, uint32x4x4_t src32x4x4)
{
    vst1q_u32(pDst, src32x4x4.val[0]);
    vst1q_u32(pDst+4, src32x4x4.val[1]);
    vst1q_u32(pDst+8, src32x4x4.val[2]);
    vst1q_u32(pDst+12, src32x4x4.val[3]);
}

static inline void vst1_u32_x4(uint32_t *pDst, uint32x2x4_t src32x2x4)
{
    vst1_u32(pDst, src32x2x4.val[0]);
    vst1_u32(pDst+2, src32x2x4.val[1]);
    vst1_u32(pDst+4, src32x2x4.val[2]);
    vst1_u32(pDst+6, src32x2x4.val[3]);
}

static inline void vst1q_u32_x2(uint32_t *pDst, uint32x4x2_t src32x4x2)
{
    vst1q_u32(pDst, src32x4x2.val[0]);
    vst1q_u32(pDst+4, src32x4x2.val[1]);
}
#else /* gcc */

#endif

#endif
