#ifndef __TINYSGEMMCONV_ARMNEON_H_
#define __TINYSGEMMCONV_ARMNEON_H_

#include <arm_neon.h>

#if 0

#define ARM_LOAD_PREFETCH_16(addr)
#define ARM_LOAD_PREFETCH_32(addr)
#define ARM_LOAD_PREFETCH_64(addr)
#define ARM_LOAD_PREFETCH_128(addr)
#define ARM_LOAD_PREFETCH_256(addr)
#define ARM_LOAD_PREFETCH_512(addr)
#define ARM_STORE_PREFETCH_16(addr)
#define ARM_STORE_PREFETCH_32(addr)
#define ARM_STORE_PREFETCH_64(addr)
#define ARM_STORE_PREFETCH_128(addr)
#define ARM_STORE_PREFETCH_256(addr)
#define ARM_STORE_PREFETCH_512(addr)

#else

#ifdef __aarch64__
#define ARM_LOAD_PREFETCH_16(addr) asm volatile(\
                                "prfm PLDL1KEEP, [%0, #16] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_32(addr) asm volatile(\
                                "prfm PLDL1KEEP, [%0, #32] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_64(addr) asm volatile(\
                                "prfm PLDL1KEEP, [%0, #64] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_128(addr) asm volatile(\
                                "prfm PLDL1KEEP, [%0, #128] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_256(addr) asm volatile(\
                                "prfm PLDL1KEEP, [%0, #256] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_16(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #16] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_32(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #32] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_64(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #64] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_128(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #128] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_256(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #256] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_512(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #512] \n"\
                                :\
                                :"r"(addr));
#else /* __aarch64__ */
#define ARM_LOAD_PREFETCH_16(addr) asm volatile(\
                                "pld [%0, #16] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_32(addr) asm volatile(\
                                "pld [%0, #32] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_64(addr) asm volatile(\
                                "pld [%0, #64] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_128(addr) asm volatile(\
                                "pld [%0, #128] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_256(addr) asm volatile(\
                                "pld [%0, #256] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_16(addr) asm volatile(\
                                "pld [%0, #16] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_32(addr) asm volatile(\
                                "pld [%0, #32] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_64(addr) asm volatile(\
                                "pld [%0, #64] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_128(addr) asm volatile(\
                                "pld [%0, #128] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_256(addr) asm volatile(\
                                "pld [%0, #256] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_512(addr) asm volatile(\
                                "pld [%0, #512] \n"\
                                :\
                                :"r"(addr));
#endif /* __aarch64__ */
#endif /* if 0 */

#ifndef __aarch64__

static inline void vst1q_f32_x2(void *address, float32x4x2_t vector)
{
    vst1q_f32((float32_t*) address,     vector.val[0]);
    vst1q_f32((float32_t*) address + 4, vector.val[1]);
    return;
}
static inline void vst1_f32_x4(void *address, float32x2x4_t vector)
{
    vst1_f32((float32_t*) address,      vector.val[0]);
    vst1_f32((float32_t*) address + 2,  vector.val[1]);
    vst1_f32((float32_t*) address + 4,  vector.val[2]);
    vst1_f32((float32_t*) address + 6,  vector.val[3]);
    return;
}
static inline void vst1q_f32_x4(void *address, float32x4x4_t vector)
{
    vst1q_f32((float32_t*) address,      vector.val[0]);
    vst1q_f32((float32_t*) address + 4,  vector.val[1]);
    vst1q_f32((float32_t*) address + 8,  vector.val[2]);
    vst1q_f32((float32_t*) address + 12, vector.val[3]);
    return;
}

#endif

#ifdef __clang__

static inline float32x4_t vld1q_f32_f16(const void* address)
{
    return vcvt_f32_f16(vld1_f16((const __fp16*) address));
}

static inline float32x4_t vld1q_f32_f16_aligned(const void* address)
{
    return vcvt_f32_f16(vld1_f16((const __fp16*)
                                 __builtin_assume_aligned(address, sizeof(float16x4_t))));
}
static inline void vst1q_f16_f32(void* address, float32x4_t vector)
{
    vst1_f16((__fp16*) address, vcvt_f16_f32(vector));
}
static inline void vst1q_f16_f32_x2(void* address, float32x4x2_t *vector)
{
    vst1_f16((__fp16*) address,      vcvt_f16_f32(vector->val[0]));
    vst1_f16((__fp16*) address + 4,  vcvt_f16_f32(vector->val[1]));
}
static inline void vst1q_f16_f32_x4(void* address, float32x4x4_t *vector)
{
    vst1_f16((__fp16*) address,      vcvt_f16_f32(vector->val[0]));
    vst1_f16((__fp16*) address + 4,  vcvt_f16_f32(vector->val[1]));
    vst1_f16((__fp16*) address + 8,  vcvt_f16_f32(vector->val[2]));
    vst1_f16((__fp16*) address + 12, vcvt_f16_f32(vector->val[3]));
}
static inline void vst1q_f16_f32_aligned(void* address, float32x4_t vector)
{
    vst1_f16((__fp16*) __builtin_assume_aligned(address, sizeof(float16x4_t)),
             vcvt_f16_f32(vector));
}
static inline float16x4_t vld1_f16_neon(void* address)
{
    return (float16x4_t) vld1_u16((const uint16_t*) address);
}

#else /* clang */

#ifdef __aarch64__

typedef void __fp16;
typedef int16x4_t float16x4_t;
typedef struct float16x4x2_t
{
    float16x4_t val[2];
} float16x4x2_t;

#define vcvt_f32_f16(a)                                                  \
   __extension__                                                         \
     ({                                                                  \
        uint16x4_t a_ = (a);                                             \
        float32x4_t result;                                              \
        __asm__ ("fcvtl %0.4s, %1.4h"                                    \
                 : "=w"(result)                                          \
                 : "w"(a_)                                               \
                 : /* No clobbers */);                                   \
        result;                                                          \
      })

#define vcvt_f16_f32(a)                                                  \
   __extension__                                                         \
     ({                                                                  \
        float32x4_t a_ = (a);                                            \
        uint16x4_t result;                                               \
        __asm__ ("fcvtn %0.4h, %1.4s"                                    \
                 : "=w"(result)                                          \
                 : "w"(a_)                                               \
                 : /* No clobbers */);                                   \
        result;                                                          \
      })

static inline void vst1q_f16_f32(void* address, float32x4_t vector)
{
    vst1_u16((uint16_t*) address, (uint16x4_t) vcvt_f16_f32(vector));
}
static inline void vst1q_f16_f32_x2(void* address, float32x4x2_t *vector)
{
    vst1_u16((uint16_t*) address,      vcvt_f16_f32(vector->val[0]));
    vst1_u16((uint16_t*) address + 4,  vcvt_f16_f32(vector->val[1]));
}
static inline void vst1q_f16_f32_x4(void* address, float32x4x4_t *vector)
{
    vst1_u16((uint16_t*) address,      vcvt_f16_f32(vector->val[0]));
    vst1_u16((uint16_t*) address + 4,  vcvt_f16_f32(vector->val[1]));
    vst1_u16((uint16_t*) address + 8,  vcvt_f16_f32(vector->val[2]));
    vst1_u16((uint16_t*) address + 12, vcvt_f16_f32(vector->val[3]));
}
static inline float16x4_t vld1_f16_neon(const void* address)
{
    return (float16x4_t) vld1_u16((const uint16_t*) address);
}
static inline float16x4x2_t vld1_f16_x2(const void* address)
{
    float16x4x2_t result;
    result.val[0] = (float16x4_t) vld1_u16((const uint16_t*) address);
    result.val[1] = (float16x4_t) vld1_u16((const uint16_t*) address + 4);
    return result;
}
static inline int16x4x2_t vld1_s16_x2(const void* address)
{
    int16x4x2_t result;
    result.val[0] = vld1_s16((const int16_t*) address);
    result.val[1] = vld1_s16((const int16_t*) address + 4);
    return result;
}
static inline int16x4x4_t vld1_s16_x4(const void* address)
{
    int16x4x4_t result;
    result.val[0] = vld1_s16((const int16_t*) address);
    result.val[1] = vld1_s16((const int16_t*) address + 4);
    result.val[2] = vld1_s16((const int16_t*) address + 8);
    result.val[3] = vld1_s16((const int16_t*) address + 12);
    return result;
}
static inline void vst1_s16_x4(const void* address, int16x4x4_t vector)
{
    vst1_s16((int16_t*) address,      vector.val[0]);
    vst1_s16((int16_t*) address + 4,  vector.val[1]);
    vst1_s16((int16_t*) address + 8,  vector.val[2]);
    vst1_s16((int16_t*) address + 12, vector.val[3]);
    return;
}
static inline float32x4x2_t vld1q_f32_x2(const void* address)
{
    float32x4x2_t result;
    result.val[0] = vld1q_f32((const float32_t*) address);
    result.val[1] = vld1q_f32((const float32_t*) address + 4);
    return result;
}
static inline void vst1q_f32_x2(void *address, float32x4x2_t vector)
{
    vst1q_f32((float32_t*) address,     vector.val[0]);
    vst1q_f32((float32_t*) address + 4, vector.val[1]);
    return;
}
static inline void vst1q_f32_x3(void *address, float32x4x3_t vector)
{
    vst1q_f32((float32_t*) address,      vector.val[0]);
    vst1q_f32((float32_t*) address + 4,  vector.val[1]);
    vst1q_f32((float32_t*) address + 8,  vector.val[2]);
    return;
}
static inline void vst1q_f32_x4(void *address, float32x4x4_t vector)
{
    vst1q_f32((float32_t*) address,      vector.val[0]);
    vst1q_f32((float32_t*) address + 4,  vector.val[1]);
    vst1q_f32((float32_t*) address + 8,  vector.val[2]);
    vst1q_f32((float32_t*) address + 12, vector.val[3]);
    return;
}
static inline void vst1_f32_x4(void *address, float32x2x4_t vector)
{
    vst1_f32((float32_t*) address,      vector.val[0]);
    vst1_f32((float32_t*) address + 2,  vector.val[1]);
    vst1_f32((float32_t*) address + 4,  vector.val[2]);
    vst1_f32((float32_t*) address + 6,  vector.val[3]);
    return;
}
static inline float32x4x3_t vld1q_f32_x3(const void* address)
{
    float32x4x3_t result;
    result.val[0] = vld1q_f32((const float32_t*) address);
    result.val[1] = vld1q_f32((const float32_t*) address + 4);
    result.val[2] = vld1q_f32((const float32_t*) address + 8);
    return result;
}
static inline float32x4x4_t vld1q_f32_x4(const void* address)
{
    float32x4x4_t result;
    result.val[0] = vld1q_f32((const float32_t*) address);
    result.val[1] = vld1q_f32((const float32_t*) address + 4);
    result.val[2] = vld1q_f32((const float32_t*) address + 8);
    result.val[3] = vld1q_f32((const float32_t*) address + 12);
    return result;
}

#else /* __aarch64__ */

typedef struct float16x4x2_t
{
    float16x4_t val[2];
} float16x4x2_t;

static inline float16x4_t vld1_f16_neon(const void* address)
{
    return (float16x4_t) vld1_u16((const uint16_t*) address);
}
static inline void vst1_f16(const void* address, uint16x4_t vector)
{
    vst1_u16((uint16_t*) address, vector);
}
// GCC 4.x doesn't support vst1_f16/vld1_f16, workaround.
static inline float32x4_t vld1q_f32_f16(const void* address)
{
    return vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address));
}

static inline float32x4_t vld1q_f32_f16_aligned(const void* address)
{
    return vcvt_f32_f16((float16x4_t)
                        vld1_u16((const uint16_t*) __builtin_assume_aligned(address, sizeof(float16x4_t))));
}

static inline void vst1q_f16_f32(void* address, float32x4_t vector)
{
    vst1_u16((uint16_t*) address, (uint16x4_t)vcvt_f16_f32(vector));
}

static inline void vst1q_f16_f32_x2(void* address, float32x4x2_t *vector)
{
    vst1_u16((uint16_t*) address,      (uint16x4_t)vcvt_f16_f32(vector->val[0]));
    vst1_u16((uint16_t*) address + 4,  (uint16x4_t)vcvt_f16_f32(vector->val[1]));
}

static inline void vst1q_f16_f32_x4(void* address, float32x4x4_t *vector)
{
    vst1_u16((uint16_t*) address,      (uint16x4_t)vcvt_f16_f32(vector->val[0]));
    vst1_u16((uint16_t*) address + 4,  (uint16x4_t)vcvt_f16_f32(vector->val[1]));
    vst1_u16((uint16_t*) address + 8,  (uint16x4_t)vcvt_f16_f32(vector->val[2]));
    vst1_u16((uint16_t*) address + 12, (uint16x4_t)vcvt_f16_f32(vector->val[3]));
}

static inline void vst1q_f16_f32_aligned(void* address, float32x4_t vector)
{
    vst1_u16((uint16_t*) __builtin_assume_aligned(address, sizeof(uint16x4_t)),
             (uint16x4_t)vcvt_f16_f32(vector));
}
#endif /* __aarch64__ */

#endif  /* clang */

#endif /* __TINYSGEMMCONV_ARMNEON_H_ */
