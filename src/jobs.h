#ifndef _TINY_JOBS_H
#define _TINY_JOBS_H

#ifdef __cplusplus
extern "C" {
#endif

void im2col_cpu_reduce(const float* pInput, const int channels,
                       const int height, const int width,
                       const int kernel_h, const int kernel_w,
                       const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       float* data_col);

#ifdef __cplusplus
}
#endif

#endif