/*
 * Copyright (C) 2018 tianylijun@163.com. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 * Contributors:
 *     Lee (tianylijun@163.com)
 */

#include <string.h>
#include <arm_neon.h>
#include "im2col.h"

static void im2col_cpu_channel_fp32_fp32(const float* data_im,
        const int height, const int width,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        float* data_col)
{
    int kernel_row,kernel_col,output_rows,output_col,output_cols;
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    for (kernel_row = 0; kernel_row < kernel_h; kernel_row++)
    {
        for (kernel_col = 0; kernel_col < kernel_w; kernel_col++)
        {
            int input_row = -pad_h + kernel_row * dilation_h;
            for (output_rows = output_h; output_rows; output_rows--)
            {
                if ((input_row < 0) || (input_row >= height))
                    for (output_cols = output_w; output_cols; output_cols--)
                        *(data_col++) = 0;
                else
                {
                    int input_col = -pad_w + kernel_col * dilation_w;
                    for (output_col = output_w; output_col; output_col--)
                    {
                        if ((input_row >= 0) && (input_row < height))
                            *(data_col++) = data_im[input_row * width + input_col];
                        else
                            *(data_col++) = 0;
                        input_col += stride_w;
                    }
                }
                input_row += stride_h;
            }
        }
    }
}

static void im2col_cpu_reduce_channel_fp32_fp32(const float* data_im,
        const int height, const int width,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        float* data_col)
{
    const int output_h = height - kernel_h + 1;
    const int output_w = width  - kernel_w + 1;

    for (int kh = 0; kh < kernel_h; kh++)
    {
        const float* data_im_kh = data_im + kh*width;
        float* data_col_kh = data_col + kh*kernel_w*output_h*output_w;
        for (int kw = 0; kw < kernel_w; kw++)
        {
            const float* data_im_kw = data_im_kh + kw;
            float* data_col_kw = data_col_kh + kw*output_h*output_w;
            for (int i = 0; i < output_h; i++)
            {
                memcpy(data_col_kw+i*output_w, data_im_kw + i*width, output_w*sizeof(float));
            }
        }
    }
}

void im2col_channel_fp32_fp32(const float* data_im,
                              const int height, const int width,
                              const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int dilation_h, const int dilation_w,
                              float* data_col)
{
    if ((3 == kernel_h) && (3 == kernel_w) &&
            (0 == pad_h) && (0 == pad_w) &&
            (1 == stride_h) && (1 == stride_w) &&
            (1 == dilation_h) && (1 == dilation_w))
    {
        im2col_cpu_reduce_channel_fp32_fp32(data_im,
                                            height, width,
                                            kernel_h, kernel_w,
                                            pad_h, pad_w,
                                            stride_h, stride_w,
                                            dilation_h, dilation_w,
                                            data_col);
    }
    else
    {
        im2col_cpu_channel_fp32_fp32(data_im,
                                     height, width,
                                     kernel_h, kernel_w,
                                     pad_h, pad_w,
                                     stride_h, stride_w,
                                     dilation_h, dilation_w,
                                     data_col);
    }
}
