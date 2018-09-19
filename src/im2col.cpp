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

void im2col_cpu_reduce(const float* pInput, const int channels,
                       const int height, const int width,
                       const int kernel_h, const int kernel_w,
                       const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       float* data_col)
{
    int channel, kh, kw, i;
    const int output_h = height - kernel_h + 1;
    const int output_w = width  - kernel_w + 1;
    (void)pad_h;
    (void)pad_w;
    (void)stride_h;
    (void)stride_w;
    (void)dilation_h;
    (void)dilation_w;
    for (channel = 0; channel < channels; channel++)
    {
        const float* data_im_channel  = pInput  + channel*height*width;
        float* data_col_channel = data_col + channel*kernel_h*kernel_w*output_h*output_w;
        for (kh = 0; kh < kernel_h; kh++)
        {
            const float* data_im_kh = data_im_channel + kh*width;
            float* data_col_kh = data_col_channel + kh*kernel_w*output_h*output_w;
            for (kw = 0; kw < kernel_w; kw++)
            {
                const float* data_im_kw = data_im_kh + kw;
                float* data_col_kw = data_col_kh + kw*output_h*output_w;
                for (i = 0; i < output_h; i++)
                {
                    memcpy(data_col_kw+i*output_w, data_im_kw + i*width, output_w*sizeof(float));
                }
            }
        }
    }
}