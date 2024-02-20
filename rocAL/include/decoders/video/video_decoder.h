/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <cstddef>
#include <iostream>
#include <vector>
#ifdef ROCAL_VIDEO

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avassert.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libswscale/swscale.h>
}
#endif
#include "parameter_factory.h"
#include "parameter_random_crop_decoder.h"

#ifdef ROCAL_VIDEO
#include "rpp.h"
#include "rppdefs.h"
#include "rppi.h"

struct RppLocalData {
    RppPtr_t pSrc;
    RppPtr_t pDst;
    RpptDescPtr pSrcDesc;
    RpptDescPtr pDstDesc;
    RpptInterpolationType interpolationType;
    RpptROI *roiTensorPtrSrc;
    RpptImagePatch *dstImgSizes;
    RpptRoiType roiType;
    rppHandle_t handle;
};

class VideoDecoder {
   public:
    enum class Status {
        OK = 0,
        HEADER_DECODE_FAILED,
        CONTENT_DECODE_FAILED,
        UNSUPPORTED,
        FAILED,
        NO_MEMORY
    };
    enum class ColorFormat {
        GRAY = 0,
        RGB,
        BGR
    };
    virtual VideoDecoder::Status Initialize(const char *src_filename) = 0;
    virtual VideoDecoder::Status Decode(unsigned char *output_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_format) = 0;
    virtual int seek_frame(AVRational avg_frame_rate, AVRational time_base, unsigned frame_number) = 0;
    virtual void release() = 0;
    virtual ~VideoDecoder() = default;
    virtual void set_crop_window(CropWindow &crop_window) = 0;
    virtual void set_rpp_params(RppLocalData *rpp_params) = 0;
    virtual int get_codec_width() = 0;
    virtual int get_codec_height() = 0;
    virtual void set_crop_type(CropType crop_type) = 0;
    virtual void set_resize_width(unsigned resize_width) = 0;
    virtual void set_resize_height(unsigned resize_height) = 0;
    CropWindow _crop_window;
};
#endif
