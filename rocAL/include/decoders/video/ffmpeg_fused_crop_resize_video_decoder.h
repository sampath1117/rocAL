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

#include "video_decoder.h"

#ifdef ROCAL_VIDEO
class FFmpegFusedCropResizeVideoDecoder : public VideoDecoder {
   public:
    //! Default constructor
    FFmpegFusedCropResizeVideoDecoder();
    VideoDecoder::Status Initialize(const char *src_filename) override;
    VideoDecoder::Status Decode(unsigned char *output_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_format) override;
    int seek_frame(AVRational avg_frame_rate, AVRational time_base, unsigned frame_number) override;
    void release() override;
    ~FFmpegFusedCropResizeVideoDecoder() override;
    int get_codec_width() override { return _codec_width; }
    int get_codec_height() override { return _codec_height; }

   private:
    const char *_src_filename = NULL;
    AVFormatContext *_fmt_ctx = NULL;
    AVCodecContext *_video_dec_ctx = NULL;
    AVCodec *_decoder = NULL;
    AVStream *_video_stream = NULL;
    int _video_stream_idx = -1;
    AVPixelFormat _dec_pix_fmt;
    int _codec_width, _codec_height;
    CropWindow _crop_window;
    RppLocalData _rpp_params;
    CropType _crop_type;
    unsigned _resize_width, _resize_height;
    void set_crop_window(CropWindow &crop_window) override { _crop_window = crop_window; }
    void set_rpp_params(RppLocalData *rpp_params) override { _rpp_params = std::move(*rpp_params); }
    void set_crop_type(CropType crop_type) override { _crop_type = crop_type; }
    void set_resize_width(unsigned resize_width) { _resize_width = resize_width; }
    void set_resize_height(unsigned resize_height) { _resize_height = resize_height; }
};
#endif
