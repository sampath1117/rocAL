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

#include "ffmpeg_video_decoder.h"

#include <commons.h>
#include <stdio.h>
#include "rpp.h"
#include "rppdefs.h"
#include "rppi.h"

inline void set_descriptor_dims_and_strides(RpptDescPtr descPtr, int noOfImages, int maxHeight, int maxWidth, int numChannels, int offsetInBytes)
{
    descPtr->numDims = 4;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->n = noOfImages;
    descPtr->h = maxHeight;
    descPtr->w = maxWidth;
    descPtr->c = numChannels;

    // set strides
    if (descPtr->layout == RpptLayout::NHWC)
    {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->c * descPtr->w;
        descPtr->strides.wStride = descPtr->c;
        descPtr->strides.cStride = 1;
    }
    else if(descPtr->layout == RpptLayout::NCHW)
    {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.cStride = descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->w;
        descPtr->strides.wStride = 1;
    }
}

#ifdef ROCAL_VIDEO
FFmpegVideoDecoder::FFmpegVideoDecoder(){};

int FFmpegVideoDecoder::seek_frame(AVRational avg_frame_rate, AVRational time_base, unsigned frame_number) {
    auto seek_time = av_rescale_q((int64_t)frame_number, av_inv_q(avg_frame_rate), AV_TIME_BASE_Q);
    int64_t select_frame_pts = av_rescale_q((int64_t)frame_number, av_inv_q(avg_frame_rate), time_base);
    int ret = av_seek_frame(_fmt_ctx, -1, seek_time, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        ERR("Error in seeking frame..Unable to seek the given frame in a video");
        return ret;
    }
    return select_frame_pts;
}

// Seeks to the frame_number in the video file and decodes each frame in the sequence.
VideoDecoder::Status FFmpegVideoDecoder::Decode(unsigned char *out_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_pix_format) {
    VideoDecoder::Status status = Status::OK;
    // std::cout << "_codec_width, codec_height: " << _codec_width << ", " << _codec_height << std::endl;
    // std::cout << "out_width, out_height: " << out_width << ", " << out_height << std::endl;
    // std::cout << "out_stride: " << out_stride << std::endl;
    
    // std::cout << "printing crop values"<< std::endl;
    // std::cout << "x, y, width, height: " <<_crop_window.x << ", "<<_crop_window.y<<", "<<_crop_window.W <<", "<<_crop_window.H<<std::endl;
    
    // Initialize the SwsContext
    SwsContext *swsctx = nullptr;
    if ((out_width != _codec_width) || (out_height != _codec_height) || (out_pix_format != _dec_pix_fmt)) {
        swsctx = sws_getCachedContext(nullptr, _codec_width, _codec_height, _dec_pix_fmt,
                                      _codec_width, _codec_height, out_pix_format, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!swsctx) {
            ERR("Fail to get sws_getCachedContext");
            return Status::FAILED;
        }
    }
    int select_frame_pts = seek_frame(_video_stream->avg_frame_rate, _video_stream->time_base, seek_frame_number);
    if (select_frame_pts < 0) {
        ERR("Error in seeking frame..Unable to seek the given frame in a video");
        return Status::FAILED;
    }
    unsigned frame_count = 0;
    unsigned filled_frames_count = 0;
    bool end_of_stream = false;
    bool sequence_filled = false;
    uint8_t *dst_data[4] = {0};
    int dst_linesize[4] = {0};
    Rpp32u channels = 3;
    int image_size = out_height * out_stride * sizeof(unsigned char);
    int input_image_size = _codec_height * _codec_width * channels * sizeof(unsigned char);
    AVPacket pkt;
    AVFrame *dec_frame = av_frame_alloc();
    
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;
    srcDescPtr->dataType = RpptDataType::U8;
    dstDescPtr->dataType = RpptDataType::U8;
    
    if(channels == 1){
        srcDescPtr->layout = RpptLayout::NCHW;        
        dstDescPtr->layout = RpptLayout::NCHW;       
    } else {
        srcDescPtr->layout = RpptLayout::NHWC;    
        dstDescPtr->layout = RpptLayout::NHWC;
    }
    
    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    RpptImagePatch *dstImgSizes = static_cast<RpptImagePatch *>(calloc(1, sizeof(RpptImagePatch)));
    dstImgSizes->width = out_width;
    dstImgSizes->height = out_height;
    
    // Set ROI tensors types for src/dst
    RpptROI *roiTensorPtrSrc = static_cast<RpptROI *>(calloc(1, sizeof(RpptROI)));
    roiTensorPtrSrc[0].xywhROI.xy.x = _crop_window.x;
    roiTensorPtrSrc[0].xywhROI.xy.y = _crop_window.y;
    roiTensorPtrSrc[0].xywhROI.roiWidth = _crop_window.W;
    roiTensorPtrSrc[0].xywhROI.roiHeight = _crop_window.H;
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;
    set_descriptor_dims_and_strides(srcDescPtr, 1, _codec_height, _codec_width, channels, 0);
    set_descriptor_dims_and_strides(dstDescPtr, 1, out_height, out_width, channels, 0);
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, 1, 1);
    
    if (!dec_frame) {
        ERR("Could not allocate dec_frame");
        return Status::NO_MEMORY;
    }
    do {
        int ret;
        // read packet from input file
        ret = av_read_frame(_fmt_ctx, &pkt);
        if (ret < 0 && ret != AVERROR_EOF) {
            ERR("Fail to av_read_frame: ret=" + TOSTR(ret));
            status = Status::FAILED;
            break;
        }
        if (ret == 0 && pkt.stream_index != _video_stream_idx) continue;
        end_of_stream = (ret == AVERROR_EOF);
        if (end_of_stream) {
            // null packet for bumping process
            pkt.data = nullptr;
            pkt.size = 0;
        }

        // submit the packet to the decoder
        ret = avcodec_send_packet(_video_dec_ctx, &pkt);
        if (ret < 0) {
            ERR("Error while sending packet to the decoder\n");
            status = Status::FAILED;
            break;
        }

        // get all the available frames from the decoder
        while (ret >= 0) {
            ret = avcodec_receive_frame(_video_dec_ctx, dec_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if ((dec_frame->pts < select_frame_pts) || (ret < 0)) continue;
            if (frame_count % stride == 0) {
                if (swsctx)
                {
                    std::vector<unsigned char> temp_buffer;
                    temp_buffer.resize(input_image_size);
                    dst_data[0] = temp_buffer.data();
                    dst_linesize[0] = _codec_width * channels;
                    sws_scale(swsctx, dec_frame->data, dec_frame->linesize, 0, dec_frame->height, dst_data, dst_linesize);
                    std::cout << "coming to swsctx" << std::endl;
                    void *input = reinterpret_cast<void *>(dst_data[0]);
                    void *output = reinterpret_cast<void *>(out_buffer);
                    rppt_resize_host(input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                    temp_buffer.clear();
                }
                else {
                    std::cout << "coming to else case" << std::endl;
                    // copy from frame to out_buffer
                    memcpy(out_buffer, dec_frame->data[0], dec_frame->linesize[0] * out_height);
                }
                out_buffer = out_buffer + image_size;
                filled_frames_count += 1;
            }
            ++frame_count;
            av_frame_unref(dec_frame);
            if (frame_count == sequence_length * stride) {
                sequence_filled = true;
                break;
            }
        }
        av_packet_unref(&pkt);
        if (sequence_filled) break;
    } while (!end_of_stream);
    if (!sequence_filled && (sequence_length != filled_frames_count)) {
        memset(out_buffer, 0, image_size * (sequence_length - filled_frames_count));
    }
    avcodec_flush_buffers(_video_dec_ctx);
    av_frame_free(&dec_frame);
    sws_freeContext(swsctx);
    return status;
}

// Initialize will open a new decoder and initialize the context
VideoDecoder::Status FFmpegVideoDecoder::Initialize(const char *src_filename) {
    VideoDecoder::Status status = Status::OK;
    int ret;
    AVDictionary *opts = NULL;

    // open input file, and initialize the context required for decoding
    _fmt_ctx = avformat_alloc_context();
    _src_filename = src_filename;
    if (avformat_open_input(&_fmt_ctx, src_filename, NULL, NULL) < 0) {
        ERR("Couldn't Open video file " + STR(src_filename));
        return Status::FAILED;
    }
    if (avformat_find_stream_info(_fmt_ctx, NULL) < 0) {
        ERR("av_find_stream_info error");
        return Status::FAILED;
    }
    ret = av_find_best_stream(_fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (ret < 0) {
        ERR("Could not find %s stream in input file " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " " + STR(src_filename));
        return Status::FAILED;
    }
    _video_stream_idx = ret;
    _video_stream = _fmt_ctx->streams[_video_stream_idx];
    if (!_video_stream) {
        ERR("Could not find video stream in the input, aborting");
        return Status::FAILED;
    }

    // find decoder for the stream
    _decoder = avcodec_find_decoder(_video_stream->codecpar->codec_id);
    if (!_decoder) {
        ERR("Failed to find " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec");
        return Status::FAILED;
    }

    // Allocate a codec context for the decoder
    _video_dec_ctx = avcodec_alloc_context3(_decoder);
    if (!_video_dec_ctx) {
        ERR("Failed to allocate the " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec context");
        return Status::NO_MEMORY;
    }

    // Copy codec parameters from input stream to output codec context
    if ((ret = avcodec_parameters_to_context(_video_dec_ctx, _video_stream->codecpar)) < 0) {
        ERR("Failed to copy " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec parameters to decoder context");
        return Status::FAILED;
    }

    // Init the decoders
    if ((ret = avcodec_open2(_video_dec_ctx, _decoder, &opts)) < 0) {
        ERR("Failed to open " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec");
        return Status::FAILED;
    }
    _dec_pix_fmt = _video_dec_ctx->pix_fmt;
    _codec_width = _video_stream->codecpar->width;
    _codec_height = _video_stream->codecpar->height;
    return status;
}

void FFmpegVideoDecoder::release() {
    if (_video_dec_ctx)
        avcodec_free_context(&_video_dec_ctx);
    if (_fmt_ctx)
        avformat_close_input(&_fmt_ctx);
}

FFmpegVideoDecoder::~FFmpegVideoDecoder() {
    release();
}
#endif
