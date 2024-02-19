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

#include "node_fused_crop_resize_video_loader.h"

#include <memory>
#include <numeric>
#include <sstream>
#ifdef ROCAL_VIDEO

FusedCropResizeVideoLoaderNode::FusedCropResizeVideoLoaderNode(Tensor *output, void *device_resources) : Node({}, {output}) {
    _loader_module = std::make_shared<VideoLoaderSharded>(device_resources);
}

void FusedCropResizeVideoLoaderNode::init(unsigned internal_shard_count, const std::string &source_path, StorageType storage_type, DecoderType decoder_type, DecodeMode decoder_mode,
                           unsigned sequence_length, unsigned step, unsigned stride, VideoProperties &video_prop, bool shuffle, bool loop, size_t load_batch_count, RocalMemType mem_type,
                           bool pad_sequences, unsigned num_attempts, std::vector<float> &random_area, std::vector<float> &random_aspect_ratio,
                           unsigned crop_type, unsigned resize_shorter, unsigned resize_width, unsigned resize_height) {
    _decode_mode = decoder_mode;
    if (!_loader_module)
        THROW("ERROR: loader module is not set for FusedCropResizeVideoLoaderNode, cannot initialize")
    if (internal_shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the FusedCropResizeVideoLoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, "", std::map<std::string, std::string>(), shuffle, loop);
    reader_cfg.set_shard_count(internal_shard_count);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_sequence_length(sequence_length);
    reader_cfg.set_frame_step(step);
    reader_cfg.set_frame_stride(stride);
    reader_cfg.set_video_properties(video_prop);
    reader_cfg.set_pad_sequences(pad_sequences);

    auto decoder_cfg = DecoderConfig(decoder_type);
    decoder_cfg.set_random_area(random_area);
    decoder_cfg.set_random_aspect_ratio(random_aspect_ratio);
    decoder_cfg.set_num_attempts(num_attempts);
    decoder_cfg.set_seed(ParameterFactory::instance()->get_seed());
    decoder_cfg.set_crop_type(crop_type);
    decoder_cfg.set_resize_shorter(resize_shorter);
    decoder_cfg.set_resize_width(resize_width);
    decoder_cfg.set_resize_height(resize_height);

    _loader_module->initialize(reader_cfg, decoder_cfg, mem_type, _batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> FusedCropResizeVideoLoaderNode::get_loader_module() {
    if (!_loader_module)
        WRN("FusedCropResizeVideoLoaderNode's loader module is null, not initialized")
    return _loader_module;
}

FusedCropResizeVideoLoaderNode::~FusedCropResizeVideoLoaderNode() {
    _loader_module = nullptr;
}

#endif
