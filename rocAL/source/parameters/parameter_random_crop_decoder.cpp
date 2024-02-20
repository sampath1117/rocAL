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

#include "parameter_random_crop_decoder.h"

#include <cassert>

// Initializing the random generator so all objects of the class can share it.
thread_local std::mt19937 RocalRandomCropDecParam::_rand_gen(time(0));

RocalRandomCropDecParam::RocalRandomCropDecParam(
    AspectRatioRange aspect_ratio_range,
    AreaRange area_range,
    int64_t seed,
    int num_attempts,
    int batch_size,
    CropType crop_type)
    : _aspect_ratio_range(aspect_ratio_range), _aspect_ratio_log_dis(std::log(aspect_ratio_range.first), std::log(aspect_ratio_range.second)), _area_dis(area_range.first, area_range.second), _seed(seed), _num_attempts(num_attempts), _batch_size(batch_size), _crop_type(crop_type){
    _seeds.resize(_batch_size);
}

RocalRandomCropDecParam::RocalRandomCropDecParam(
    int64_t seed,
    std::vector<float> scales,
    int batch_size,
    CropType crop_type)
    :_seed(seed), _scales(scales), _batch_size(batch_size), _crop_type(crop_type){
    _seeds.resize(_batch_size);
}

CropWindow RocalRandomCropDecParam::generate_crop_window_implementation(const Shape& shape) {
    assert(shape.size() == 2);
    CropWindow crop;
    int H = shape[0], W = shape[1];
    if (W <= 0 || H <= 0) {
        return crop;
    }

    if(_crop_type == CropType::RANDOM_CROP) {
        float min_wh_ratio = _aspect_ratio_range.first;
        float max_wh_ratio = _aspect_ratio_range.second;
        float max_hw_ratio = 1 / _aspect_ratio_range.first;
        float min_area = W * H * _area_dis.a();
        int maxW = std::max<int>(1, H * max_wh_ratio);
        int maxH = std::max<int>(1, W * max_hw_ratio);
        // detect two impossible cases early
        if (H * maxW < min_area) {  // image too wide
            crop.set_shape(H, maxW);
        } else if (W * maxH < min_area) {  // image too tall
            crop.set_shape(maxH, W);
        } else {  // it can still fail for very small images when size granularity matters
            int attempts_left = _num_attempts;
            for (; attempts_left > 0; attempts_left--) {
                float scale = _area_dis(_rand_gen);
                size_t original_area = H * W;
                float target_area = scale * original_area;
                float ratio = std::exp(_aspect_ratio_log_dis(_rand_gen));
                auto w = static_cast<int>(
                    std::roundf(sqrtf(target_area * ratio)));
                auto h = static_cast<int>(
                    std::roundf(sqrtf(target_area / ratio)));
                w = std::max(1, w);
                h = std::max(1, h);
                crop.set_shape(h, w);
                ratio = static_cast<float>(w) / h;
                if (w <= W && h <= H && ratio >= min_wh_ratio && ratio <= max_wh_ratio)
                    break;
            }
            if (attempts_left <= 0) {
                float max_area = _area_dis.b() * W * H;
                float ratio = static_cast<float>(W) / H;
                if (ratio > max_wh_ratio) {
                    crop.set_shape(H, maxW);
                } else if (ratio < min_wh_ratio) {
                    crop.set_shape(maxH, W);
                } else {
                    crop.set_shape(H, W);
                }
                float scale = std::min(1.0f, max_area / (crop.W * crop.H));
                crop.W = std::max<int>(1, crop.W * std::sqrt(scale));
                crop.H = std::max<int>(1, crop.H * std::sqrt(scale));
            }
        }
        crop.x = std::uniform_int_distribution<int>(0, W - crop.W)(_rand_gen);
        crop.y = std::uniform_int_distribution<int>(0, H - crop.H)(_rand_gen);
    } else if (_crop_type == CropType::CORNER_CROP) {
        // 1. Takes the shortest dimension
        // 2. Multiplies that dimension with the random scale value selected. This is the corner crop size
        // 3. Get a random corner crop position and generate x, y as per the corner crop position
        int num_scales = _scales.size() - 1;
        int scale_position = std::uniform_int_distribution<int>(0, num_scales)(_rand_gen);
        float scale = _scales[scale_position]; // get the random value here
        int min_dimension = std::min(H, W);
        int crop_size = int(min_dimension * scale);
        crop.H = crop_size;
        crop.W = crop_size;
        int crop_position = std::uniform_int_distribution<int>(0, 4)(_rand_gen);
        if (crop_position == 0) { // center
            crop.y = std::round((H - crop_size) / 2.0f);
            crop.x = std::round((W - crop_size) / 2.0f);
        } else if (crop_position == 1) { // top left
            crop.y = 0;
            crop.x = 0;
        } else if (crop_position == 2) { // top right
            crop.y = 0;
            crop.x = W - crop_size;
        } else if (crop_position == 3) { // bottom left
            crop.y = H - crop_size;
            crop.x = 0;
        } else if (crop_position == 4) { // bottom right
            crop.y = H - crop_size;
            crop.x = W - crop_size;
        }
    }
    return crop;
}

// seed the rng for the instance and return the random crop window.
CropWindow RocalRandomCropDecParam::generate_crop_window(const Shape& shape, const int instance) {
    _rand_gen.seed(_seeds[instance]);
    return generate_crop_window_implementation(shape);
}

void RocalRandomCropDecParam::generate_random_seeds() {
    ParameterFactory::instance()->generate_seed();  // Renew and regenerate
    std::seed_seq seq{ParameterFactory::instance()->get_seed()};
    seq.generate(_seeds.begin(), _seeds.end());
}
