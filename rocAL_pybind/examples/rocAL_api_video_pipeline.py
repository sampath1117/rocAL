# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import numpy as np
import cupy as cp
from parse_config import parse_args


class ROCALVideoIterator(object):
    """
    ROCALVideoIterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, display=False, sequence_length=3, device = "cpu", device_id=0):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)

        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.device = device
        self.device_id = device_id
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.batch_size = self.loader._batch_size
        self.rim = self.loader.get_remaining_images()
        self.display = display
        self.iter_num = 0
        self.sequence_length = sequence_length
        print("____________REMAINING IMAGES____________:", self.rim)
        self.output_list = self.dimensions = self.dtype = None

    def next(self):
        return self.__next__()

    def __next__(self):
        if (self.loader.is_empty()):
            raise StopIteration

        if self.loader.rocal_run() != 0:
            raise StopIteration
        self.output_tensor_list = self.loader.get_output_tensors()
        self.iter_num += 1
        # Copy output from buffer to numpy array
        if self.output_list is None:  # Checking if output_list is empty and initializing the buffers
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                self.dimensions = self.output_tensor_list[i].dimensions()
                if self.device == "cpu":
                    self.dtype = self.output_tensor_list[i].dtype()
                    if len(self.dimensions) == 5:
                        first_dim = self.dimensions[0] * self.dimensions[1]
                        self.output = np.empty((first_dim, self.dimensions[2], self.dimensions[3], self.dimensions[4]) , dtype=self.dtype)
                    else:
                        self.output = np.empty((self.dimensions) , dtype=self.dtype)
                else:
                    self.dtype = self.output_tensor_list[i].dtype()
                    with cp.cuda.Device(device=self.device_id):
                        if len(self.dimensions) == 5:
                            first_dim = self.dimensions[0] * self.dimensions[1]
                            self.output = cp.empty((first_dim, self.dimensions[2], self.dimensions[3], self.dimensions[4]) , dtype=self.dtype)
                        else:
                            self.output = cp.empty((self.dimensions) , dtype=self.dtype)
                if self.device == "cpu":
                    self.output_tensor_list[i].copy_data(self.output)
                else:
                    self.output_tensor_list[i].copy_data(self.output.data.ptr)
                self.output_list.append(self.output)
        else:
            for i in range(len(self.output_tensor_list)):
                    self.output_tensor_list[i].copy_data(self.output_list[i])

        # Display Frames in a video sequence
        for i in range(len(self.output_list)):
            for list_i in range(len(self.output_list[i])):
                draw_frames(self.output_list[i][list_i], i, list_i, self.iter_num, self.tensor_format)
        return self.output_list

    def reset(self):
        self.loader.rocal_reset_loaders()

    def __iter__(self):
        return self

    def __del__(self):
        self.loader.rocal_release()


def draw_frames(image, output_list_num, batch_sample_idx, iter_idx, layout,):
    # image is expected as a tensor, bboxes as numpy
    import cv2
    if isinstance(image, cp.ndarray):
        image = image.get()
    if layout == (types.NFCHW or types.NCHW):
        image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    import os
    if not os.path.exists("OUTPUT_FOLDER/VIDEO_READER"):
        os.makedirs("OUTPUT_FOLDER/VIDEO_READER")
    image = cv2.UMat(image).get()
    cv2.imwrite("OUTPUT_FOLDER/VIDEO_READER/" + "output_list_num_" + str(output_list_num) +
                "iter_" + str(iter_idx) + "_batch_sample_"+str(batch_sample_idx) + ".png", image)


def main():
    # Args
    args = parse_args()
    video_path = args.video_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    user_sequence_length = args.sequence_length
    display = args.display
    num_threads = args.num_threads
    random_seed = args.seed
    tensor_format = types.NFHWC if args.NHWC else types.NFCHW
    tensor_dtype = types.FLOAT16 if args.fp16 else types.FLOAT
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=0, seed=random_seed, rocal_cpu=rocal_cpu,
                    tensor_layout=tensor_format, tensor_dtype=tensor_dtype)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        images = fn.readers.video(file_root=video_path, sequence_length=user_sequence_length,
                                  random_shuffle=False, image_type=types.RGB, resize_width = 640, resize_height = 480,
                                  scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
        # elements_extracted1, elements_extracted = fn.element_extract(images, element_map=[1,2])
        pipe.set_outputs(images)
    # Build the pipeline
    pipe.build()
    # Dataloader
    data_loader = ROCALVideoIterator(pipe, multiplier=pipe._multiplier, tensor_layout= types.NHWC, device="gpu" if rocal_cpu is False else "cpu", 
                                     offset=pipe._offset, display=display, sequence_length=user_sequence_length)
    import timeit
    start = timeit.default_timer()
    # Enumerate over the Dataloader
    for epoch in range(int(args.num_epochs)):
        print("EPOCH:::::", epoch)
        for i, it in enumerate(data_loader, 0):
            if args.print_tensor:
                print("**************", i, "*******************")
                print("**************starts*******************")
                print("\n IMAGES : \n", it)
                print("**************ends*******************")
                print("**************", i, "*******************")
        data_loader.reset()
    # Your statements here
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)
    print("##############################  VIDEO READER  SUCCESS  ############################")


if __name__ == '__main__':
    main()
