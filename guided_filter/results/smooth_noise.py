

# -*- coding: utf-8 -*-
## @package guided_filter.results.smooth_noise
#
#  Simple guided filter test.
#  @author      tody
#  @date        2015/08/26

import os
import numpy as np
import matplotlib.pyplot as plt

from guided_filter.datasets.google_image import dataFile
from guided_filter.results.results import resultFile
from guided_filter.io_util.image import loadRGB
from guided_filter.cv.image import to32F

from guided_filter.core.filters import FastGuidedFilter, GuidedFilter

import cv2

def run(src_path, label_path, sigmas, epsilon=0.02, same=False):
    src_name = os.path.basename(src_path)
    src_name = os.path.splitext(src_name)[0]
    label_name = os.path.basename(label_path)
    label_name = os.path.splitext(label_name)[0]

    src_C_8U = loadRGB(src_path)
    src_C_32F = to32F(src_C_8U)
    
    label_C_8U = loadRGB(label_path)
    label_C_32F = to32F(label_C_8U)

    os.makedirs('./output/', exist_ok=True)
    
    for sigma in sigmas:
        print(sigma)
        for eps in epsilon:
            if same:
                guided_filter = GuidedFilter(label_C_32F, radius=sigma, epsilon=eps)
            else:
                guided_filter = GuidedFilter(src_C_32F, radius=sigma, epsilon=eps)
            result = guided_filter.filter(label_C_32F)
            cv2.imwrite('./output/{}_radius{}_eps{:.03f}.png'.format(label_name, sigma, eps),
                cv2.cvtColor(result*255, cv2.COLOR_RGB2GRAY))        

def runSmoothNoiseResult(image_file):
    image_name = os.path.basename(image_file)
    image_name = os.path.splitext(image_name)[0]

    C_8U = loadRGB(image_file)
    C_32F = to32F(C_8U)

    aspect = C_32F.shape[0] / float(C_32F.shape[1])

    fig_width = 10
    fig_height = int(2 * fig_width * aspect / 3) + 2
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.82, wspace=0.02, hspace=0.3)
    h, w = C_32F.shape[:2]
    image_size_str = "Image size: %s x %s" %(w, h)
    fig.suptitle("Filtering noise image\n%s" % image_size_str)

    plt.subplot(231)
    plt.title("Original")
    plt.imshow(C_32F)
    plt.axis('off')

    h, w, cs = C_32F.shape

    C_noise = np.float32(C_32F + 0.3 * np.random.rand(h, w, cs))
    C_noise = np.clip(C_noise, 0.0, 1.0)

    plt.subplot(232)
    plt.title("Noise")
    plt.imshow(C_noise)
    plt.axis('off')

    sigmas = [5, 10, 20]

    plot_id = 234
    for sigma in sigmas:
        guided_filter = FastGuidedFilter(C_noise, radius=sigma, epsilon=0.02)
        C_smooth = guided_filter.filter(C_noise)
        C_smooth = np.clip(C_smooth, 0.0, 1.0)

        plt.subplot(plot_id)
        plt.title("Filtered ($r$=%s)" %sigma)
        plt.imshow(C_smooth)
        plt.axis('off')
        plot_id +=1

    result_file = resultFile(image_name)
    plt.savefig(result_file)


def runSmoothNoiseResults(data_names, data_ids):
    for data_name in data_names:
        print("Smooth noise: {}").format(data_name)
        for data_id in data_ids:
            print("Data ID: {}").format(data_id)
            image_file = dataFile(data_name, data_id)
            runSmoothNoiseResult(image_file)


if __name__ == '__main__':
    data_names = ["flower"]
    data_ids = range(3)

    runSmoothNoiseResults(data_names, data_ids)
