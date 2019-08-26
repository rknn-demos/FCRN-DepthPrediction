#!/usr/bin/env python3.6
import argparse
import os
import os.path as ops

from rknn.api import RKNN
from PIL import Image

import numpy as np
import cv2
import matplotlib.pyplot as plt


INPUT_WIDTH = 304
INPUT_HEIGHT = 228

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=True, help='The image path or the src image save dir')
    parser.add_argument('-r', '--rknn', type=str, required=True, help='The model rknn file')
    return parser.parse_args()


def init_rknn(rknn_path):
    rknn = RKNN(verbose=True)
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('starting init runtime ...')
    rknn.init_runtime(target='rk1808', perf_debug=True)
    return rknn


def perf(rknn, image_path):
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
    img = Image.open(image_path)
    img = img.resize([INPUT_WIDTH, INPUT_HEIGHT], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0) 
    # orig_img = cv2.imread(image_path)
    # img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
    print('starting eval perf ...')
    rknn.eval_perf(inputs=[img], is_print=True)
    print('done')


if __name__ == '__main__':
    args = init_args()
    rknn = init_rknn(args.rknn)
    perf(rknn, args.image_path)
    rknn.release()
