#!/usr/bin/env python3.6
import argparse
import os.path as ops
import time

from rknn.api import RKNN
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

INPUT_WIDTH = 304
INPUT_HEIGHT = 228

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('-r', '--rknn', type=str, help='The model rknn file')
    return parser.parse_args()


def init_rknn(rknn_path):
    rknn = RKNN()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('starting init runtime ...')
    # ret = rknn.init_runtime(target='rk1808', device_id='TS018080000000053')
    ret = rknn.init_runtime(target='rk1808')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


def run_test(rknn, image_path):
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
    # orig_img = cv2.imread(image_path)
    # img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = Image.open(image_path)
    img = img.resize([INPUT_WIDTH, INPUT_HEIGHT], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    # img = np.array(img).astype('uint8')
    img = np.expand_dims(np.asarray(img), axis=0)


    print('starting inference ...')
    infer_start = time.time()
    pred = rknn.inference(inputs=[img], data_type='float32')
    infer_end = time.time()
    print('done')
    print('inference time: ', infer_end-infer_start)
    print('inference result: ', pred)
    pred = np.array(pred).reshape(1, 128, 160, 1)
    print('inference result shape: ', pred.shape)

    fig = plt.figure()
    ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
    #ii = plt.imshow(pred[:, :], interpolation='nearest')
    fig.colorbar(ii)
    plt.savefig("output.jpg")
    plt.show()






if __name__ == '__main__':
    args = init_args()

    rknn = init_rknn(args.rknn)
    run_test(rknn, args.image_path)
    rknn.release()
