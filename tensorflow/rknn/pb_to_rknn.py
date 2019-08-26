#!/usr/bin/env python3.6

from rknn.api import RKNN
import argparse


INPUT_WIDTH = 304
INPUT_HEIGHT = 228

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='The model pb file')
    parser.add_argument('-o', '--output', type=str, required=True, help='The output rknn file')
    return parser.parse_args()


def to_rknn(pb_path, rknn_path):
    rknn = RKNN(verbose=True)
    rknn.config(channel_mean_value='0 0 0 1', reorder_channel='0 1 2')
    # rknn.config(channel_mean_value='128 128 128 128', reorder_channel='0 1 2')
    print('--> Loading model')
    rknn.load_tensorflow(tf_pb=pb_path,
                         inputs=['Placeholder'],
                         outputs=['ConvPred/ConvPred'],
                         input_size_list=[[INPUT_HEIGHT, INPUT_WIDTH, 3]])
    print('done')
    print('--> Building model')
    rknn.build(do_quantization=False, pre_compile=True)
    print('done')
    rknn.export_rknn(rknn_path)


if __name__ == '__main__':
    args = init_args()
    to_rknn(args.input, args.output)
