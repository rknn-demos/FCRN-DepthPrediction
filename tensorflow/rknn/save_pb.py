#!/usr/bin/env python3.6
import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph(input_checkpoint,output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "ConvPred/ConvPred"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
 
        # for op in sess.graph.get_operations():
        #     print(op.name, op.values())


if __name__ == '__main__':
    # 输入ckpt模型路径
    input_checkpoint='./NYU_FCRN-checkpoint/NYU_FCRN.ckpt'
    # 输出pb模型的路径
    out_pb_path="./NYU_FCRN-checkpoint/freeze_NYU_FCRN.pb"
    # 调用freeze_graph将ckpt转为pb
    freeze_graph(input_checkpoint,out_pb_path)
