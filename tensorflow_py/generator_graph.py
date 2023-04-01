import argparse
import tensorflow as tf
import numpy as np
import time
from shared_functions import make_matmul
import os

import argparse
import tensorflow as tf
import numpy as np

"""
生成计算图并保存
"""

# 定义了一个 Attention 函数，用于生成计算图中的节点
def attention(input, heads):
    # 获取输入张量的维度
    d_model = input.shape[1].value
    # 分别对输入张量乘以三个权重矩阵，得到查询矩阵 q、键矩阵 k 和值矩阵 v
    q = make_matmul(input, d_model)
    k = make_matmul(input, d_model)
    v = make_matmul(input, d_model)
    # reshape query, key, value，将查询、键、值矩阵进行变形
    q = tf.reshape(q, shape=(64,16,64))
    k = tf.reshape(k, shape=(64,16,64))
    v = tf.reshape(v, shape=(64,16,64))
    # transpose q, k, v for batched matmul，将查询、键、值矩阵进行转置，以便进行批量矩阵乘法
    q = tf.transpose(q, perm=(1,0,2))
    k = tf.transpose(k, perm=(1,0,2))
    v = tf.transpose(v, perm=(1,0,2))
    # 通过两次矩阵乘法和 softmax 函数计算出注意力权重，再次与值矩阵相乘得到最终的输出矩阵
    logits = tf.matmul(q, k)
    output = tf.matmul(logits, v)
    # transpose the output back，将输出矩阵进行转置，以便进行下一步的变形操作
    output = tf.transpose(output, perm=(1,0,2))
    output = tf.reshape(output, shape=(64, 1024))
    # a final linear layer，通过一个线性变换得到最终的输出张量
    output = make_matmul(tf.nn.relu(make_matmul(input, 4*d_model)), d_model)
    return output

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=1000)
parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
args = parser.parse_args()

# 定义输入张量
input = tf.placeholder(tf.float32, shape=(64,1024))
input_dictionary = {}
# 生成随机输入数据
input_dictionary[input] = np.random.random_sample((64, 1024))
t = input

# 通过多次调用 Attention 函数构建计算图
for i in range(1):
    t = attention(t, 16)
# 定义输出张量并保存计算图

output_nodes = [t]
print("save graph start")
graph_path = "/usr/metaflow/metaflow_sysml19/tensorflow_py/graph/"
graph_def = tf.get_default_graph().as_graph_def()
# 将计算图保存为文件
tf.train.write_graph(graph_def, graph_path, 'graph.pb', as_text=False)
print("save graph end")

# 保存节点信息
with open(graph_path+'graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

operator_map = {}
graph_outputs = set()
with open(graph_path+'operators.txt', 'w') as f:
    for node in graph_def.node:
        # 第一行：节点名称和类型
        f.write('{}\t{}\n'.format(node.name, node.op))

        # 第二行：输入张量名称
        input_names = [input_tensor for input_tensor in node.input]
        f.write('{}\n'.format(','.join(input_names)))

        # 第三行：输出张量名称
        if len(node.output) == 0:
            f.write(' \n')
        else:
            output_names = [output_tensor for output_tensor in node.output]
            f.write('{}\n'.format(','.join(output_names)))

        # 第四行：节点属性
        for attr in node.attr:
            if attr.HasField('list'):
                f.write('{}:{}:{}\n'.format(attr.name, ','.join([str(val) for val in attr.list.i]), attr.ListFields()[0][0].type_name))
            elif attr.HasField('f'):
                f.write('{}:{}:float\n'.format(attr.name, attr.f))
            elif attr.HasField('i'):
                f.write('{}:{}:int\n'.format(attr.name, attr.i))
            elif attr.HasField('b'):
                f.write('{}:{}:bool\n'.format(attr.name, attr.b))
            elif attr.HasField('s'):
                f.write('{}:{}:str\n'.format(attr.name, attr.s.decode('utf-8')))