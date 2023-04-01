import argparse
import tensorflow as tf
import numpy as np
import time

"""
使用现编图的方式，构建tensorflow的计算图
"""

CONST_VALUE = .00000182
GLOBAL_DATA_FORMAT = "NCHW"


class PaddingType:
    SAME = 0
    VALID = 1


class ActiMode:
    AC_MODE_NONE = 0
    AC_MODE_SIGMOID = 1
    AC_MODE_RELU = 2
    AC_MODE_TANH = 3

# 定义一个名为 OpType 的类，用于表示操作的类型


class OpType:
    OP_NOOP = 0  # 定义常量 OP_NOOP 的值为 0，表示不进行任何操作
    OP_ANY = 1   # 定义常量 OP_ANY 的值为 1，表示可以代表任何类型的操作
    OP_CONV2D = 2  # 定义常量 OP_CONV2D 的值为 2，表示进行计算机视觉中常用的二维卷积操作
    OP_LINEAR = 3  # 定义常量 OP_LINEAR 的值为 3，表示进行神经网络中常用的线性变换操作
    OP_POOL2D_MAX = 4  # 定义常量 OP_POOL2D_MAX 的值为 4，表示进行计算机视觉中常用的二维最大池化操作
    OP_POOL2D_AVG = 5  # 定义常量 OP_POOL2D_AVG 的值为 5，表示进行计算机视觉中常用的二维平均池化操作
    OP_RELU = 6  # 定义常量 OP_RELU 的值为 6，表示进行神经网络中常用的修正线性单元激活函数操作
    OP_SIGMOID = 7  # 定义常量 OP_SIGMOID 的值为 7，表示进行神经网络中常用的 sigmoid 激活函数操作
    OP_BATCHNORM = 8  # 定义常量 OP_BATCHNORM 的值为 8，表示进行神经网络中常用的批归一化操作
    OP_CONCAT = 9  # 定义常量 OP_CONCAT 的值为 9，表示进行多个张量的连接操作
    OP_SPLIT = 10  # 定义常量 OP_SPLIT 的值为 10，表示进行将一个张量拆分成多个较小张量的操作
    # 下面是 RNN 算法中常用的操作
    OP_EW_ADD = 11  # 定义常量 OP_EW_ADD 的值为 11，表示进行逐元素加法操作
    OP_EW_MUL = 12  # 定义常量 OP_EW_MUL 的值为 12，表示进行逐元素乘法操作
    OP_MATMUL = 13  # 定义常量 OP_MATMUL 的值为 13，表示进行矩阵乘法操作


def get_padding_string(padding_type):
    if (padding_type == PaddingType.SAME):
        return "SAME"
    elif (padding_type == PaddingType.VALID):
        return "VALID"
    else:
        print("Unknown padding string")
        assert (0)


def split_string_ints(string, delim):
    ints = []
    splits = string.split(delim)
    for s in splits:
        ints.append(int(s))
    return ints


def split_string_int_pairs(string, delim1, delim2):
    pairs = []
    splits = string.split(delim1)
    for s in splits:
        pair_split = s.split(delim2)
        pairs.append((int(pair_split[0]), int(pair_split[1])))
    return pairs


def make_conv2d_with_bias(input_tensor, filter_shape, strides, padding, bias_dim, add_relu, name):
    weights_name = name + "_weights"
    bias_name = name + "_bias"
    conv_name = name + "_conv2d"
    bias_add_name = name + "_bias_add"

    weights = tf.constant(CONST_VALUE, shape=filter_shape, name=weights_name)
    bias = tf.constant(CONST_VALUE, shape=[bias_dim], name=bias_name)
    conv2d = tf.nn.conv2d(input_tensor, weights, strides, get_padding_string(
        padding), data_format=GLOBAL_DATA_FORMAT, name=conv_name)
    bias_add = tf.nn.bias_add(
        conv2d, bias, data_format=GLOBAL_DATA_FORMAT, name=bias_add_name)

    if (add_relu):
        relu_name = name + "_relu"
        relu = tf.nn.relu(bias_add, name=relu_name)
        return relu
    else:
        return bias_add


def create_input(line, operator_map):
    dims = split_string_ints(line, ',')
    input_shape = []
    for i in xrange(0, len(dims)):
        if (dims[i] > 0):
            input_shape.append(dims[i])
    input_placeholder = tf.placeholder(tf.float32, shape=input_shape)
    operator_map[(0, 0)] = input_placeholder
    return input_shape


def parse_operator(line1, line2, line3, line4, operator_map, graph_outputs):
    guid = int(line1)
    op_type = int(line2)
    deps = split_string_int_pairs(line3, ',', ':')
    for dep in deps:
        if dep in graph_outputs:
            graph_outputs.remove(dep)
    if (op_type == OpType.OP_CONV2D):
        params = split_string_ints(line4, ',')
        filter_shape = [params[5], params[6], params[1], params[4]]
        strides = [1, 1, params[7], params[8]]
        if (params[9] > 0 or params[10] > 0):
            padding = PaddingType.SAME
        else:
            padding = PaddingType.VALID
        name = "conv2d_" + str(guid)
        conv = make_conv2d_with_bias(
            operator_map[deps[0]], filter_shape, strides, padding, params[4], params[11], name=name)
        operator_map[(guid, 0)] = conv
        return [(guid, 0)]
    elif (op_type == OpType.OP_POOL2D_MAX or op_type == OpType.OP_POOL2D_AVG):
        params = split_string_ints(line4, ',')
        ksize = [1, 1, params[5], params[6]]
        strides = [1, 1, params[7], params[8]]
        if (params[9] > 0 or params[10] > 0):
            padding = PaddingType.SAME
        else:
            padding = PaddingType.VALID
        created_op = None
        name = None
        if (op_type == OpType.OP_POOL2D_MAX):
            name = "maxpool_" + str(guid)
            max_pool = tf.nn.max_pool(operator_map[deps[0]], ksize, strides, get_padding_string(
                padding), data_format=GLOBAL_DATA_FORMAT, name=name)
            created_op = max_pool
        else:
            name = "avgpool_" + str(guid)
            avg_pool = tf.nn.avg_pool(operator_map[deps[0]], ksize, strides, get_padding_string(
                padding), data_format=GLOBAL_DATA_FORMAT, name=name)
            created_op = avg_pool
        if params[10]:  # If add relu
            relu_name = name + "_relu"
            relu = tf.nn.relu(created_op, name=relu_name)
            operator_map[(guid, 0)] = relu
            return [(guid, 0)]
        else:
            operator_map[(guid, 0)] = created_op
            return [(guid, 0)]
    elif (op_type == OpType.OP_SPLIT):
        params = split_string_ints(line4, ',')
        name = "split_" + str(guid)
        splits = tf.split(operator_map[deps[0]], params, 1, name=name)
        rets = []
        for i in xrange(0, len(splits)):
            operator_map[(guid, i)] = splits[i]
            rets.append((guid, i))
        return rets
    elif (op_type == OpType.OP_CONCAT):
        name = "concat_" + str(guid)
        dep_tensors = []
        for i in xrange(0, len(deps)):
            dep_tensors.append(operator_map[deps[i]])
        concat = tf.concat(dep_tensors, 1, name=name)
        operator_map[(guid, 0)] = concat
        return [(guid, 0)]
    elif (op_type == OpType.OP_EW_ADD):
        name = "ew_add_" + str(guid)
        ew_add = tf.add(operator_map[deps[0]],
                        operator_map[deps[1]], name=name)
        operator_map[(guid, 0)] = ew_add
        return [(guid, 0)]
    elif (op_type == OpType.OP_EW_MUL):
        name = "ew_mul_" + str(guid)
        ew_mul = tf.multiply(
            operator_map[deps[0]], operator_map[deps[1]], name=name)
        operator_map[(guid, 0)] = ew_mul
        return [(guid, 0)]
    elif (op_type == OpType.OP_RELU):
        name = "relu_" + str(guid)
        relu = tf.nn.relu(operator_map[deps[0]], name=name)
        operator_map[(guid, 0)] = relu
        return [(guid, 0)]
    elif (op_type == OpType.OP_SIGMOID):
        name = "sigmoid_" + str(guid)
        sigmoid = tf.nn.sigmoid(operator_map[deps[0]], name=name)
        operator_map[(guid, 0)] = sigmoid
        return [(guid, 0)]
    elif (op_type == OpType.OP_BATCHNORM or op_type == OpType.OP_NOOP):
        operator_map[(guid, 0)] = operator_map[deps[0]]
        return [(guid, 0)]
    elif (op_type == OpType.OP_MATMUL):
        params = split_string_ints(line4, ',')
        assert (len(params) == 5)
        name = "matmul_" + str(guid)
        reshape_name = name + "_reshape"
        weights_name = name + "_weights"
        bias_name = name + "_bias"
        matmul_name = name + "_matmul"
        bias_add_name = name + "_bias_add"
        weights_shape = [params[2], params[3]]
        bias_dim = params[3]
        weights = tf.constant(
            CONST_VALUE, shape=weights_shape, name=weights_name)
        bias = tf.constant(CONST_VALUE, shape=[bias_dim], name=bias_name)
        reshape = tf.reshape(operator_map[deps[0]], [
                             params[0]*params[1], params[2]], name=reshape_name)
        matmul = tf.matmul(reshape, weights, name=matmul_name)
        bias_add = tf.nn.bias_add(matmul, bias, name=bias_add_name)
        actimode = params[4]
        if (actimode == ActiMode.AC_MODE_NONE):
            operator_map[(guid, 0)] = bias_add
            return [(guid, 0)]
        elif (actimode == ActiMode.AC_MODE_SIGMOID):
            name += "_sigmoid"
            sigmoid = tf.nn.sigmoid(bias_add, name=name)
            operator_map[(guid, 0)] = sigmoid
            return [(guid, 0)]
        elif (actimode == ActiMode.AC_MODE_RELU):
            name += "_relu"
            relu = tf.nn.relu(bias_add, name=name)
            operator_map[(guid, 0)] = relu
            return [(guid, 0)]
        elif (actimode == ActiMode.AC_MODE_TANH):
            name += "_tanh"
            tanh = tf.nn.tanh(bias_add, name=name)
            operator_map[(guid, 0)] = tanh
            return [(guid, 0)]
        else:
            print("unknown actimode!!!!")
            assert (0)
    else:
        print("Found unknown opcode")
        assert (0)


# /**
# --xla：布尔值参数，用于指示是否使用 TensorFlow XLA 优化。
# --graph_file：字符串参数，指定从中加载图的文件。
# --print_tensorboard：字符串参数，指定输出 TensorBoard 信息的文件夹。
# --iterations：整数参数，指定计时时要平均多少次迭代（默认为 5000）。
# --discard_iter：整数参数，指定在预热期间要舍弃多少次迭代的时间信息（默认为 1000）。
# **/
parser = argparse.ArgumentParser()
parser.add_argument(
    "--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument(
    "--graph_file", help="The file from which to load the graph")
parser.add_argument("--print_tensorboard",
                    help="Name of folder to output the tensorboard information")
parser.add_argument(
    "--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=5000)
parser.add_argument(
    "--discard_iter", help="How many iterations to discard timing information during warm up (default 1000)", type=int, default=1000)
args = parser.parse_args()

input_shape = []
graph_outputs = set()
# 自定义简单模型

# model_path = "/usr/metaflow/metaflow_sysml19/tensorflow_py/graph/graph.pbtxt"
# graph_path = "/usr/metaflow/metaflow_sysml19/tensorflow_py/graph/graph.pb"
operators_path = "/usr/metaflow/metaflow_sysml19/tensorflow_py/graph/graph.txt"


def generate_node_by_hand(index):
    """
    手动构建节点

    参数：
        index：节点索引

    返回：
        line1：节点索引，String类型
        line2：节点依赖
        line3：节点参数
        line4：节点参数
    """
    if (index == 0):
        line1 = "0"
        line2 = "0"
        line3 = "0"
        line4 = "1,1,1,1"
    else:
        line1 = "0"
        line2 = "0"
        line3 = "0"
        line4 = "1,1,1,1"
    return line1, line2, line3, line4


with open(operators_path, 'r') as graph_file:
    # The graph nodes are repesented by 4 lines
    operator_map = {}
    need_input = False
    line1 = graph_file.readline()

    while line1:
        line2 = graph_file.readline()
        line3 = graph_file.readline()
        line4 = graph_file.readline()
        # Cut off the newlines
        line1 = line1[0:-1]
        line2 = line2[0:-1]
        line3 = line3[0:-1]
        line4 = line4[0:-1]
        # 构建第一个节点
        lin1, line2, line3, line4 = generate_node_by_hand(0)

        if (need_input):
            need_input = False
            input_shape = create_input(line4, operator_map)
            graph_outputs.add((0, 0))

        recent_outputs = parse_operator(
            line1, line2, line3, line4, operator_map, graph_outputs)
        for output in recent_outputs:
            graph_outputs.add(output)

        # Using this as the test of if the file is empty
        line1 = graph_file.readline()

if (len(graph_outputs) == 0):
    print("Could not read the graph!!!")
    assert (0)

graph_outputs = set()
for output in recent_outputs:
    graph_outputs.add(output)

output_nodes = []
for graph_output in graph_outputs:
    output_nodes.append(operator_map[graph_output])


config = tf.ConfigProto()
if (args.xla):
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

input_data = np.random.random_sample(input_shape)
with tf.Session(config=config) as sess:
    if (args.print_tensorboard):
        writer = tf.summary.FileWriter(args.print_tensorboard, sess.graph)
    times = []
    for i in range(args.discard_iter + args.iterations):
        t0 = time.time()
        sess.run(output_nodes, {operator_map[(0, 0)]: input_data})
        t1 = time.time()
        # print(str(t1 - t0) + " seconds")
        times.append(t1 - t0)

    total = 0
    for i in range(args.discard_iter, len(times)):
        total += times[i]
    avg = total / (args.iterations)
    print("Average time of the last " + str(args.iterations) +
          " iterations: " + str(avg) + " seconds")
