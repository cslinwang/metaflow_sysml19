import argparse
import tensorflow as tf
import numpy as np
import time

# CONST_VALUE 是一个常量，用于初始化参数。
# NCHW表示张量的数据格式
CONST_VALUE = .00000182
GLOBAL_DATA_FORMAT = "NCHW"

# PaddingType 枚举类型表示卷积操作的填充类型，包括 SAME 和 VALID


class PaddingType:
    SAME = 0
    VALID = 1

# ActiMode 枚举类型表示激活函数的类型，包括 AC_MODE_NONE、AC_MODE_SIGMOID、AC_MODE_RELU 和 AC_MODE_TANH。


class ActiMode:
    AC_MODE_NONE = 0
    AC_MODE_SIGMOID = 1
    AC_MODE_RELU = 2
    AC_MODE_TANH = 3

# OpType 枚举类型表示各种操作的类型


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

# 这个函数接收一个padding_type参数，并根据枚举类型PaddingType返回相应的填充类型字符串。如果填充类型未被识别，则会打印一个错误消息并终止程序


def get_padding_string(padding_type):
    if (padding_type == PaddingType.SAME):
        return "SAME"
    elif (padding_type == PaddingType.VALID):
        return "VALID"
    else:
        print("Unknown padding string")
        assert (0)

# 这个函数接收一个字符串和一个分隔符，并将字符串按照分隔符进行拆分。然后它将每个结果子字符串转换为整数，并返回整数列表


def split_string_ints(string, delim):
    ints = []
    splits = string.split(delim)
    for s in splits:
        ints.append(int(s))
    return ints


def split_string_int_pairs(string, delim1, delim2):
    '''
    这个函数接收一个字符串和两个分隔符。它先按照第一个分隔符拆分字符串，然后再将每个结果子字符串按照第二个分隔符拆分。
    然后它将两个结果子字符串转换为整数，并将它们作为元组附加到列表中。该函数返回一个整数对列表

    参数：
        string：输入字符串
        delim1：第一个分隔符
        delim2：第二个分隔符

    返回：
        一个整数对列表
    '''
    pairs = []
    splits = string.split(delim1)
    for s in splits:
        pair_split = s.split(delim2)
        pairs.append((int(pair_split[0]), int(pair_split[1])))
    return pairs

# 这个函数用于创建一个带偏置的卷积层。它接受七个参数，分别是输入张量 input_tensor，卷积核形状 filter_shape，步长 strides，填充类型 padding，偏置维度 bias_dim，
# 是否添加 ReLU 激活函数的标志 add_relu，以及层的名称 name。


def make_conv2d_with_bias(input_tensor, filter_shape, strides, padding, bias_dim, add_relu, name):
    weights_name = name + "_weights"
    bias_name = name + "_bias"
    conv_name = name + "_conv2d"
    bias_add_name = name + "_bias_add"

    # 创建常量权重和偏置张量
    weights = tf.constant(CONST_VALUE, shape=filter_shape, name=weights_name)
    bias = tf.constant(CONST_VALUE, shape=[bias_dim], name=bias_name)
    # 使用给定的参数调用 TensorFlow 的 conv2d 函数进行二维卷积操作，并将结果保存在 conv2d 变量中。之后，它使用 bias_add 函数将偏置添加到卷积输出中，保存在 bias_add 变量中
    conv2d = tf.nn.conv2d(input_tensor, weights, strides, get_padding_string(
        padding), data_format=GLOBAL_DATA_FORMAT, name=conv_name)
    bias_add = tf.nn.bias_add(
        conv2d, bias, data_format=GLOBAL_DATA_FORMAT, name=bias_add_name)

    # 如果 add_relu 标志被设置，则在 bias_add 上应用 ReLU 激活函数，并将结果返回；否则，直接返回 bias_add
    if (add_relu):
        relu_name = name + "_relu"
        relu = tf.nn.relu(bias_add, name=relu_name)
        return relu
    else:
        return bias_add


def create_input(line, operator_map):
    # 将输入的line字符串按逗号分割，并将其转换为一个整数列表，然后将其存储在变量dims中
    dims = split_string_ints(line, ',')
    # 创建一个名为input_shape的空列表，然后遍历dims列表中的每个元素。如果元素的值大于0，则将该元素添加到input_shape列表中
    input_shape = []
    for i in xrange(0, len(dims)):
        if (dims[i] > 0):
            input_shape.append(dims[i])
    # 使用 TensorFlow 库中的 tf.placeholder 函数创建一个输入占位符，该占位符将被用于后续的计算图中。这个占位符是一个float32类型的张量，它的形状由 input_shape 指定。
    input_placeholder = tf.placeholder(tf.float32, shape=input_shape)
    # 将刚刚创建的输入占位符存储到一个名为 operator_map 的字典中，字典的键为元组(0,0)
    operator_map[(0, 0)] = input_placeholder
    return input_shape


def parse_operator(line1, line2, line3, line4, operator_map, graph_outputs):
    '''
    这个函数根据给定的文本输入解析和创建TensorFlow操作。
    guid和op_type变量分别用于保存操作的唯一标识符和操作类型。deps变量是一个包含操作的输入张量的列表，它通过split_string_int_pairs函数从line3中解析得到。
    如果一个依赖张量在graph_outputs列表中，则将其从该列表中移除，以确保该依赖张量不会被误认为是整个图的输出张量

    参数：
        line1：操作的唯一标识符
        line2：操作的类型
        line3：操作的输入依赖关系
        line4：操作的参数
        operator_map：一个字典，用于保存已经创建的操作
        graph_outputs：一个列表，用于保存整个计算图的输出张量

    返回值：
        无

    '''
    guid = int(line1)
    op_type = int(line2)
    deps = split_string_int_pairs(line3, ',', ':')
    for dep in deps:
        if dep in graph_outputs:
            graph_outputs.remove(dep)
    # 使用if/elif语句对不同类型的操作进行分类，然后分别对每种类型的操作进行了解析和创建。每个操作类型都有不同的参数和输入张量，
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
    # 先测这个
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
    # 这个函数还支持矩阵乘法操作OP_MATMUL。对于矩阵乘法操作，该函数使用给定的权重张量和偏置张量，对输入张量进行矩阵乘法和偏置操作，并将结果张量添加到操作映射表中。
    elif (op_type == OpType.OP_MATMUL):
        params = split_string_ints(line4, ',')
        assert (len(params) == 5)
        # 在矩阵乘法操作中，首先从params参数列表中获取操作的参数。接下来，为矩阵乘法操作创建一些中间变量和名称。例如，使用reshape_name和matmul_name变量创建了重塑和矩阵乘法操作的名称，
        # 并使用weights_name和bias_name变量创建了权重和偏置变量的名称
        name = "matmul_" + str(guid)
        reshape_name = name + "_reshape"
        weights_name = name + "_weights"
        bias_name = name + "_bias"
        matmul_name = name + "_matmul"
        bias_add_name = name + "_bias_add"
        weights_shape = [params[2], params[3]]
        bias_dim = params[3]
        # 接下来，使用tf.constant函数创建权重和偏置张量。这些张量的值都是CONST_VALUE常量，该常量的值为0.1。这里使用tf.constant函数创建张量的原因是，这些张量的值在训练过程中不会改变，
        weights = tf.constant(
            CONST_VALUE, shape=weights_shape, name=weights_name)
        bias = tf.constant(CONST_VALUE, shape=[bias_dim], name=bias_name)
        # 接下来，使用tf.reshape函数将输入张量重塑为一个二维张量。这里使用tf.reshape函数的原因是，矩阵乘法操作的输入张量是一个四维张量，而矩阵乘法操作的权重张量是一个二维张量，
        reshape = tf.reshape(operator_map[deps[0]], [
                             params[0]*params[1], params[2]], name=reshape_name)
        # 接下来，使用tf.matmul函数对重塑后的输入张量和权重张量进行矩阵乘法操作。接下来，使用tf.nn.bias_add函数对矩阵乘法操作的结果张量进行偏置操作。
        # 最后，根据激活模式参数，根据激活模式参数，该函数应用sigmoid、ReLU、tanh等激活函数,并将结果添加到操作映射表中。
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
    # 如果解析到未知的操作类型，该函数将输出一条错误信息并抛出一个异常
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


def generate_node_by_hand(index):
    """
    手动构建节点

    参数：
        index：节点索引

    返回：
        line1：操作的唯一标识符，String类型
        line2：操作的类型
        line3：操作的输入依赖关系
        line4：操作的参数
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


# 函数入口，首先，该函数从命令行参数中获取图的文件名。接下来，该函数使用with语句打开图的文件，并使用while循环逐行读取图的文件。在读取图的文件时，该函数使用了一个名为operator_map的字典。
with open(args.graph_file, 'r') as graph_file:
    # The graph nodes are repesented by 4 lines
    # Maps (guid, output_index) to the tensor
    operator_map = {}
    # 是否需要输入
    need_input = True
    # line1 节点名称
    # line2 节点操作
    # line3 节点输入
    # line4 节点输出
    line1 = graph_file.readline()

    while line1:
        line2 = graph_file.readline()
        line3 = graph_file.readline()
        line4 = graph_file.readline()
        # Cut off the newlines 删除回车
        line1 = line1[0:-1]
        line2 = line2[0:-1]
        line3 = line3[0:-1]
        line4 = line4[0:-1]
        # 构建第一个节点
        lin1, line2, line3, line4 = generate_node_by_hand(0)

        if (need_input):
            need_input = False
            # 初始值：line4 = [], operator_map = {}
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


# 首先，代码创建一个tf.ConfigProto对象，该对象包含用于配置TensorFlow会话的选项。如果args.xla为真，则启用XLA编译器，这是TensorFlow的实验性功能，可用于加速模型执行
config = tf.ConfigProto()
if (args.xla):
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

# 代码生成一个随机的输入张量数据，形状为input_shape，该张量将用于对模型进行推理测试
input_data = np.random.random_sample(input_shape)
# 代码使用上面创建的配置对象创建一个TensorFlow会话对象。该会话将在with语句的作用域内运行，这样在退出该作用域时会话会自动关闭
with tf.Session(config=config) as sess:
    # 如果args.print_tensorboard为真，则将TensorBoard摘要写入指定目录。这将允许用户可视化TensorFlow图
    if (args.print_tensorboard):
        writer = tf.summary.FileWriter(args.print_tensorboard, sess.graph)
    # 代码在循环中运行模型args.discard_iter + args.iterations次。前args.discard_iter次迭代将被丢弃，因为这些迭代通常会花费更长的时间，并且可能不会提供有意义的时间测量。
    # 对于每次迭代，代码将运行模型，使用output_nodes作为要获取的张量列表，并将输入数据传递给模型。然后，代码将计算这次迭代的执行时间，并将其添加到times列表中
    times = []
    for i in range(args.discard_iter + args.iterations):
        t0 = time.time()
        sess.run(output_nodes, {operator_map[(0, 0)]: input_data})
        t1 = time.time()
        # print(str(t1 - t0) + " seconds")
        times.append(t1 - t0)

    # 在所有迭代完成后，代码将计算除前args.discard_iter个迭代之外的所有迭代的平均执行时间，并将其打印到控制台中
    total = 0
    for i in range(args.discard_iter, len(times)):
        total += times[i]
    avg = total / (args.iterations)
    print("Average time of the last " + str(args.iterations) +
          " iterations: " + str(avg) + " seconds")
