import tensorflow as tf

# 定义卷积层
def conv2d(inputs, filters, kernel_size, strides, padding, activation=None, name=None):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            bias_initializer=tf.zeros_initializer(),
            name=name
        )
        return conv

# 定义模型
def complex_conv_model(inputs):
    with tf.variable_scope("complex_conv_model"):
        conv1 = conv2d(inputs, filters=32, kernel_size=[3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu, name="conv1")
        conv2 = conv2d(conv1, filters=64, kernel_size=[3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu, name="conv2")
        conv3 = conv2d(conv2, filters=128, kernel_size=[3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu, name="conv3")
        conv4 = conv2d(conv3, filters=256, kernel_size=[3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu, name="conv4")
        conv5 = conv2d(conv4, filters=512, kernel_size=[3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu, name="conv5")
        flatten = tf.contrib.layers.flatten(conv5)
        fc1 = tf.layers.dense(flatten, 1024, activation=tf.nn.relu, name="fc1")
        fc2 = tf.layers.dense(fc1, 512, activation=tf.nn.relu, name="fc2")
        fc3 = tf.layers.dense(fc2, 10, name="fc3")
        return fc3

# 创建输入占位符
inputs = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="inputs")

# 构建计算图
logits = complex_conv_model(inputs)

# 创建会话并保存计算图
with tf.Session() as sess:
    # 导出计算图
    graph_def = sess.graph_def
    # 保存计算图为四行类型文本文件
    graph_path = "/usr/metaflow/metaflow_sysml19/tensorflow_py/graph/"
    with open(graph_path+"graph.txt", "w") as f:
        for node in graph_def.node:
            node_name = node.name
            node_type = node.op
            input_names = ",".join(node.input)
            output_names = node_name + ":0"
            f.write("{}\n{}\n{}\n{}\n".format(node_name, node_type, input_names, output_names))
