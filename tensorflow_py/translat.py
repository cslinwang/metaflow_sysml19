import tensorflow as tf

# 加载计算图文件
graph_path = "/usr/metaflow/metaflow_sysml19/tensorflow_py/graph/"
with tf.gfile.GFile(graph_path + "graph.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# 创建一个 Session 对象
with tf.Session() as sess:
    # 将计算图导入当前默认的计算图中
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

        # 将变量节点转换为常量节点
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            ["output_node_name"]  # 指定要输出的节点名称，可以是多个节点
        )

        # 删除与训练相关的节点
        output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def)

        # 提取指定输出节点的子图
        output_graph_def = tf.graph_util.extract_sub_graph(
            output_graph_def,
            ["output_node_name"]  # 指定要输出的节点名称，可以是多个节点
        )

        # 将计算图保存为四行类型文本文件
        tf.train.write_graph(output_graph_def, graph_path, 'graph.txt', as_text=True)