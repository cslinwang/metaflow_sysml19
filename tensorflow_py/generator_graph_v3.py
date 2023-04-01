import tensorflow as tf
from tensorflow.python.platform import gfile

def convert_graph_pb_to_graph_file(graph_pb_path, graph_file_path):
    with tf.Session() as sess:
        with gfile.GFile(graph_pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        with open(graph_file_path, 'w') as graph_file:
            for node in graph_def.node:
                # 在这里，我们需要为 graph_file 格式生成相应的四行信息
                # 第一行：节点的 GUID
                guid = node.name.replace("/", "_")
                graph_file.write(guid + "\n")

                # 第二行：操作类型（OpType）
                op_type = node.op
                graph_file.write(op_type + "\n")

                # 第三行：依赖关系（输入节点）
                deps = ",".join([input_node.replace("/", "_") for input_node in node.input])
                graph_file.write(deps + "\n")

                # 第四行：操作参数
                params = ""
                for attr in node.attr:
                    params += str(node.attr[attr]) + ","
                graph_file.write(params[:-1] + "\n")

if __name__ == "__main__":
    graph_pb_path = "/usr/metaflow/metaflow_sysml19/tensorflow_py/graph/graph.pb"
    graph_file_path = "/usr/metaflow/metaflow_sysml19/tensorflow_py/graph/graph.txt"
    convert_graph_pb_to_graph_file(graph_pb_path, graph_file_path)
    print("转换完成！")