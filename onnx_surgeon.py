import onnx
import onnx_graphsurgeon as gs
import numpy as np

# 加载模型
graph = gs.import_onnx(onnx.load("Attention_clean.onnx"))

# 目标：找到特定位置的 QDQ 节点
# 比如你想修改名为 "linear_matmul" 之前的那个权重量化节点
target_node_name = "/attention/qact3/QuantizeLinear"

for node in graph.nodes:
    if node.name == target_node_name and node.op == "QuantizeLinear":
        dq_node = node.outputs[0].outputs[0]

        # 1. 修改 Q 节点为 Cast
        node.op = "Cast"
        node.attrs["to"] = 6  # 6 代表 INT32
        # Cast 只需要一个输入，删掉 scale 和 zp
        node.inputs = [node.inputs[0]]
        node.outputs[0].dtype = np.int32
        
        if dq_node.op == "DequantizeLinear":
            # C. 找到 DQ 的输出张量
            dq_output_tensor = dq_node.outputs[0]

            # D. 处理普通节点的连接：把所有消耗 DQ 输出的节点，改连到 Q 的输入上
            for consumer in dq_output_tensor.outputs:
                for i, inp in enumerate(consumer.inputs):
                    if inp == dq_output_tensor:
                        consumer.inputs[i] = node.outputs[0]

            # E. 【关键点】处理图的全局输出：
            # 如果 DQ 的输出本身就是整个模型的输出，必须把全局输出指向 Q 的输入
            for i, out_var in enumerate(graph.outputs):
                if out_var == dq_output_tensor:
                    graph.outputs[i] = node.outputs[0]
            
            # F. 清空被删除节点的引用，让 cleanup 能回收它们
            # 不要手动修改 graph.nodes 列表，让 cleanup 自动处理更安全
            dq_node.inputs.clear()
            dq_node.outputs.clear()
            print(f"Modified node {node.name} from QDQ to Cast.")


# # 清理并保存
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "modified_bitwidth.onnx")