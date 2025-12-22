import onnx
from onnx import helper

in_path = "test.onnx"
out_path = "test_clean.onnx"

model = onnx.load(in_path)
graph = model.graph

# 1. 建索引
output_to_node = {}
for node in graph.node:
    for out_name in node.output:
        output_to_node[out_name] = node

init_names = {init.name for init in graph.initializer}

# 2. 先修补 Q/DQ 的 zero_point 输入
changed = 0

for node in graph.node:
    if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
        continue

    if len(node.input) < 3:
        continue

    zp_input = node.input[2]

    # 已经是 initializer 的就不用动
    if zp_input in init_names:
        continue

    src_node = output_to_node.get(zp_input, None)
    if src_node is None or src_node.op_type != "Identity":
        continue

    # 只处理单输入 Identity
    if len(src_node.input) != 1:
        continue

    id_in = src_node.input[0]
    # 只在 Identity 输入为 initializer 时修补
    if id_in not in init_names:
        continue

    # 改 Q/DQ 的 zp 输入指向 initializer
    print(f"[patch] {node.name or node.op_type}: zp {zp_input} -> {id_in}")
    node.input[2] = id_in
    changed += 1

print(f"patched {changed} Q/DQ zero_points")

# 3. 重新统计每个 tensor 被使用次数，用来判断 Identity 是否“无人使用”
use_count = {}
for node in graph.node:
    for inp in node.input:
        use_count[inp] = use_count.get(inp, 0) + 1
for out in [o.name for o in graph.output]:
    use_count[out] = use_count.get(out, 0) + 1

# 4. 删除所有“输出没人用，且不在 graph 输出里”的 Identity
new_nodes = []
removed_id = 0
graph_output_names = {o.name for o in graph.output}

for node in graph.node:
    if node.op_type == "Identity":
        # Identity 可以有多个输出，这里通常是 1 个
        can_remove = True
        for out_name in node.output:
            # 如果被任何节点或 graph 作为输入/输出使用，就不能删
            if use_count.get(out_name, 0) > 0:
                can_remove = False
                break
            if out_name in graph_output_names:
                can_remove = False
                break

        if can_remove:
            print(f"[remove] Identity: {node.name or node.output[0]}")
            removed_id += 1
            # 不加入 new_nodes，相当于从 graph 中删除
            continue

    new_nodes.append(node)

graph.ClearField("node")
graph.node.extend(new_nodes)

print(f"removed {removed_id} dangling Identity nodes")

onnx.save(model, out_path)
print(f"saved to {out_path}")