import torch
import torch.nn as nn
import torch.optim as optim

from simple_test.simple_mlp_quant import SimpleMLP, collect_qparams, set_export_mode, save_qparams


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleMLP(input_dim=16, hidden_dim=32, output_dim=10).to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 1) 用随机数据跑一点训练 / calibration，让 QuantAct 统计到范围
    for step in range(200):
        x = torch.randn(32, 16, device=device)
        y = torch.randint(0, 10, (32,), device=device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"step {step+1}: loss={loss.item():.4f}")

    # 2) 收集量化参数并保存
    model.eval()
    qparams = collect_qparams(model)
    save_qparams(qparams, "simple_mlp_qparams.json")
    print("saved qparams to simple_mlp_qparams.json")
    print(qparams)

    # 3) 导出 ONNX：注意要先让 QuantAct 变为 identity
    set_export_mode(model, True)

    dummy_input = torch.randn(1, 16, device=device)
    onnx_path = "simple_mlp_fp.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"exported ONNX to {onnx_path}")


if __name__ == "__main__":
    main()
    print("Pipeline completed successfully.")