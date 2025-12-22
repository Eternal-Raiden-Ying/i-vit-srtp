import torch
import torch.nn as nn
import torch.optim as optim

from models.simple_quant_model import SimpleModel
from models.vit_quant import deit_tiny_patch16_224
from models.model_utils import un_int_model, unfreeze_model, freeze_model, int_model


def export_model(model):
    for child in model.children():
        if hasattr(child, 'export_mode'):
            child.export_mode = True
        export_model(child)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = SimpleModel().to(device)
    model = deit_tiny_patch16_224(pretrained=False,
                                  num_classes=50,
                                  drop_rate=0.0,
                                  drop_path_rate=0.1).to(device)
    model.train()

    unfreeze_model(model)
    un_int_model(model)

    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 1) 用随机数据跑一点训练 / calibration，让 QuantAct 统计到范围
    for step in range(10):
        x = torch.randn(5, 3, 224, 224, device=device, requires_grad=True)
        y = torch.randint(0,50,(5,), device=device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 2 == 0:
            print(f"step {step+1}: loss={loss.item():.4f}")

    model.eval()
    freeze_model(model)
    export_model(model)

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    onnx_path = "test.onnx"

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