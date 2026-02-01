from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.quantization_utils import QuantAct
from models.vit_quant import deit_tiny_patch16_224, Attention, Block, PatchEmbed
from models.model_utils import un_int_model, unfreeze_model, freeze_model, int_model


def export_model(model):
    for child in model.children():
        if hasattr(child, 'export_mode'):
            child.export_mode = True
        export_model(child)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.qact = QuantAct()
        self.attention = Attention(
            dim=192,
            num_heads=3,
            qkv_bias=True
        )   
    
    def forward(self, x):
        x, scale = self.qact(x)
        x, scale = self.attention(x, scale)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = deit_tiny_patch16_224()
    model.set_float_op(True)
    model.train()
    un_int_model(model)
    unfreeze_model(model)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 1) 用随机数据跑一点训练 / calibration，让 QuantAct 统计到范围
    for step in range(10):
        x = torch.randn(5,3,224,224, device=device, requires_grad=True)
        y = torch.randn(5,1000, device=device)

        output = model(x)
        loss = F.mse_loss(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 2 == 0:
            print(f"step {step+1}: loss={loss.item():.4f}")

    model.eval()
    freeze_model(model)
    export_model(model)


    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    onnx_path = "QuantViT.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        opset_version=13,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"exported ONNX to {onnx_path}")


if __name__ == "__main__":
    main()
    print("Pipeline completed successfully.")