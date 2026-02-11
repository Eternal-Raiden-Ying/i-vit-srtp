from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.vit_quant import deit_tiny_patch16_224
from models.model_utils import un_int_model, unfreeze_model, freeze_model, int_model, export_model


checkpoint_pth = "./results/training/best_checkpoint.pth.tar"
onnx_path = "QuantViT.onnx"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = deit_tiny_patch16_224()
    ckpt = torch.load(checkpoint_pth, map_location='cpu', weights_only=False)

    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    freeze_model(model)
    export_model(model)

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

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



if __name__ == "__main__":
    main()
    print(f"Successfully exported ONNX model to {onnx_path}.")