import json
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantAct(nn.Module):
    def __init__(self, name: str, qmin: int = -128, qmax: int = 127):
        super().__init__()
        self.name = name
        self.qmin = qmin
        self.qmax = qmax

        self.register_buffer("min_val", torch.tensor(float("+inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

        self.export_mode = False  # 可选

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor):
        cur_min = x.min()
        cur_max = x.max()
        self.min_val.copy_(torch.minimum(self.min_val, cur_min))
        self.max_val.copy_(torch.maximum(self.max_val, cur_max))

    def calc_qparams(self):
        """
        统一采用对称量化：zero_point = 0, qmin=-128, qmax=127
        """
        min_val = float(self.min_val)
        max_val = float(self.max_val)
        if max_val <= min_val:
            max_val = min_val + 1e-6

        # 对称范围：取绝对最大值
        abs_max = max(abs(min_val), abs(max_val))
        if abs_max == 0.0:
            abs_max = 1e-6

        qmin, qmax = self.qmin, self.qmax  # 默认为 -128, 127
        scale = abs_max / max(abs(qmin), abs(qmax))
        zp = 0
        return scale, zp

    def _fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用对称量化（zero_point = 0）的 fake-quant，
        以便 TensorRT 能解析 Q/DQ（不再有非零 zero-point）。
        """
        min_val = float(self.min_val)
        max_val = float(self.max_val)
        if max_val <= min_val:
            return x

        # 计算对称 scale
        abs_max = max(abs(min_val), abs(max_val))
        if abs_max == 0.0:
            return x

        qmin, qmax = self.qmin, self.qmax
        scale = abs_max / max(abs(qmin), abs(qmax))
        if scale == 0.0:
            return x

        zp = 0  # 对称量化

        return torch.fake_quantize_per_tensor_affine(
            x, float(scale), int(zp), qmin, qmax
        )

    def forward(self, x: torch.Tensor):
        if self.training:
            self._update_stats(x.detach())
        x = self._fake_quant(x)
        return x


class QuantLinear(nn.Module):
    """
    带权重（以及可选 bias）fake-quant 的 Linear。
    使用对称量化（zero_point = 0）。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 w_qmin: int = -128, w_qmax: int = 127, b_qmin: int = -128, b_qmax: int = 127):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.w_qmin, self.w_qmax = w_qmin, w_qmax
        self.b_qmin, self.b_qmax = b_qmin, b_qmax

        self.register_buffer("w_min", torch.tensor(float("+inf")))
        self.register_buffer("w_max", torch.tensor(float("-inf")))
        if bias:
            self.register_buffer("b_min", torch.tensor(float("+inf")))
            self.register_buffer("b_max", torch.tensor(float("-inf")))
        else:
            self.b_min = None
            self.b_max = None

    @torch.no_grad()
    def update_weight_stats(self):
        w = self.linear.weight
        self.w_min.copy_(torch.minimum(self.w_min, w.min()))
        self.w_max.copy_(torch.maximum(self.w_max, w.max()))
        if self.linear.bias is not None:
            b = self.linear.bias
            self.b_min.copy_(torch.minimum(self.b_min, b.min()))
            self.b_max.copy_(torch.maximum(self.b_max, b.max()))

    def _fake_quant_param(self, t: torch.Tensor, qmin: int, qmax: int,
                          cur_min: float, cur_max: float) -> torch.Tensor:
        """
        使用对称量化（zero_point = 0）对权重/偏置做 fake-quant。
        """
        if cur_max <= cur_min:
            return t

        abs_max = max(abs(cur_min), abs(cur_max))
        if abs_max == 0.0:
            return t

        scale = abs_max / max(abs(qmin), abs(qmax))
        if scale == 0.0:
            return t

        zp = 0  # 对称量化

        return torch.fake_quantize_per_tensor_affine(
            t, float(scale), int(zp), qmin, qmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update_weight_stats()

        w = self.linear.weight
        w_fq = self._fake_quant_param(
            w, self.w_qmin, self.w_qmax,
            float(self.w_min), float(self.w_max)
        )

        if self.linear.bias is not None:
            b = self.linear.bias
            b_fq = self._fake_quant_param(
                b, self.b_qmin, self.b_qmax,
                float(self.b_min), float(self.b_max)
            )
        else:
            b_fq = None

        return F.linear(x, w_fq, b_fq)


class SimpleMLP(nn.Module):
    """
    两层 Linear: input_dim -> hidden_dim -> output_dim
    在输入端、fc1 输出和 fc2 输出都插入 QuantAct。
    """
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=10):
        super().__init__()
        self.input_q = QuantAct("input")
        self.fc1 = QuantLinear(input_dim, hidden_dim, bias=False)
        self.act_q = QuantAct("fc1_out")
        self.relu = nn.ReLU()
        self.fc2 = QuantLinear(hidden_dim, hidden_dim, bias=False)
        self.act_q2 = QuantAct("fc2_out")
        self.fc3 = QuantLinear(hidden_dim, output_dim, bias=True)
        self.act_q3 = QuantAct("fc3_out")

    def forward(self, x):
        x = self.input_q(x)
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.act_q(x)
        x = self.fc2(x)
        x = self.act_q2(x)
        x = self.fc3(x)
        x = self.act_q3(x)
        return x


def _calc_tensor_qparams(t: torch.Tensor, qmin: int = -128, qmax: int = 127):
    """
    weight/bias 的 qparams 也统一成对称量化（zp=0）。
    """
    min_val = float(t.min())
    max_val = float(t.max())
    if max_val <= min_val:
        max_val = min_val + 1e-6
    abs_max = max(abs(min_val), abs(max_val))
    if abs_max == 0.0:
        abs_max = 1e-6
    scale = abs_max / max(abs(qmin), abs(qmax))
    zp = 0
    return scale, int(zp), min_val, max_val


def collect_qparams(model: nn.Module) -> Dict[str, Any]:
    """
    收集：
      1) 所有 QuantAct 的激活统计（input, fc1_out, fc2_out）
      2) 所有 QuantLinear 层的 weight / bias 量化参数
    """
    qparams: Dict[str, Any] = {}

    # 激活
    for name, m in model.named_modules():
        if isinstance(m, QuantAct):
            scale, zp = m.calc_qparams()
            qparams[m.name] = {
                "type": "activation",
                "module_path": name,
                "scale": scale,
                "zero_point": zp,
                "qmin": m.qmin,
                "qmax": m.qmax,
                "min_val": float(m.min_val),
                "max_val": float(m.max_val),
            }

    # 权重 + bias
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear):
            w_scale, w_zp, w_min, w_max = _calc_tensor_qparams(
                m.linear.weight.data, -128, 127
            )
            qparams[f"{name}.weight"] = {
                "type": "weight",
                "module_path": f"{name}.weight",
                "scale": w_scale,
                "zero_point": w_zp,  # = 0
                "qmin": -128,
                "qmax": 127,
                "min_val": w_min,
                "max_val": w_max,
            }
            if m.linear.bias is not None:
                b_scale, b_zp, b_min, b_max = _calc_tensor_qparams(
                    m.linear.bias.data, -128, 127
                )
                qparams[f"{name}.bias"] = {
                    "type": "bias",
                    "module_path": f"{name}.bias",
                    "scale": b_scale,
                    "zero_point": b_zp,  # = 0
                    "qmin": -128,
                    "qmax": 127,
                    "min_val": b_min,
                    "max_val": b_max,
                }

    return qparams


def set_export_mode(model: nn.Module, export: bool = True):
    # 目前不建议在导出时关掉 fake-quant（需要 Q/DQ），保留接口以备后用。
    for m in model.modules():
        if isinstance(m, QuantAct):
            m.export_mode = export


def save_qparams(qparams: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(qparams, f, indent=2)


def calibrate_model(model: nn.Module, data_loader, device: torch.device, num_batches: int = 10):
    model.to(device)
    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            x = x.to(device)
            _ = model(x)
    model.eval()
    return model