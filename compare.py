import torch
import tensorrt as trt
import numpy as np
import time
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets.folder import ImageFolder
import torchvision.models as models
from tqdm import tqdm
import timm
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform, resolve_data_config

# 参数设置

BATCH_SIZE = 256  
MAX_BATCH_SIZE=512  # max batch size在build_trt_engine中设定
IMGNET = '/root/autodl-tmp/imagenet'
ENGINE_PTH = '/root/autodl-tmp/QuantViT.engine'
MODEL_NAME = 'deit_tiny_patch16_224'

# ==========================================
# 1.1 TensorRT 推理封装类 (核心工具)
# ==========================================
class TRTWrapper:
    def __init__(self, engine_path, max_batch_size=256):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 加载 Engine
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
        # 预分配显存（绑定输入输出）
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        # 遍历 Engine 的绑定节点
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            # 获取形状 (此时可能是动态的，比如 [-1, 3, 224, 224])
            shape = list(self.engine.get_tensor_shape(name))
            
            # ========================================================
            # 🛠️ 修改 1: 显存分配策略
            # 如果维度是 -1 (动态)，我们必须按最大可能的 Batch Size 分配显存
            # ========================================================
            if shape[0] == -1:
                shape[0] = max_batch_size
                print(f"Allocating memory for dynamic tensor {name} with max batch: {max_batch_size}")
            
            # 将 TRT dtype 转换为 PyTorch dtype
            torch_dtype = self._trt_to_torch_dtype(dtype)
            
            # 在 GPU 上分配显存 (按最大形状分配)
            tensor = torch.zeros(tuple(shape), dtype=torch_dtype, device='cuda')
            self.allocations.append(tensor.data_ptr())
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'tensor': tensor, 'idx': i})
            else:
                self.outputs.append({'name': name, 'tensor': tensor, 'idx': i})

    def _trt_to_torch_dtype(self, trt_dtype):
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool
        }
        return mapping.get(trt_dtype, torch.float32)

    def infer(self, input_tensor):
        """
        执行推理
        input_tensor: PyTorch Tensor (在 GPU 上), shape=[current_batch, 3, 224, 224]
        """
        # 获取当前实际的 batch size
        current_batch_size = input_tensor.size(0)
        
        # 1. 找到输入节点
        eg_input_info = self.inputs[0]
        eg_input_tensor = eg_input_info['tensor'] # 这是预分配的 [Max_Batch, ...]
        eg_input_name = eg_input_info['name']

        # ========================================================
        # 🛠️ 修改 2: 设置动态维度 (Context Set Shape)
        # ========================================================
        # 告诉 TensorRT 这次推理的实际形状
        self.context.set_input_shape(eg_input_name, input_tensor.shape)

        # ========================================================
        # 🛠️ 修改 3: 内存拷贝 (Slice Copy)
        # ========================================================
        # 预分配的显存可能比当前输入大 (例如分配了 256，但当前只来了 128)
        # 我们只拷贝有效数据部分
        # 确保显存清零或直接覆盖 (这里是覆盖前 N 个位置)
        eg_input_tensor[:current_batch_size].copy_(input_tensor)
        
        # 2. 设置 Tensor 地址
        # 注意：地址还是原来分配的首地址，不需要变
        for i, addr in enumerate(self.allocations):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, addr)

        # 3. 执行异步推理
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        
        # 4. 同步流
        self.stream.synchronize()
        
        # ========================================================
        # 🛠️ 修改 4: 输出截断 (Slice Output)
        # ========================================================
        # 输出的显存也是按 Max Batch 分配的，我们只返回当前 Batch 的部分
        output_tensor = self.outputs[0]['tensor']
        return output_tensor[:current_batch_size]

# ==========================================
# 1.2 Data预加载 封装类 (核心工具)
# ==========================================
class DataPrefetcher:
    def __init__(self, loader, device='cuda'):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        
        # 定义归一化参数 (注意要放到 GPU 上)
        # 这里的 mean/std 是 ViT 的 0.5/0.5
        self.mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        # 关键：在专门的 stream 中进行传输
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)
            
            # 关键：在 GPU 上做归一化 (速度极快)
            # 假设 ToTensor 出来是 [0, 1]，这里做 (x - 0.5) / 0.5
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


# ==========================================
# 2. 通用评测函数
# ==========================================
def run_benchmark(model_name, predict_func, dataloader, device='cuda'):
    print(f"\n🚀 开始评测: {model_name} ...")
    
    correct = 0
    total = 0
    total_time = 0.0
    
    # Warmup
    print("   正在预热 (Warmup)...")
    # 使用随机数据预热，避免 DataLoader 带来的额外 I/O 干扰
    # 使用 Max Batch 进行预热，确保所有 Kernel 都初始化
    max_bs = dataloader.batch_size
    dummy_input = torch.randn(max_bs, 3, 224, 224, device=device) 
    
    with torch.no_grad(): # 加上 no_grad 防止 OOM
        for _ in range(10):
            _ = predict_func(dummy_input)
    torch.cuda.synchronize()
    
    print("   开始正式测试...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, ascii=True):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            torch.cuda.synchronize()
            start = time.time()
            
            # 推理
            outputs = predict_func(inputs)
            
            torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_latency = (total_time / len(dataloader)) * 1000  # ms
    throughput = total / total_time  # samples per second
    accuracy = 100 * correct / total
    
    return avg_latency, throughput, accuracy

# ==========================================
# 3. 主程序
# ==========================================
def main():
    # 1. PyTorch 模型
    print("1. 加载 PyTorch 模型...")
    torch_model = timm.create_model(MODEL_NAME, pretrained=True).cuda().eval()
    config = resolve_data_config({}, model=torch_model)
    val_transform = create_transform(**config, is_training=False)
    
    # 2. TensorRT Engine
    engine_file = ENGINE_PTH    
    print(f"2. 加载 TensorRT Engine: {engine_file}...")
    
    # ⚠️ 传入 BATCH_SIZE 告诉 Wrapper 最大需要分配多少显存
    trt_model = TRTWrapper(engine_file, max_batch_size=MAX_BATCH_SIZE)

    # 3. 数据集
    print("3. 准备数据集...")    
    dataset = ImageFolder(os.path.join(IMGNET, 'val'), transform=val_transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True, 
        drop_last=False, # ✅ 即使最后剩下不足 256 个，现在的代码也能处理了
        num_workers=16
    )

    # ================= 运行对比 =================
    def torch_predict(x):
        return torch_model(x)
            
    t_lat, t_fps, t_acc = run_benchmark("PyTorch (FP32)", torch_predict, dataloader)
    
    def trt_predict(x):
        return trt_model.infer(x)
        
    trt_lat, trt_fps, trt_acc = run_benchmark("TensorRT Engine", trt_predict, dataloader)

    
    # ================= 打印报告 =================
    print(f"batch size: {BATCH_SIZE}")
    print("\n" + "="*50)
    print(f"{'Metric':<20} | {'PyTorch':<15} | {'TensorRT':<15} | {'Gain':<10}")
    print("-" * 65)
    print(f"{'Latency (ms/batch)':<20} | {t_lat:<15.4f} | {trt_lat:<15.4f} | x{t_lat/trt_lat:.2f}")
    print(f"{'Throughput (FPS)':<20} | {t_fps:<15.1f} | {trt_fps:<15.1f} | x{trt_fps/t_fps:.2f}")
    print(f"{'Accuracy (%)':<20} | {t_acc:<15.2f} | {trt_acc:<15.2f} | Diff: {trt_acc-t_acc:.2f}")
    print("="*50)


if __name__ == "__main__":
    main()