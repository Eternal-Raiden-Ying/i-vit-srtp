import torch
import timm
import os
import argparse
import time
from torchvision import datasets
from torch.utils.data import DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

# ==========================================
# 配置区域
# ==========================================
# 你的 ImageNet 验证集路径 (包含 val 文件夹, 或者直接指向 val 文件夹)
# 结构应该是: /root/.../imagenet/val/n01440764/images...
DEFAULT_DATA_PATH = '/root/autodl-tmp/imagenet/val' 

# 模型名称
MODEL_NAME = 'deit_tiny_patch16_224'
# 如果你想测蒸馏版 (精度更高 ~74.5%)，请改用: 'deit_tiny_distilled_patch16_224'

def validate(model, loader, device):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 切换到评估模式
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(loader, desc="Evaluating")):
            input = input.to(device)
            target = target.to(device)

            # 计算输出
            output = model(input)

            # 测量精度
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            # 更新统计
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # 测量耗时
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, top5.avg

def main():
    parser = argparse.ArgumentParser(description='DeiT FP32 Evaluation')
    parser.add_argument('--data', default=DEFAULT_DATA_PATH, help='path to dataset')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--workers', default=8, type=int)
    args = parser.parse_args()

    # 1. 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ Error: CUDA is not available.")
        return
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # 2. 创建模型 (自动下载权重)
    print(f"🚀 Creating model: {MODEL_NAME}")
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model = model.to(device)
    
    # 打印参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Number of params: {n_parameters/1e6:.2f}M")

    # 3. 【关键步骤】自动获取模型所需的 Transform 配置
    # 这步会自动设置: Input Size (224), Interpolation (bicubic), Mean, Std, Crop Pct
    config = resolve_data_config({}, model=model)
    print(f"✅ Data Config: {config}")
    
    transform = create_transform(**config, is_training=False)
    print(f"🛠️  Transforms: {transform}")

    # 4. 加载数据
    if not os.path.exists(args.data):
        print(f"❌ Error: Dataset path not found: {args.data}")
        return

    dataset = datasets.ImageFolder(args.data, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"📂 Dataset loaded: {len(dataset)} images.")

    # 5. 开始测试
    print("running validation...")
    acc1, acc5 = validate(model, loader, device)

    print("\n" + "="*40)
    print(f"🏁 Final Results for {MODEL_NAME}:")
    print(f"   Top-1 Accuracy: {acc1:.3f}%")
    print(f"   Top-5 Accuracy: {acc5:.3f}%")
    print("="*40)

    """
    Final Results for deit_tiny_patch16_224:
        Top-1 Accuracy: 72.166%
        Top-5 Accuracy: 91.110%
    """

# ==========================================
# 辅助函数
# ==========================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

if __name__ == '__main__':
    main()