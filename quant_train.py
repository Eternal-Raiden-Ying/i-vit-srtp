import argparse
import os
import time
import math
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path

from timm.models import create_model
from timm.utils import accuracy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import *
from utils import * 

parser = argparse.ArgumentParser(description="I-ViT")

# basic
parser.add_argument("--name", type=str, default="ivit", help="logging file name")
parser.add_argument("--model", default='deit_tiny', help="model",
                    choices=['deit_tiny', 'deit_small', 'deit_base', 'vit_base', 'vit_large'])
parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument('--output-dir', type=str, default='./results/training_freeze/',
                    help='path to save log and quantized model')
parser.add_argument('--resume', default='', help='resume from checkpoint')

# dataloader & transform
parser.add_argument('--data', metavar='DIR', default=r"/root/autodl-tmp/imagenet",
                    help='path to dataset')
parser.add_argument('--crop-pct', type=float, default=0.9, help='Crop percentage')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--num-workers', default=28, type=int)
parser.add_argument('--pin-mem', action='store_true',default=True,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                    help='')

# model relevant
parser.add_argument("--nb-classes", default=1000, type=int, help="number of classes")
parser.add_argument('--input-size', default=224, type=int, help='images input size')
parser.add_argument("--device", default="cuda:0", type=str, help="device")
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

# training relevant
parser.add_argument('--lr', default=1e-6, type=float)
parser.add_argument('--epochs', default=4, type=int) # 微调通常不需要 90 epoch，建议改小
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--freeze-epoch', default=2, type=int, help='freeze scale to help model adapt to the fixed scale')
parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.1)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--weight-decay', type=float, default=1e-2,
                    help='weight decay (default: 1e-2 for AdamW)')

# other
parser.add_argument('--best-acc1', type=float, default=0, help='best_acc1')


def str2model(name):
    # 请确保这些模型定义在 models/__init__.py 或 models/vit_quant.py 中可用
    d = {'deit_tiny': deit_tiny_patch16_224,
         'deit_small': deit_small_patch16_224,
         'deit_base': deit_base_patch16_224,
         'vit_base': vit_base_patch16_224,
         'vit_large': vit_large_patch16_224,
         }
    print('Model: %s' % d[name].__name__)
    return d[name]


def main():
    args = parser.parse_args()

    assert os.path.exists(args.data), "invalid dataset path, make sure '--data' instruction is set right"

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    import warnings
    warnings.filterwarnings('ignore')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 设置 Logging
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', 
                        filename=args.output_dir + f'/{args.name}_training.log',
                        level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logging.info(args)

    device = torch.device(args.device)

    # ==========================
    # 1. Dataset & Transform
    # ==========================
    # QAT/微调 推荐配置：弱增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    
    # 验证集：高精度配置 (Resize 248 -> CenterCrop 224)
    resize_size = int(args.input_size / args.crop_pct)
    val_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    train_loader = DataLoader(
        dataset=ImageFolder(os.path.join(args.data, 'train'), transform=train_transform), 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=args.pin_mem, 
        drop_last=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        dataset=ImageFolder(os.path.join(args.data, 'val'), transform=val_transform),
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=args.pin_mem, 
        drop_last=False,
        num_workers=args.num_workers
    )
    
    # ==========================
    # 2. Model
    # ==========================
    print(f"Creating model: {args.model}")
    model = str2model(args.model)(pretrained=True,
                                  num_classes=args.nb_classes,
                                  drop_rate=args.drop,
                                  drop_path_rate=args.drop_path)
    model.to(device)
        
    # ==========================
    # 3. Optimizer & Scheduler
    # ==========================
    # 学习率给小一点 (1e-5)，因为是微调
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    
    # 线性预热衰减策略
    def lr_lambda(current_step):
        num_warmup_steps = int(0.1 * total_steps)
        num_training_steps = total_steps
        
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    # 注意：scheduler 是基于 step 的，需要在每个 batch 后 step
    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    # Loss Function
    if args.smoothing > 0.0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    criterion_v = nn.CrossEntropyLoss()

    # Resume Logic (简化版)
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # ==========================
    # 4. Training Loop
    # ==========================
    print("开始校准 (Calibration)...")
    NUM_CALIB_BATCHES = 100
    unfreeze_model(model)
    un_int_model(model) # 训练用fp
    set_float_op(model, True) # 精度敏感模块采用fp
    
    with torch.no_grad():
        # 关键点：
        # - total=NUM_CALIB_BATCHES: 告诉 tqdm 分母是 100，这样跑完就是 100%
        # - desc: 进度条前缀文字
        # - leave=True: 跑完后保留进度条显示
        pbar = tqdm(train_loader, total=NUM_CALIB_BATCHES, desc="🔍 Calibrating", leave=True)
        
        for i, (data, target) in enumerate(pbar):
            if i >= NUM_CALIB_BATCHES:
                pbar.close() # 手动关闭一下更安全
                break  
            data = data.to(device, non_blocking=True)
            model(data)    
            
    print("✅ 校准完成，开始正式训练 (QAT)...\n")

    
    print(f"Start training for {args.epochs} epochs")
    best_epoch = 0
    freeze_flag = False
    for epoch in range(args.start_epoch, args.epochs):
        if args.freeze_epoch >= args.epochs - epoch:
            freeze_flag = True
        train_acc1 = train(args, train_loader, model, criterion, optimizer, lr_scheduler, epoch, device, freeze_flag)
        
        # 验证
        acc1 = validate(args, val_loader, model, criterion_v, device)
         
        # 保存最佳模型
        is_best = acc1 > args.best_acc1
        if is_best:
            args.best_acc1 = acc1
            best_epoch = epoch
            
        # 无论是否最佳，都保存 checkpoint (覆盖式保存节省空间，或者按 epoch 保存)
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint.pth.tar')
        if is_best:
            best_checkpoint_path = os.path.join(args.output_dir, f'best_checkpoint.pth.tar')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_acc1': args.best_acc1
            }, best_checkpoint_path)
            logging.info(f'Saved best model at epoch {epoch} with acc {acc1:.2f}')

        logging.info(f'Epoch {epoch} finished. Train Acc: {train_acc1:.2f}, Val Acc: {acc1:.2f}, Best Val Acc: {args.best_acc1:.2f} (Epoch {best_epoch})')


def train(args, train_loader, model, criterion, optimizer, scheduler, epoch, device, freeze_flag):
    """
    修改点：
    1. 传入 scheduler，并在每个 batch 后 step。
    2. 移除 loss_scaler, model_ema, mixup_fn。
    3. 使用纯 FP32 训练逻辑。
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    if not freeze_flag:
        unfreeze_model(model)
    
    # 如果你的 utils 里有这些函数，请保留；如果是标准 PyTorch 模型，其实 model.train() 就够了
    # 这里的 unfreeze 可能是为了解冻量化参数

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Forward
        output = model(data)
        loss = criterion(output, target)

        # Measure Accuracy
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))

        # Backward & Optimize
        optimizer.zero_grad()
        loss.backward()
        
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
        optimizer.step()
        
        # NOTE: Scheduler Step 必须在这里 (Per Iteration)
        scheduler.step()

        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            # 获取当前 LR 方便打印
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Step {i}, LR: {current_lr:.2e}")
            progress.display(i)
            
    return top1.avg


def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    
    # 同理，如果 utils 有定义则保留
    freeze_model(model)

    end = time.time()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    # logging.info instead of print for consistency
    logging.info(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main()