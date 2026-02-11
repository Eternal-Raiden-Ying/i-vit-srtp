import argparse
import os
import time
import math
import logging
import numpy as np
from tqdm import tqdm

import torch
from pathlib import Path

import timm
from timm.models import create_model
from timm.utils import accuracy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from models import *
from utils import *

parser = argparse.ArgumentParser(description="I-ViT")

parser.add_argument("--name", type=str, default="ivit", help="specify the name, which is used in filename")
parser.add_argument("--model", default='deit_tiny', help="model",
                    choices=['deit_tiny', 'deit_small', 'deit_base', 'vit_base', 'vit_large'],)
parser.add_argument('--data', metavar='DIR', default=r"/root/autodl-tmp/imagenet",
                    help='path to imagenet')
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument('--output-dir', type=str, default='./results/val/',
                    help='path to save log and quantized model')
parser.add_argument('--ckpt-pth', default='', type=str, help="checkpoint path")

parser.add_argument("--nb-classes", default=1000, type=int, help="number of classes")
parser.add_argument('--input-size', default=224, type=int, help='images input size')
parser.add_argument('--crop-pct', default=0.9, type=float, help='img crop percent, for transform cfg (val)')
parser.add_argument("--device", default="cuda:0", type=str, help="device")

parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--pin-mem', action='store_true', default=True,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                    help='')

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')


def str2model(name):
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
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=args.output_dir + f'{args.name}(validate).log',
                        level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)

    device = torch.device(args.device)

    val_transform = transforms.Compose([
        transforms.Resize(int(args.input_size/args.crop_pct), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Google ViT style
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)  # timm style
    ])

    # Dataset
    dataset = ImageFolder(os.path.join(args.data, 'val'), transform=val_transform)
    val_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=args.pin_mem, 
        drop_last=False,                 # validate 需要验证所有 val img
        num_workers=args.num_workers
    )

    # Model
    model = str2model(args.model)(pretrained=True,
                                  num_classes=args.nb_classes,
                                  drop_rate=args.drop,
                                  drop_path_rate=args.drop_path)

    # ckpt =  torch.load('/root/autodl-tmp/i-vit-srtp/results/training/best_checkpoint.pth.tar', weights_only=False)
    ckpt =  torch.load(args.ckpt_pth, weights_only=False)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt.keys() else ckpt)
    model.to(device).eval()
    freeze_model(model)
    export_model(model)

    # model = timm.create_model('deit_tiny_patch16_224', pretrained=True).cuda().eval() # Acc@1:72.166, Acc@5:91.110
    
    validate(args, val_loader, model, device)



def validate(args, val_loader, model, device):
    print("   正在预热 (Warmup)...")
    dummy_input, _ = next(iter(val_loader))
    dummy_input = dummy_input.to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    print("   开始正式测试...")
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    total_img = 0
    total_time = 0
    
    for data, target in tqdm(val_loader, ascii=True):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output = model(data)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        torch.cuda.synchronize()
        end = time.time()
        total_time += (end - start)
        total_img += target.size(0)

    avg_latency = (total_time / len(val_loader)) * 1000  # ms
    throughput = total_img / total_time  # samples per second
    print(f" * Prec@1 {top1.avg:.3f}")
    print(f" * Prec@5 {top5.avg:.3f}")
    print(f" * avg-latency: {avg_latency} ms ")
    print(f" * throughput : {throughput:.3f} img per sec")


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

if __name__ == "__main__":
    main()
