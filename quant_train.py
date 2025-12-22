import argparse
import os
import time
import logging
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler.scheduler_factory import create_scheduler
from timm.optim._optim_factory import create_optimizer
from timm.utils.cuda import NativeScaler 
from timm.utils.model import get_state_dict
from timm.utils.model_ema import ModelEma
from timm.utils.metrics import accuracy

from models import *
from utils import *


parser = argparse.ArgumentParser(description="I-ViT")

""">>>>>>>>>>>>>>>>>>>>>>>>>>>邹子涵 参数加入>>>>>>>>>>>>>>>>>>>>>>>>"""
parser.add_argument("--run_mode", default='inference', choices=['train', 'inference'],
                    help='使用infernece模式进行推理验证精度，并默认使用results/checkpoint.pth.tar模型\
                        对train则同I-ViT的默认方式进行训练')
parser.add_argument("--calibrated", default=False, type=bool, help="推理验证时有效，若使能则对检查点进行量化因子校准，默认为False，即没有校准需要重新校准")
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<邹子涵 参数加入<<<<<<<<<<<<<<<<<<<<<<<<"""

parser.add_argument("--model", default='deit_tiny',
                    choices=['deit_tiny', 'deit_small', 'deit_base', 
                             'swin_tiny', 'swin_small', 'swin_base'],
                    help="model")
parser.add_argument('--data', metavar='DIR', default='/home/_shareFolder/_dataSets/Imagenet2012/',
                    help='path to dataset')
parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET'],
                    type=str, help='Image Net dataset path')
parser.add_argument("--nb-classes", default=1000, type=int, help="number of classes")
parser.add_argument('--input-size', default=224, type=int, help='images input size')
parser.add_argument("--device", default="cuda", type=str, help="device")
parser.add_argument("--print-freq", default=1000,
                    type=int, help="print frequency")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument('--output-dir', type=str, default='results/',
                    help='path to save log and quantized model')

parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=90, type=int)
# 使用CPU来转移一部分DataLoader的计算，以提高GPU的利用率和性能
parser.add_argument('--pin-mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                    help='')
parser.set_defaults(pin_mem=True)

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

# 添加了一些与模型指数移动平均（Exponential Moving Average，EMA）相关的参数。
# 指数移动平均技术是一种用于平滑模型参数的技术，在训练过程中可以提高模型的性能和泛化能力。
parser.add_argument('--model-ema', action='store_true')
parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
# 这个参数用于指定优化器的 epsilon 值。epsilon 是一个很小的数，用于防止除零错误（例如在计算优化器的动量时)
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
# 指定优化器的 beta 参数。beta 是优化器中一些算法（如 Adam）中的超参数，控制了梯度的指数加权平均的衰减率。
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
# 这个参数用于指定梯度裁剪的阈值。梯度裁剪可以防止梯度爆炸问题。
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
#这个参数用于指定 SGD 优化器的动量。
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# 用于指定权重衰减（weight decay）的参数。权重衰减是一种正则化方法，用于控制模型的复杂度，防止过拟合。
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')

# Learning rate schedule parameters
# 这个参数用于指定学习率调度器的类型。在这里，'cosine' 表示使用余弦退火调度器（Cosine Annealing Scheduler）
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default: 1e-6)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
# 指定学习率噪声的标准差
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=5e-7, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
# 线性退火参数
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
# 耐心周期数内，模型性能没有明显提升，就可以将学习率降低。
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
# 指定学习率的衰减率；新学习率 = 旧学习率 * 衰减率。
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
# 随机调整图像的颜色和对比度来生成新的训练样本，从而提高模型的鲁棒性。
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
# 自动化的数据增强方法选择
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + \
                           "(default: rand-m9-mstd0.5-inc1)'),
# 控制标签平滑的程度；标签平滑可以帮助改善模型的泛化能力，并减少过拟合的风险。
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

# * Random Erase params
# 数据增强技术，随机擦除（Random Erasing），用于提高模型的鲁棒性。
# 指定执行随机擦除的概率，即每个样本被随机擦除的概率。
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
# 指定随机擦除的模式，即采用哪种方式进行擦除。例如像素擦除（pixel）或区域擦除。
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
# 指定随机擦除的次数，即每个样本被擦除的次数。
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
# 控制是否在第一个增强操作（干净增强）之前执行随机擦除
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# * Mixup params
#  Mixup 是一种数据增强技术，它通过对两张图像的像素值及其对应的标签进行线性插值来生成新的训练样本。
# 控制 Mixup 技术的应用程度
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
#  CutMix 是一种类似于 Mixup 的数据增强技术，它不是简单地对两张图像进行线性插值，
# 而是通过随机选择一个区域并用另一张图像的对应区域来替换它，然后混合两个标签。
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
# 指定 CutMix 的最小和最大替换区域的大小比例
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
# 指定执行 Mixup 或 CutMix 的概率
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
# 指定在同时启用 Mixup 和 CutMix 时切换到 CutMix 的概率
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
# 指定如何应用 Mixup 或 CutMix 参数。可以选择批次级别（"batch"）、对级别（"pair"）或元素级别（"elem"）。
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

parser.add_argument('--best-acc1', type=float, default=0, help='best_acc1')


def str2model(name):
    '''
    通过字符建立网络模型导入
    '''
    d = {'deit_tiny': deit_tiny_patch16_224,
         'deit_small': deit_small_patch16_224,
         'deit_base': deit_base_patch16_224,
         'swin_tiny': swin_tiny_patch4_window7_224,
         'swin_small': swin_small_patch4_window7_224,
         'swin_base': swin_base_patch4_window7_224,
         }
    print('Model: %s' % d[name].__name__) # 打印函数名
    return d[name]


def main_train(args):
    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Python日志和输出配置与保存>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    # 忽略Python输出的警告信息
    import warnings
    warnings.filterwarnings('ignore')

    # 建立checkpoint文件夹
    if not os.path.exists(args.output_dir): # ./results
        os.makedirs(args.output_dir)

    import logging  # 导入 logging 模块，用于记录日志信息
    # 配置日志记录器
    logging.basicConfig(
        format='%(asctime)s - %(message)s',  # 日志记录的格式，包括时间戳和消息内容
        datefmt='%d-%b-%y %H:%M:%S',  # 时间戳的格式
        filename=args.output_dir + 'train.log'  # 日志记录写入的文件路径
    )
    # 设置日志记录器的级别为 INFO，即只记录 INFO 级别及以上的日志消息
    logging.getLogger().setLevel(logging.INFO)
    # 添加一个处理程序，将日志消息输出到标准输出（终端窗口）
    logging.getLogger().addHandler(logging.StreamHandler())
    # 记录并输出一条信息，包含参数 args 的值
    logging.info(args)
    

    """>>>>>>>>>>>>>>>>>>>>>>>>Mixup/CutMix 数据增强>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    # 创建一个变量用于存储 Mixup 类的实例
    mixup_fn = None
    # 检查是否启用了 Mixup 或 CutMix 数据增强
    # 如果 args.mixup、args.cutmix 或 args.cutmix_minmax 有任意一个不为 0 或 None，则认为启用了 Mixup 或 CutMix
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # 如果启用了 Mixup 或 CutMix，则创建 Mixup 类的实例
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,  # Mixup 参数 alpha，用于控制 Mixup 的强度
            cutmix_alpha=args.cutmix,  # CutMix 参数 alpha，用于控制 CutMix 的强度
            cutmix_minmax=args.cutmix_minmax,  # CutMix 参数 min/max ratio，用于控制替换区域的大小范围
            prob=args.mixup_prob,  # Mixup 或 CutMix 执行的概率
            switch_prob=args.mixup_switch_prob,  # 在启用了 Mixup 和 CutMix 时切换到 CutMix 的概率
            mode=args.mixup_mode,  # Mixup 或 CutMix 的应用方式（batch、pair 或 elem）
            label_smoothing=args.smoothing,  # 标签平滑参数，用于减少模型对训练数据的过拟合
            num_classes=args.nb_classes  # 数据集中的类别数
        )
        
        
    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>pytorch 配置>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    # 设定种子，保证结果可复现
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # 启动cudnn的benchmark模式，针对不同的硬件配置和网络结构进行了优化，以提供最佳的性能。
    torch.backends.cudnn.benchmark = True
    
    
    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>模型配置>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    # Model 建立
    model = str2model(args.model)(pretrained=True,
                                  num_classes=args.nb_classes,
                                  drop_rate=args.drop,
                                  drop_path_rate=args.drop_path)
    device = torch.device(args.device)
    model.to(device)
    
    # Model EMA 配置
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
    
    args.min_lr = args.lr / 15
    optimizer = create_optimizer(args, model)
    # NativeScaler 可能是一个用于在训练过程中缩放损失值的工具类，用于处理梯度以防止梯度爆炸或梯度消失等问题。
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # 损失函数配置，如果使用图像增强了会使用别的损失函数
    """
        CrossEntropyLoss（交叉熵损失）：

        这是最常见的分类问题损失函数之一。
        用于多分类问题，要求目标值为类别索引。
        通常用于训练标准的分类模型。
        不做任何平滑处理，对训练数据的拟合较为直接，可能会导致模型对训练数据过拟合。
        SoftTargetCrossEntropy（软目标交叉熵损失）：

        用于在混合数据增强（如 Mixup）时处理软标签。
        Mixup 是一种数据增强技术，它通过对两张图像的像素值及其对应的标签进行线性插值来生成新的训练样本。SoftTargetCrossEntropy 用于处理 Mixup 中生成的软标签，即混合两个标签的加权平均值。
        在 SoftTargetCrossEntropy 中，平滑处理通常会包含在 Mixup 标签转换中，因此不需要额外的平滑处理。
        LabelSmoothingCrossEntropy（标签平滑交叉熵损失）：

        用于缓解标签过度自信的问题，增加模型的泛化能力。
        在标签平滑中，将真实标签从 0 或 1 平滑到一个更小的值，以减少模型对训练数据的过拟合。
        通过将真实标签替换为一个介于 0 和 1 之间的值，使得模型更加鲁棒，能够更好地处理噪声和不确定性。
        标签平滑的方法可以有多种，最简单的是将原始的 0 或 1 标签分别替换为一个较小的值和一个较大的值，从而产生一个连续的标签分布。
    """
    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion_v = nn.CrossEntropyLoss() # 验证时不使用增强，使用普通交叉熵损失函数 

    # 如果设置了 resume 参数，表示要从之前的检查点恢复训练
    if args.resume:
        # 如果恢复的路径是一个 URL 地址
        if args.resume.startswith('https'):
            # 从 URL 加载模型的状态字典
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        # 如果恢复的路径是本地文件路径
        else:
            # 从本地文件加载模型的状态字典
            checkpoint = torch.load(args.resume, map_location='cpu')

        # 加载模型的状态字典到当前模型中
        model.load_state_dict(checkpoint['model'])

        # 如果不是在评估模式下，并且检查点中包含了优化器、学习率调度器和训练轮数信息
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # 加载优化器和学习率调度器的状态字典
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            # 更新开始的训练轮数，使其从上次训练结束的轮数 + 1 开始
            args.start_epoch = checkpoint['epoch'] + 1

            # 如果启用了模型指数移动平均
            if args.model_ema:
                # 从检查点中加载指数移动平均模型的状态
                load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

            # 如果检查点中包含了损失缩放器的状态
            if 'scaler' in checkpoint:
                # 加载损失缩放器的状态
                loss_scaler.load_state_dict(checkpoint['scaler'])

        # 更新学习率调度器的状态，使其从开始的训练轮数开始
        lr_scheduler.step(args.start_epoch)

    # 开始训练
    print(f"Start training for {args.epochs} epochs")
    train_loader, val_loader = dataloader(args) # 获取训练集和验证集的数据加载器
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch,
              loss_scaler, args.clip_grad, model_ema, mixup_fn, device)
        lr_scheduler.step(epoch)

        acc1 = validate(args, val_loader, model, criterion_v, device)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > args.best_acc1
        args.best_acc1 = max(acc1, args.best_acc1)
        if is_best:
            # record the best epoch
            best_epoch = epoch
            checkpoint_path = os.path.join(args.output_dir, f'cpt_{args.model}_acc{args.best_acc1}_{datetime.now().strftime("%Y-%m%d-%H:%M")}.pth.tar')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)
        logging.info(f'Acc at epoch {epoch}: {acc1}')
        logging.info(f'Best acc at epoch {best_epoch}: {args.best_acc1}')


def train(args, train_loader, model, criterion, optimizer, epoch, loss_scaler, max_norm, model_ema, mixup_fn, device):
    # 定义用于记录训练统计信息的各种指标
    batch_time = AverageMeter('Time', ':6.3f')  # 批处理时间
    data_time = AverageMeter('Data', ':6.3f')  # 数据加载时间
    losses = AverageMeter('Loss', ':.4e')      # 损失值
    progress = ProgressMeter(                  # 训练进度条
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # 将模型设置为训练模式
    model.train()
    unfreeze_model(model)  # 解冻模型，以便在训练时更新所有参数
    un_int_model(model)

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # 记录数据加载时间
        data_time.update(time.time() - end)

        # 将数据和目标移动到设备上（GPU 或 CPU）
        # non_blocking=True 是用于异步数据传输的参数；即不会阻塞当前线程，而是允许线程继续执行其他操作。可以提高代码的效率和性能。
        data = data.to(device, non_blocking=True) 
        target = target.to(device, non_blocking=True)

        # 如果启用了 Mixup 数据增强，对数据进行 Mixup 处理
        if mixup_fn is not None:
            data, target = mixup_fn(data, target)

        # 前向传播，计算模型输出和损失值
        output = model(data)
        loss = criterion(output, target)

        # 记录损失值
        losses.update(loss.item(), data.size(0))

        # 梯度清零，进行反向传播，更新模型参数
        optimizer.zero_grad()
        # is_second_order 的值表示当前是否需要计算二阶梯度，即是否需要创建计算图以便进行二阶梯度计算
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        #执行损失的梯度缩放和反向传播
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        # 同步所有 GPU，确保当前批次的计算完成
        torch.cuda.synchronize()

        # 更新指数移动平均模型的参数
        if model_ema is not None:
            model_ema.update(model)

        # 记录批处理时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 如果达到打印频率，则打印训练进度
        if i % args.print_freq == 0:
            progress.display(i)


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
    
    freeze_model(model) # 递归冻结模型的激活范围，以便在验证时不更新激活范围
    int_model(model)

    end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (args.print_freq/100) == 0:
            progress.display(i)

    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    return top1.avg

def calibrate_model(model, val_loader, args, patch_num=10):
    """_summary_
    模型校准，跑若干patch的验证集，来校准量化因子
    pretrained: 
        是否之前已经矫正过的模型，而不是基于数据重新矫正。
        默认情况下载训练模式中自动保存的检查点是经过数据集矫正的。
    """
    logging.info('INFO: 开始校准模型量化因子并加载检查点...')
    
    device = torch.device(args.device)
    checkpoint_path = args.resume
    
    # 跑一个batch，初始化模型量化参数
    # 若不加则参数检查点导入报错 
    # 何意味
    model.eval()
    for _, (data, target) in enumerate(val_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            model(data)
            break
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    
    if args.calibrated:
        # 不用重新矫正，直接退出函数
        logging.info("INFO: 使用默认模型，跳过矫正！")
        return model
    
    # 量化因子矫正
    for i, (data, target) in enumerate(val_loader):
        if i > patch_num:
            break
        
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            model(data)
            
        logging.info("校准进度：[%d/%d]", i, patch_num)
    
    # 保存校准模型
    calibrated_model_path = args.output_dir + 'calibrated_model.pth'
    torch.save(model.state_dict(), calibrated_model_path)
    logging.info("INFO: 校准模型保存在：%s", calibrated_model_path)
    # 打印校准模型参数摘要
    calibrate_model_abstract_path = args.output_dir + 'calibrated_model_abstract.log'
    cpt = torch.load(calibrated_model_path)
    with open(calibrate_model_abstract_path, "w") as file:
        print(cpt, file=file)
    logging.info("INFO: 校准模型参数摘要保存在：%s", calibrate_model_abstract_path)   

    logging.info("INFO: 校准完成！")
    return model


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

def main_inference(args):
    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>邹子涵 新建推理函数>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Python日志和输出配置与保存>>>>>>>>>>>>>>>>>>>>>>>>>"""
    # 忽略Python输出的警告信息
    import warnings
    warnings.filterwarnings('ignore')

    # 建立checkpoint文件夹
    if not os.path.exists(args.output_dir): # ./results
        os.makedirs(args.output_dir)

    import logging  # 导入 logging 模块，用于记录日志信息
    # 配置日志记录器
    logging.basicConfig(
        format='%(asctime)s - %(message)s',  # 日志记录的格式，包括时间戳和消息内容
        datefmt='%d-%b-%y %H:%M:%S',  # 时间戳的格式
        filename=args.output_dir + 'inference_log.log'  # 日志记录写入的文件路径
    )
    # 设置日志记录器的级别为 INFO，即只记录 INFO 级别及以上的日志消息
    logging.getLogger().setLevel(logging.INFO)
    # 添加一个处理程序，将日志消息输出到标准输出（终端窗口）
    logging.getLogger().addHandler(logging.StreamHandler())
    # 记录并输出一条信息，包含参数 args 的值
    logging.info(args)
    logging.info('log format is:\t name current_patch_val_acc (average_val_acc)')
    
    
    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>pytorch 配置>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    # 设定种子，保证结果可复现
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # 启动cudnn的benchmark模式，针对不同的硬件配置和网络结构进行了优化，以提供最佳的性能。
    torch.backends.cudnn.benchmark = True
    
    
    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>模型配置>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    # 检查点路径
    if args.resume:
        checkpoint_path = args.resume
    else:
        raise ValueError('Please specify the checkpoint path')
    
    # 加载模型
    model = str2model(args.model)(pretrained=True,
                                  num_classes=args.nb_classes,
                                  drop_rate=args.drop,
                                  drop_path_rate=args.drop_path)
    device = torch.device(args.device)
    model.to(device)

    # 加载数据
    _, val_loader = dataloader(args)
    
    # 校准模型&加载检查点
    model = calibrate_model(model, val_loader, args)

    # 验证模型
    validate(args, val_loader, model, nn.CrossEntropyLoss(), device)

if __name__ == "__main__":
    args = parser.parse_args()
    args.run_mode = 'inference'
    args.resume = 'results/calibrated_model.pth' # 接续训练和推理时使用, 重新训练不用
    args.model = 'deit_tiny'
    args.data = '/home/_shareFolder/_dataSets/Imagenet2012/'
    args.calibrated = False # 推理验证模式下有效

    args.print_freq = 1000
    args.batch_size = 1
   
    args.epochs = 3
    args.lr = 5e-7

    if args.run_mode == 'train':
        # 训练模式下使用I-ViT的默认配置进行训练
        main_train(args)
    elif args.run_mode == 'inference':
        # 推理模式下使用I-ViT的默认配置进行推理
        main_inference(args)
