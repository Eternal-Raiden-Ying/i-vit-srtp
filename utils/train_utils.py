import torch
from torch.nn import functional as F
import io


class DistillationLoss(torch.nn.Module):
    """
    该模块包装了一个标准的损失函数，并通过添加额外的知识蒸馏损失来提供额外的监督。
    """
    
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        """
        初始化函数，接受标准的损失函数、教师模型、知识蒸馏类型、alpha 和 tau 参数。
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        正向传播函数，计算损失。
        
        Args:
            inputs: 输入数据，用于教师模型的前向传播。
            outputs: 训练模型的输出，可以是一个 Tensor 或一个 Tuple[Tensor, Tensor]，
                     其中第一个位置是原始输出，第二个位置是蒸馏预测。
            labels: 基准损失函数的标签。
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # 假设模型输出一个元组 [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("当启用知识蒸馏时，预期模型返回一个包含类标记和蒸馏标记的 Tuple[Tensor, Tensor]")

        # 不要通过教师模型反向传播
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # 从 https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100 获取
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss



def load_checkpoint_for_ema(model_ema, checkpoint):
    """
    为 ModelEma._load_checkpoint 提供的解决方案，使其可以接受已加载的对象。
    
    这段代码定义了一个函数 load_checkpoint_for_ema，
    用于为 ModelEma._load_checkpoint 提供一个解决方案，使其可以接受已加载的对象。具体来说：

    该函数接受两个参数，分别是 model_ema（ModelEma 对象）和 checkpoint（已加载的对象）。
    首先，代码创建了一个 io.BytesIO 对象 mem_file，然后使用 torch.save 将 checkpoint 对象保存到 mem_file 中。
    接着，通过将文件指针移回文件的开头（mem_file.seek(0)），确保后续读取操作从文件的起始位置开始。
    最后，调用 model_ema._load_checkpoint(mem_file)，将 mem_file 中的内容加载到 ModelEma 对象中。
    
    这个函数的作用是将已加载的对象转换为字节流，并将其加载到 ModelEma 对象中，
    从而允许 ModelEma._load_checkpoint 方法接受已加载的对象作为输入。
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)
