"""
规范化层的作用:
它是所有深层网络模型都需要的标准网络层，因为随着网络层数的增加，
通过多层的计算后参数可能开始出现过大或过小的情况，这样可能会导致学习过程出现异常，模型可能收敛非常的慢. 因此都会在一定层数后接规范化层进行数值的规范化，使其特征数值在合理范围内
"""
import torch
import torch.nn as nn


class NormLayer(nn.Module):
    """初始化函数有两个参数, 一个是features, 表示词嵌入的维度,
       另一个是eps它是一个足够小的数, 在规范化公式的分母中出现,防止分母为0.默认是1e-6."""

    def __init__(self, d_model, eps=1e-6):
        super(NormLayer, self).__init__()

        # 根据features的形状初始化两个参数张量a2，和b2，第一个初始化为1张量，
        # 也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，这两个张量就是规范化层的参数，
        # 因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子，
        # 使其即能满足规范化要求，又能不改变针对目标的表征.最后使用nn.parameter封装，代表他们是模型的参数。
        self.a2 = nn.Parameter(torch.ones(d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # 输入参数x代表来自上一层的输出
        # 在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致.
        # 接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果，
        # 最后对结果乘以我们的缩放参数，即a2，*号代表同型点乘，即对应位置进行乘法操作，加上位移参数b2.返回即可.
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2
