"""
子层连接结构:
如transformer图所示，输入到每个子层以及规范化层的过程中，
还使用了残差链接（跳跃连接），因此我们把这一部分结构整体叫做子层连接（代表子层及其链接结构），
在每个编码器层中，都有两个子层，这两个子层加上周围的链接结构就形成了两个子层连接结构
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from transformerArc.NormLayer import NormLayer


class SubLayerConnection(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super(SubLayerConnection, self).__init__()

        # NormLayer
        self.norm = NormLayer(d_model)

        # dropout
        self.dropout = nn.Dropout(p=dropout)

    """前向逻辑函数中, 接收上一个层或者子层的输入作为第一个参数，
       将该子层连接中的子层函数作为第二个参数"""

    def forward(self, x, subLayer):
        # 我们首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作，
        # 随机停止一些网络中神经元的作用，来防止过拟合. 最后还有一个add操作，
        # 因为存在跳跃连接，所以是将输入x与dropout后的子层输出结果相加作为最终的子层连接输出.
        return x + self.dropout(subLayer(self.norm(x)))
