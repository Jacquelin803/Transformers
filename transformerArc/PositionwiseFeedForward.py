
"""
前馈全连接层:
在Transformer中前馈全连接层就是具有两层线性层的全连接网络

前馈全连接层的作用:
考虑注意力机制可能对复杂过程的拟合程度不够, 通过增加两层网络来增强模型的能力.

"""

import torch
import torch.nn as nn
import torch.functional as F

class PositionwiseFeedForward(nn.Module):

    """
    初始化函数有三个输入参数分别是d_model, d_ff,和dropout=0.1，
    第一个是线性层的输入维度也是第二个线性层的输出维度，因为我们希望输入通过前馈全连接层后输入和输出的维度不变.
    第二个参数d_ff就是第二个线性层的输入维度和第一个线性层的输出维度.
    最后一个是dropout置0比率.
    """
    def __init__(self,d_model,d_ff,dropout):

        super(PositionwiseFeedForward, self).__init__()

        self.w1=nn.Linear(d_model,d_ff)
        self.w2=nn.Linear(d_ff,d_model)

        self.dropout=nn.Dropout(dropout)

    """输入参数为x，代表来自上一层的输出"""
    # 首先经过第一个线性层，然后使用Funtional中relu函数进行激活,
    # 之后再使用dropout进行随机置0，最后通过第二个线性层w2，返回最终结果.
    def forward(self,x):
        return self.w2(self.dropout(torch.relu(self.w1(x))))


# if __name__ == '__main__':
#
#     test in "EncoderLayer.py"




