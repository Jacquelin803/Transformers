
import torch.nn as nn
import torch
import copy

from torch.autograd import Variable

from transformerArc.NormLayer import NormLayer

"""
编码器用于对输入进行指定的特征提取过程, 也称为编码, 由N个编码器层堆叠而成.
"""

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    """初始化函数的两个参数分别代表编码器层和编码器层的个数"""
    def __init__(self,layer,N):
        super(Encoder, self).__init__()
        self.layers=clones(layer,N)
        self.norm=NormLayer(layer.d_model)

    """forward函数的输入和编码器层相同, x代表上一层的输出, mask代表掩码张量"""
    def forward(self,x,mask):
        # 首先就是对我们克隆的编码器层进行循环，每次都会得到一个新的x，
        # 这个循环的过程，就相当于输出的x经过了N个编码器层的处理.
        # 最后再通过规范化层的对象self.norm进行处理，最后返回结果.
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
