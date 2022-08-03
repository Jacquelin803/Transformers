"""
位置编码器的作用
因为在Transformer的编码器结构中, 并没有针对词汇位置信息的处理，
因此需要在Embedding层后加入位置编码器，将词汇位置不同可能会产生不同语义的信息加入到词嵌入张量中, 以弥补位置信息的缺失.
"""

import torch
import matplotlib.pyplot as plt
# 预定义的网络层torch.nn, 工具开发者已经帮文明开发好的一些常用层
# 例如，卷积层、lstm层、embedding层等，不需要在造轮子
import torch.nn as nn

# 数学计算工具包
import math

# torch中变量封装函数
from torch.autograd import Variable

# 定义位置编码器类, 我们同样把它看做一个层, 因此会继承nn.Module
from transformerArc.EmbLayer import Embeddings


class PositionalEncoding(nn.Module):
    """位置编码器类初始化函数，有三个参数，d_model 词嵌入维度，dropout 置0比率，max_len 每个句子的最大长度"""

    def __init__(self, d_model, drop_out=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 实例化nn中预定义的Dropout层，并将dropout传入其中，获得对象self.dropout
        self.dropout = nn.Dropout(p=drop_out)

        # 初始一个位置编码矩阵，它是一个0矩阵，矩阵的大小为Max_len * d_model
        pe = torch.zeros(max_len, d_model)
        # print("pe",pe,pe.shape)

        # 初始化一个绝对位置矩阵, 在这里，词汇的绝对位置就是用它的索引去表示.
        # 所以我们首先使用arange方法获得一个连续自然数向量，然后再使用unsqueeze方法拓展向量维度使其成为矩阵，
        # 又因为参数传的是1，代表矩阵拓展的位置，会使向量变成一个max_len x 1 的矩阵，
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len]->[max_len,1]
        # print("position shape [max_len,1]",position,position.shape)

        # 绝对位置矩阵初始化之后，接下来就是考虑如何将这些位置信息加入到位置编码矩阵中，
        # 最简单思路就是先将max_len x 1的绝对位置矩阵， 变换成max_len x d_model形状，然后覆盖原来的初始位置编码矩阵即可，
        # 要做这种矩阵变换，就需要一个1xd_model形状的变换矩阵div_term，我们对这个变换矩阵的要求除了形状外，
        # 还希望它能够将自然数的绝对位置编码缩放成足够小的数字，有助于在之后的梯度下降过程中更快的收敛.  这样我们就可以开始初始化这个变换矩阵了.
        # 首先使用arange获得一个自然数矩阵， 但是细心的同学们会发现， 我们这里并没有按照预计的一样初始化一个1xd_model的矩阵，
        # 而是有了一个跳跃，只初始化了一半即1xd_model/2 的矩阵。 为什么是一半呢，其实这里并不是真正意义上的初始化了一半的矩阵，
        # 我们可以把它看作是初始化了两次，而每次初始化的变换矩阵会做不同的处理，第一次初始化的变换矩阵分布在正弦波上， 第二次初始化的变换矩阵分布在余弦波上，
        # 并把这两个矩阵分别填充在位置编码矩阵的偶数和奇数位置上，组成最终的位置编码矩阵.
        temp = torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        # print("temp",temp)
        div_term = torch.exp(temp)
        # print("div_term",div_term.shape,div_term)
        # print("position*div_term",position*div_term)
        pe[:, 0::2] = torch.sin(position * div_term)  # [6.1]*[1,temp.shape/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        # print("pe after div",pe,pe.shape)

        # 这样我们就得到了位置编码矩阵pe, pe现在还只是一个二维矩阵，要想和embedding的输出（一个三维张量,like [2,4,512]）相加，
        # 就必须拓展一个维度，所以这里使用unsqueeze拓展维度.
        pe = pe.unsqueeze(0)
        # print("pe add dimention",pe, pe.shape)

        # 最后把pe位置编码矩阵注册成模型的buffer，什么是buffer呢，
        # 我们把它认为是对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要随着优化步骤进行更新的增益对象.
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数一同被加载.
        self.register_buffer('pe', pe)

    """forward函数的参数是x, 表示文本序列的词嵌入表示"""

    def forward(self, x):
        # 在相加之前我们对pe做一些适配工作， 将这个三维张量的第二维也就是句子最大长度的那一维将切片到与输入的x的第二维相同即x.size(1)，
        # 因为我们默认max_len为5000一般来讲实在太大了，很难有一条句子包含5000个词汇，所以要进行与输入张量的适配.
        # 最后使用Variable进行封装，使其与x的样式相同，但是它是不需要进行梯度求解的，因此把requires_grad设置成false.
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # print(x)
        return self.dropout(x)


if __name__ == '__main__':
    # nn.Dropout演示
    # m=nn.Dropout(p=0.2)
    # input=torch.randn(10,2)
    # output=m(input)
    # print(output)

    # torch.unsqueeze演示: 在哪个维度上增加一个象限就参数设为几
    # x=torch.tensor([[1,2,3,4],[5,6,7,8]])  # 2*4
    # x=torch.tensor([1,2,3,4])  # 1*4
    # print("input.shape",x.shape)
    # out1=torch.unsqueeze(x,0)
    # out2=torch.unsqueeze(x, 1)
    # print(out1,out1.shape)  #
    # print(out2,out2.shape)
    """
    tensor([[[1, 2, 3, 4],
         [5, 6, 7, 8]]]) torch.Size([1, 2, 4])
    tensor([[[1, 2, 3, 4]],
        [[5, 6, 7, 8]]]) torch.Size([2, 1, 4])
    """

    ######## 输入部分 ########
    # 实例化
    # 词嵌入维度是512维
    d_model = 512
    vocab = 1000

    # 置0比率为0.1
    dropout = 0.1

    # 句子最大长度
    max_len = 60

    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    emb = Embeddings(d_model, vocab)
    embRes = emb(x)
    print(embRes, embRes.shape)  # torch.Size([2, 4, 3]
    pe = PositionalEncoding(d_model, dropout, max_len)
    peRes = pe(embRes)
    print(peRes, peRes.shape)
