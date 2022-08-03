

import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from transformerArc.EmbLayer import Embeddings
from transformerArc.Encoder import Encoder
from transformerArc.MultiHeadAttention import MultiHeadAttention
from transformerArc.NormLayer import NormLayer
from transformerArc.PosEncLayer import PositionalEncoding

# 用于深度拷贝的copy工具包
import copy

from transformerArc.PositionwiseFeedForward import PositionwiseFeedForward
from transformerArc.SubLayerConnection import SubLayerConnection


"""
编码器层的作用:
作为编码器的组成单元, 每个编码器层完成一次对输入的特征提取过程, 即编码过程
编码器由N个编码器层堆叠而成.
"""


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderLayer(nn.Module):
    """它的初始化函数参数有四个，分别是d_model，其实就是我们词嵌入维度的大小，它也将作为我们编码器层的大小,
       第二个self_attn，之后我们将传入多头自注意力子层实例化对象, 并且是自注意力机制,
       第三个是feed_froward, 之后我们将传入前馈全连接层实例化对象, 最后一个是置0比率dropout."""

    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 如图所示, 编码器层中有两个子层连接结构, 所以使用clones函数进行克隆
        self.subLayer = clones(SubLayerConnection(d_model, dropout), 2)

        self.d_model = d_model



    """forward函数中有两个输入参数，x和mask，分别代表上一层的输出，和掩码张量mask"""

    def forward(self, x, mask):
        # 里面就是按照结构图左侧的流程. 首先通过第一个子层连接结构，其中包含多头自注意力子层，
        # 然后通过第二个子层连接结构，其中包含前馈全连接子层. 最后返回结果.
        x = self.subLayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.subLayer[1](x, self.feed_forward)






if __name__ == '__main__':
    # tensor.masked_fill演示
    # input=Variable(torch.randn(2,2))
    # mask=Variable(torch.zeros(2,2))
    # print(input.masked_fill(mask==0,-1e9))
    """
    tensor([[-1.0000e+09, -1.0000e+09],
        [-1.0000e+09, -1.0000e+09]])"""

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

    # # 矩阵乘法
    # # a和b除了最后两个维度能够不一致，其余维度要相同(好比上面代码第一维和第二维分别都是1, 2)
    # # a和b最后两维的维度要符合矩阵乘法的要求（好比a的(3, 4)能和b的(4, 6)进行矩阵乘法）
    # x = Variable(torch.LongTensor([[[1,0],[0, 2]],
    #                                 [[2,1],[1 ,4]],
    #                                [[2,1],[1 ,4]]]))
    # print(x,x.shape)
    # x2 = Variable(torch.LongTensor([[[1], [2]]]))
    # print(x2,x2.shape)
    # y=torch.matmul(x,x2)
    # print(y,y.shape)

    emb = Embeddings(d_model, vocab)
    emb_res = emb(x)
    # print(embRes, embRes.shape)  # torch.Size([2, 4, 3]
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_res = pe(emb_res)
    print("pe_res", pe_res, pe_res.shape)

    # 注意力
    query = key = value = pe_res
    # attn,p_attn=attention(query,key,value)
    # print("attn",attn,attn.shape)
    # print("p_attn",p_attn,p_attn.shape)
    #
    # # 令mask为一个2x4x4的零张量
    # mask=Variable(torch.zeros(2,4,4))
    # attn, p_attn = attention(query, key, value,mask=mask)
    # print("attn", attn, attn.shape)
    # print("p_attn", p_attn, p_attn.shape)

    # multi head
    head = 8
    # mask = Variable(torch.zeros(8, 4, 4))
    # mha = MultiHeadAttention(head, d_model, dropout)
    # mha_res = mha(query, key, value, mask)
    # print("mha_res", mha_res, mha_res.shape)

    # FeedForward
    # d_ff = 64
    # ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # ff_res = ff(mha_res)
    # print(ff_res)

    # NormLayer
    # ln = NormLayer(d_model)
    # ln_res = ln(ff_res)
    # print(ln_res, ln_res.shape)

    # EncoderLayer
    # el=EncoderLayer(d_model,mha,ff,dropout)
    # el_res=el(pe_res,mask)
    # print(el_res,el_res.shape)

    # Encoder
    c = copy.deepcopy
    d_ff=64
    mha = MultiHeadAttention(head, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    el = EncoderLayer(d_model, c(mha), c(ff), dropout)
    mask = Variable(torch.zeros(8, 4, 4))
    en=Encoder(el,8)
    en_res=en(pe_res,mask)
    print(en_res,en_res.shape)