"""
解码器的作用:
根据编码器的结果以及上一次预测的结果, 对下一次可能出现的'值'进行特征表示
"""

import torch.nn as nn
import torch
import copy

from torch.autograd import Variable

from transformerArc.DecoderLayer import DecoderLayer
from transformerArc.EmbLayer import Embeddings
from transformerArc.Generator import Generator
from transformerArc.MultiHeadAttention import MultiHeadAttention
from transformerArc.NormLayer import NormLayer
from transformerArc.PosEncLayer import PositionalEncoding
from transformerArc.PositionwiseFeedForward import PositionwiseFeedForward


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Decoder(nn.Module):
    """初始化函数的参数有两个，第一个就是解码器层layer，第二个是解码器层的个数N."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = NormLayer(layer.d_model)

    """forward函数中的参数有4个，x代表目标数据的嵌入表示，memory是编码器层的输出，
       source_mask, target_mask代表源数据和目标数据的掩码张量"""

    def forward(self, x, memory, source_mask, target_mask):
        # 然后就是对每个层进行循环，当然这个循环就是变量x通过每一个层的处理，
        # 得出最后的结果，再进行一次规范化返回即可.
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


if __name__ == '__main__':
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
    emb_res = emb(x)
    # print(embRes, embRes.shape)  # torch.Size([2, 4, 3]
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_res = pe(emb_res)
    print("pe_res", pe_res, pe_res.shape)

    # Decoder
    head=8
    c = copy.deepcopy
    d_ff = 64
    mha = MultiHeadAttention(head, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    dl = DecoderLayer(d_model, c(mha), c(mha), c(ff), dropout)
    source_mask=target_mask=mask = Variable(torch.zeros(8, 4, 4))
    de=Decoder(dl,8)
    memory=pe_res
    x=pe_res
    de_res=de(x,memory,source_mask,target_mask)
    print(de_res,de_res.shape)

    # output
    ge=Generator(d_model,vocab)
    ge_res=ge(de_res)
    print(ge_res,ge_res.shape)