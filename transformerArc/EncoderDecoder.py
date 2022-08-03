
"""
模型构建
    1.编码器-解码器结构
    (2.Transformer模型)
"""

import numpy as np
import math
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from transformerArc.Decoder import Decoder
from transformerArc.DecoderLayer import DecoderLayer
from transformerArc.Encoder import Encoder
from transformerArc.EncoderLayer import EncoderLayer
from transformerArc.Generator import Generator
from transformerArc.MultiHeadAttention import MultiHeadAttention
from transformerArc.PositionwiseFeedForward import PositionwiseFeedForward

"""
编码器-解码器结构
"""
class EncoderDecoder(nn.Module):
    """初始化函数中有5个参数, 分别是编码器对象, 解码器对象,
       源数据嵌入函数, 目标数据嵌入函数,  以及输出部分的类别生成器对象
    """
    def __init__(self,encoder,decoder,source_embed,target_embed,generator):
        super(EncoderDecoder, self).__init__()

        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=source_embed
        self.target_embed=target_embed
        self.generator=generator

    """编码函数, 以source和source_mask为参数"""
    def encode(self,source,source_mask):
        # 使用src_embed对source做处理, 然后和source_mask一起传给self.encoder
        return self.encoder(self.src_embed(source),source_mask)

    """解码函数, 以memory即编码器的输出, source_mask, target, target_mask为参数"""
    def decode(self,memory,source_mask,target,target_mask):
        # 使用tgt_embed对target做处理, 然后和source_mask, target_mask, memory一起传给self.decoder
        return self.decoder(self.target_embed(target),memory,source_mask,target_mask)

    """在forward函数中，有四个参数, source代表源数据, target代表目标数据, 
       source_mask和target_mask代表对应的掩码张量"""
    def forward(self,source,target,source_mask,target_mask):
        return self.decode(self.encode(source,source_mask),source_mask,target,target_mask)



if __name__ == '__main__':

    vocab_size=1000
    size = 512
    d_model = 512
    head = 8
    d_ff = 64
    dropout = 0.2
    c = copy.deepcopy
    source_embed=nn.Embedding(vocab_size,d_model)
    target_embed=nn.Embedding(vocab_size,d_model)
    attn = MultiHeadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    de_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
    en_layer = EncoderLayer(size, c(attn), c(ff), dropout)
    N = 8


    de = Decoder(de_layer, N)
    en = Encoder(en_layer, N)
    gen = Generator(d_model, vocab_size)

    # 假设源数据与目标数据相同, 实际中并不相同
    source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

    # 假设src_mask与tgt_mask相同，实际中并不相同
    source_mask = target_mask = Variable(torch.zeros(8, 4, 4))

    ed=EncoderDecoder(en,de,source_embed,target_embed,gen)
    ed_res=ed(source,target,source_mask,target_mask)
    print(ed_res,ed_res.shape)
