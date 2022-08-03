
import torch.nn as nn
import torch

from transformerArc.EmbLayer import Embeddings
from transformerArc.Generator import Generator

"""
该函数用来构建模型, 有7个参数，分别是源数据特征(词汇)总数，目标数据特征(词汇)总数，
编码器和解码器堆叠数，词向量映射维度，前馈全连接网络中变换矩阵的维度，
多头注意力结构中的多头数，以及置零比率dropout.
"""
import copy

from transformerArc.Decoder import Decoder
from transformerArc.DecoderLayer import DecoderLayer
from transformerArc.Encoder import Encoder
from transformerArc.EncoderDecoder import EncoderDecoder
from transformerArc.EncoderLayer import EncoderLayer
from transformerArc.MultiHeadAttention import MultiHeadAttention
from transformerArc.PosEncLayer import PositionalEncoding
from transformerArc.PositionwiseFeedForward import PositionwiseFeedForward


def make_model(source_vocab,target_vocab,N=6,d_model=512,d_ff=2048,head=8,dropout=0.1):
    # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝，
    # 来保证他们彼此之间相互独立，不受干扰.
    c = copy.deepcopy

    # multi-head
    attn=MultiHeadAttention(head,d_model)

    # positionalFeedForward
    ff=PositionwiseFeedForward(d_model,d_ff,dropout)

    # positional encoding
    position=PositionalEncoding(d_model,dropout)

    # 根据结构图, 最外层是EncoderDecoder，在EncoderDecoder中，
    # 分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构，
    # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层.
    # 在编码器层中有attention子层以及前馈全连接子层，
    # 在解码器层中有两个attention子层以及前馈全连接层.
    model=EncoderDecoder(
        Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
        Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout),N),
        nn.Sequential(Embeddings(d_model,source_vocab),c(position)),
        nn.Sequential(Embeddings(d_model,target_vocab),c(position)),
        Generator(d_model,target_vocab)
    )

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model






if __name__ == '__main__':
    # nn.init.xavier_uniform演示:
    # w=torch.empty(3,5)
    # res=nn.init.xavier_uniform_(w,gain=nn.init.calculate_gain('relu'))
    # print(res)

    # make model
    source_vocab=11
    target_vocab=11
    N=6
    res=make_model(source_vocab,target_vocab,N)
    print(res)

    """
    /opt/python3.6/bin/python3.6 /home/hadoop/PycharmProjects/PytrochDL/transformerArc/MakeModel.py
EncoderDecoder(
  (encoder): Encoder(
    (layers): ModuleList(
      (0): EncoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (1): EncoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (2): EncoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (3): EncoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (4): EncoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (5): EncoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (norm): NormLayer()
  )
  (decoder): Decoder(
    (layers): ModuleList(
      (0): DecoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (src_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (1): DecoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (src_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (2): DecoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (src_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (3): DecoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (src_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (4): DecoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (src_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (5): DecoderLayer(
        (self_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (src_attn): MultiHeadAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (subLayer): ModuleList(
          (0): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): SubLayerConnection(
            (norm): NormLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (norm): NormLayer()
  )
  (src_embed): Sequential(
    (0): Embeddings(
      (lut): Embedding(11, 512)
    )
    (1): PositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (target_embed): Sequential(
    (0): Embeddings(
      (lut): Embedding(11, 512)
    )
    (1): PositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (generator): Generator(
    (project): Linear(in_features=512, out_features=11, bias=True)
  )
)

Process finished with exit code 0

    """