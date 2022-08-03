
"""
多头注意力机制:
    每个头开始从词义层面分割输出的张量，也就是每个头都想获得一组Q，K，V进行注意力机制的计算，
    但是句子中的每个词的表示只获得一部分，
    也就是只分割了最后一维的词嵌入向量. 这就是所谓的多头.
    将每个头的获得的输入送到注意力机制中, 就形成了多头注意力机制
多头注意力机制的作用:
    这种结构设计能让每个注意力机制去优化每个词汇的不同特征部分，从而均衡同一种注意力机制可能产生的偏差，
    让词义拥有来自更多元的表达，从而提升模型效果
"""
import copy
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class MultiHeadAttention(nn.Module):

    """在类的初始化时, 会传入三个参数，head代表头数，embedding_dim代表词嵌入的维度，
       dropout代表进行dropout操作时置0比率，默认是0.1."""

    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除，
        # 这是因为我们之后要给每个头分配等量的词特征.也就是embedding_dim/head个.
        assert embedding_dim % head == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k= embedding_dim // head

        self.head=head

        # 然后获得线性层对象，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim x embedding_dim，然后使用clones函数克隆四个，
        # 为什么是四个呢，这是因为在多头注意力中，Q，K，V各需要一个，最后拼接的矩阵还需要一个，因此一共是四个.
        self.linears=self.clones(nn.Linear(embedding_dim,embedding_dim),4)

        self.attn=None

        self.dropout=nn.Dropout(p=dropout)

    # 首先需要定义克隆函数, 因为在多头注意力机制的实现中, 用到多个结构相同的线性层.
    # 我们将使用clone函数将他们一同初始化在一个网络层列表对象中. 之后的结构中也会用到该函数.
    """用于生成相同网络层的克隆函数, 它的参数module表示要克隆的目标网络层, N代表需要克隆的数量"""

    def clones(self,module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    """生成向后遮掩的掩码张量, 参数size是掩码张量最后两个维度的大小, 它的最后两维形成一个方阵"""
    def subsequentMask(self,size):
        # 在函数中, 首先定义掩码张量的形状
        attnShape=(1,size,size)
        # 然后使用np.ones方法向这个形状中添加1元素,形成上三角阵, 最后为了节约空间,
        # 再使其中的数据类型变为无符号8位整形unit8
        subsequent_mask=np.triu(np.ones(attnShape),k=1).astype('uint8')
        # 最后将numpy类型转化为torch中的tensor, 内部做一个1 - 的操作,
        # 在这个其实是做了一个三角阵的反转, subsequent_mask中的每个元素都会被1减,
        # 如果是0, subsequent_mask中的该位置由0变成1
        # 如果是1, subsequent_mask中的该位置由1变成0
        return torch.from_numpy(1-subsequent_mask)

    """注意力计算规则"""
    def attention(self,query,key,value,mask=None,dropout=None):
        # 在函数中, 首先取query的最后一维的大小, 一般情况下就等同于我们的词嵌入维度, 命名为d_k
        d_k = query.size(-1)
        # 按照注意力公式, 将query与key的转置相乘, 这里面key是将最后两个维度进行转置, 再除以缩放系数根号下d_k, 这种计算方法也称为缩放点积注意力计算.
        # 得到注意力得分张量scores

        scores=torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)  # 最后两个维度做矩阵乘积
        # print("scores before masked: ", scores,scores.shape)
        # 接着判断是否使用掩码张量
        if mask is not None:
        # 使用tensor的masked_fill方法, 将掩码张量和scores张量每个位置一一比较, 如果掩码张量处为0
        # 则对应的scores张量用-1e9这个值来替换
            scores=scores.masked_fill(mask==0,-1e9)
        # print("scores after masked: ",scores,scores.shape)
        # 对scores的最后一维进行softmax操作, 使用F.softmax方法, 第一个参数是softmax对象, 第二个是目标维度.
        # 这样获得最终的注意力张量
        p_attn=F.softmax(scores,dim=-1)


        # 判断是否使用dropout进行随机置0
        if dropout is not None:
            p_attn=dropout(p_attn)

        # print("value : ",value,value.shape)
        # 最后, 根据公式将p_attn与value张量相乘获得最终的query注意力表示, 同时返回注意力张量
        return torch.matmul(p_attn,value),p_attn

    """前向逻辑函数, 它的输入参数有四个，前三个就是注意力机制需要的Q, K, V，
       最后一个是注意力机制中可能需要的mask掩码张量，默认是None. """
    def forward(self,query,key,value,mask=None):

        if mask is not None:
            mask=mask.unsqueeze(0)

        # 接着，我们获得一个batch_size的变量，他是query尺寸的第1个数字，代表有多少条样本.
        batch_size=query.size(0)

        # 之后就进入多头处理环节
        # 首先利用zip将输入QKV与三个线性层组到一起，然后使用for循环，将输入QKV分别传到线性层中，
        # 做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多加了一个维度h，代表头数，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值.然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，
        # 从attention函数中可以看到，利用的是原始输入的倒数第一和第二维.这样我们就得到了每个头的输入.
        query,key,value=[model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2) for model,x in zip(self.linears,(query,key,value))]
        # print("query.shape before attention",query.shape)
        # 得到每个头的输入后，接下来就是将他们传入到attention中，
        # 这里直接调用我们之前实现的attention函数.同时也将mask和dropout传入其中.
        x,self.attn=self.attention(query,key,value,mask,dropout=self.dropout)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，
        # 因此这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用contiguous方法，
        # 这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用，
        # 所以，下一步就是使用view重塑形状，变成和输入形状相同.
        # print("x shape",x.shape)
        x=x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)
        # print("x shape",x.shape)  # 这一步，x维度回到最初的

        # 最后使用线性层列表中的最后一个线性层对输入进行线性变换得到最终的多头注意力结构的输出.
        return self.linears[-1](x)

