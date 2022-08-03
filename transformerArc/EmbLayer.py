"""
文本嵌入层的作用
无论是源文本嵌入还是目标文本嵌入，
都是为了将文本中词汇的数字表示转变为向量表示, 希望在这样的高维空间捕捉词汇间的关系.
"""

# 导入必备的工具包
import torch

# 预定义的网络层torch.nn, 工具开发者已经帮助我们开发好的一些常用层,
# 比如，卷积层, lstm层, embedding层等, 不需要我们再重新造轮子.
import torch.nn as nn

# 数学计算工具包
import math

# torch中变量封装函数Variable.
from torch.autograd import Variable


# 定义Embeddings类来实现文本嵌入层，这里s说明代表两个一模一样的嵌入层, 他们共享参数.
# 该类继承nn.Module, 这样就有标准层的一些功能, 这里我们也可以理解为一种模式, 我们自己实现的所有层都会这样去写.

class Embeddings(nn.Module):
    """类的初始化函数, 有两个参数, d_model: 指词嵌入的维度, vocab: 指词表的大小."""

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()

        # 之后就是调用nn中的预定义层Embedding, 获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)

        # 最后就是将d_model传入类中
        self.d_model = d_model

    """可以将其理解为该层的前向传播逻辑，所有层中都会有此函数
       当传给该类的实例化对象参数时, 自动调用该类函数
       参数x: 因为Embedding层是首层, 所以代表输入给模型的文本通过词汇映射后的张量"""

    def forward(self, x):
        # 将x传给self.lut并与根号下self.d_model相乘作为结果返回
        return self.lut(x) * math.sqrt(self.d_model)


if __name__ == '__main__':
    print(torch.__version__)
    # nn.Embedding演示
    # nn.Embedding(vocab_size 词表的大小，embedding_size 输出词向量的维度，padding_idx 初始化为0）
    # embdding=nn.Embedding(10,3)
    # print(type(embdding),embdding)
    # input=torch.LongTensor([[1,2,3,4],[6,3,2,1]])  # 2*4
    # print(input)
    # print(embdding(input))
    """
    tensor([[1, 2, 3, 4],
        [6, 3, 2, 1]])
    tensor([[[-0.3528, -0.1028, -0.6341],
         [ 0.6336, -0.4085,  0.4172],
         [ 2.5672,  0.6208, -1.3051],
         [ 0.2318,  0.7724, -2.2401]],

        [[-0.6832, -1.4175, -1.2105],
         [ 2.5672,  0.6208, -1.3051],
         [ 0.6336, -0.4085,  0.4172],
         [-0.3528, -0.1028, -0.6341]]], grad_fn=<EmbeddingBackward>)
    将一个单词转化为idx,idx为一个数字，embedding过程使用三个维度的信息代表这个数字也即这个单词
    [-0.3528, -0.1028, -0.6341]在这个语境下代表的就是1     
         """
    # embdding2=nn.Embedding(10,3,padding_idx=0)
    # print(embdding2(input))

    # 实例化Embeddings
    # 词嵌入维度是512维
    d_model = 512
    # 词表大小是1000
    vocab = 1000
    # 输入x是一个使用Variable封装的长整型张量, 形状是2 x 4
    x = Variable(torch.LongTensor([[100, 1, 2, 3], [34, 2, 3, 67]]))
    emb = Embeddings(d_model, vocab)
    embRes = emb(x)
    print(type(embRes), embRes.shape, embRes)
    # torch.Size([2, 4, 512]) 100这个数字有一个512维的向量表示
