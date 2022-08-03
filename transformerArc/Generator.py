
"""
输出部分包含:
    线性层 : 通过对上一步的线性变化得到指定维度的输出, 也就是转换维度的作用
    softmax层 : 使最后一维的向量中的数字缩放到0-1的概率值域内, 并满足他们的和为1
"""

import torch.nn.functional as F
import torch.nn as nn
import torch

"""将线性层和softmax计算层一起实现, 因为二者的共同目标是生成最后的结构"""
class Generator(nn.Module):
    """初始化函数的输入参数有两个, d_model代表词嵌入维度, vocab_size代表词表大小."""
    def __init__(self,d_model,vocab_size):
        super(Generator, self).__init__()

        # 首先就是使用nn中的预定义线性层进行实例化, 得到一个对象self.project等待使用,
        # 这个线性层的参数有两个, 就是初始化函数传进来的两个参数: d_model, vocab_size
        self.project=nn.Linear(d_model,vocab_size)

    """前向逻辑函数中输入是上一层的输出张量x"""
    def forward(self,x):
        # 在函数中, 首先使用上一步得到的self.project对x进行线性变化,
        # 然后使用F中已经实现的log_softmax进行的softmax处理.
        # 在这里之所以使用log_softmax是因为和我们这个pytorch版本的损失函数实现有关, 在其他版本中将修复.
        # log_softmax就是对softmax的结果又取了对数, 因为对数函数是单调递增函数,
        # 因此对最终我们取最大的概率值没有影响. 最后返回结果即可.
        return F.log_softmax(self.project(x),dim=-1)


if __name__ == '__main__':
    model=nn.Linear(20,30)
    input=torch.randn(128,20)
    out=model(input)
    print(input,input.shape,out,out.shape)



