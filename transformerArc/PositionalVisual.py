"""
绘制词汇向量中特征的分布曲线
效果分析:
每条颜色的曲线代表某一个词汇中的特征在不同位置的含义.->给不同位置加不同的位置信息
保证同一词汇随着所在位置不同它对应位置嵌入向量会发生变化.
正弦波和余弦波的值域范围都是1到-1这又很好的控制了嵌入数值的大小, 有助于梯度的快速计算.
"""
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import numpy as np
from transformerArc.PosEncLayer import PositionalEncoding

if __name__ == '__main__':
    # 创建一张15 x 5大小的画布
    plt.figure(figsize=(15, 5))

    # 实例化PositionalEncoding类得到pe对象, 输入参数是20和0
    pe = PositionalEncoding(20, 0)

    # 然后向pe传入被Variable封装的tensor, 这样pe会直接执行forward函数,
    # 且这个tensor里的数值都是0, 被处理后相当于位置编码张量
    y = pe(Variable(torch.zeros(1, 100, 20)))

    # 然后定义画布的横纵坐标, 横坐标到100的长度, 纵坐标是某一个词汇中的某维特征在不同长度下对应的值
    # 因为总共有20维之多, 我们这里只查看4，5，6，7维的值.
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())

    # 在画布上填写维度提示信息
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()
