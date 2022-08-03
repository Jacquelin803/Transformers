
"""
copy任务介绍:
任务描述: 针对数字序列进行学习, 学习的最终目标是使输出与输入的序列相同.
        如输入[1, 5, 8, 9, 3], 输出也是[1, 5, 8, 9, 3].
任务意义: copy任务在模型基础测试中具有重要意义，因为copy操作对于模型来讲是一条明显规律,
        因此模型能否在短时间内，小数据集中学会它，可以帮助我们断定模型所有过程是否正常，是否已具备基本学习能力.

"""
import torch

# 导入工具包Batch, 它能够对原始样本数据生成对应批次的掩码张量
from transformerArc import MakeModel
from transformerArc.pyitcast.TransformerUtils import Batch
import numpy as np
from torch.autograd import Variable



if __name__ == '__main__':

    # 将生成0-10的整数
    V = 11

    # 每次喂给模型20个数据进行参数更新
    batch = 20

    # 连续喂30次完成全部数据的遍历, 也就是1轮
    num_batch = 30

    # step 01 构建数据集生成器
    """该函数用于随机生成copy任务的数据, 它的三个输入参数是V: 随机生成数字的最大值+1,
       batch: 每次输送给模型更新一次参数的数据量, num_batch: 一共输送num_batch次完成一轮
    """
    def data_generator(V, batch, num_batch):

        # 使用for循环遍历nbatches
        for i in range(num_batch):
            # 在循环中使用np的random.randint方法随机生成[1, V)的整数,
            # 分布在(batch, 10)形状的矩阵中, 然后再把numpy形式转换称torch中的tensor.
            data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))

            # 接着使数据矩阵中的第一列数字都为1, 这一列也就成为了起始标志列,
            # 当解码器进行第一次解码的时候, 会使用起始标志列作为输入.
            data[:, 0] = 1

            # 因为是copy任务, 所有source与target是完全相同的, 且数据样本作用变量不需要求梯度
            # 因此requires_grad设置为False
            source = Variable(data, requires_grad=False)
            target = Variable(data, requires_grad=False)

            # 使用Batch对source和target进行对应批次的掩码张量生成, 最后使用yield返回
            yield Batch(source, target)
    # res=data_generator(V,batch,num_batch)
    # print(res)

    # step 02 获得Transformer模型及其优化器和损失函数
    # 导入优化器工具包get_std_opt, 该工具用于获得标准的针对Transformer模型的优化器
    # 该标准优化器基于Adam优化器, 使其对序列到序列的任务更有效.
    from transformerArc.pyitcast.TransformerUtils import get_std_opt

    # 导入标签平滑工具包, 该工具用于标签平滑, 标签平滑的作用就是小幅度的改变原有标签值的值域
    # 因为在理论上即使是人工的标注数据也可能并非完全正确, 会受到一些外界因素的影响而产生一些微小的偏差
    # 因此使用标签平滑来弥补这种偏差, 减少模型对某一条规律的绝对认知, 以防止过拟合. 通过下面示例了解更多.
    from transformerArc.pyitcast.TransformerUtils import LabelSmoothing

    # 导入损失计算工具包, 该工具能够使用标签平滑后的结果进行损失的计算,
    # 损失的计算方法可以认为是交叉熵损失函数.
    from transformerArc.pyitcast.TransformerUtils import SimpleLossCompute

    # 使用make_model获得model
    model=MakeModel.make_model(V,V,N=2)
    # 使用get_std_opt获得模型优化器
    model_optimizer=get_std_opt(model)
    # 使用LabelSmoothing获得标签平滑对象
    criterion=LabelSmoothing(size=V,padding_idx=0,smoothing=0.0)
    # 使用SimpleLossCompute获得利用标签平滑结果的损失计算方法
    loss=SimpleLossCompute(model.generator,criterion,model_optimizer)

    # step 03  运行模型进行训练和评估
    # 导入模型单轮训练工具包run_epoch, 该工具将对模型使用给定的损失函数计算方法进行单轮参数更新.
    # 并打印每轮参数更新的损失结果.
    from transformerArc.pyitcast.TransformerUtils import run_epoch
    # 导入贪婪解码工具包greedy_decode, 该工具将对最终结进行贪婪解码
    # 贪婪解码的方式是每次预测都选择概率最大的结果作为输出,
    # 它不一定能获得全局最优性, 但却拥有最高的执行效率.
    from transformerArc.pyitcast.TransformerUtils import greedy_decode
    def run(model,loss,epochs=10):
        """模型训练函数, 共有三个参数, model代表将要进行训练的模型
               loss代表使用的损失计算方法, epochs代表模型训练的轮数"""
        for epoch in range(epochs):
            # 模型使用训练模式, 所有参数将被更新
            model.train()
            # 训练时, batch_size是20
            run_epoch(data_generator(V,8,20),model,loss)

            # 模型使用评估模式, 参数将不会变化
            model.eval()
            # 评估时, batch_size是5
            run_epoch(data_generator(V, 8, 5), model, loss)

            # 模型进入测试模式
        model.eval()

        # 假定的输入张量
        source = Variable(torch.LongTensor([[1, 3, 2, 5, 4, 6, 7, 8, 9, 10]]))

        # 定义源数据掩码张量, 因为元素都是1, 在我们这里1代表不遮掩
        # 因此相当于对源数据没有任何遮掩.
        source_mask = Variable(torch.ones(1, 1, 10))

        # 最后将model, src, src_mask, 解码的最大长度限制max_len, 默认为10
        # 以及起始标志数字, 默认为1, 我们这里使用的也是1
        result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
        print(result)

    run(model,loss)
    # 保存模型
    path = './CopyTestModel'
    torch.save(model.state_dict(), path)





