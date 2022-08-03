import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from sklearn import metrics
from pytorch_pretrained_bert.optimization import BertAdam

"""
模型训练
"""


def train(config, model, train_iter, dev_iter, test_iter):


    start_time = time.time()
    # 启动 BatchNormalization 和 dropout
    model.train()
    # 拿到所有mode种的参数
    param_optimizer = list(model.named_parameters())
    # 不需要衰减的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    """
    lr: learning rate
    warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
    t_total: total number of training steps for the learning
        """
    optimizer = BertAdam(params=optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)

    # 记录进行多少batch
    total_batch = 0
    # 记录校验集合最好的loss
    dev_best_loss = float('inf')
    # 记录上次校验集loss下降的batch数
    last_improve = 0
    # 记录是否很久没有效果提升，停止训练
    flag = False

    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))  # 打印进度
        for i, (trains, labels) in enumerate(train_iter):  # train_iter: (x, seq_len, mask), y
            # print("train,label: ",trains,trains[0].shape,labels,labels.shape)
            # 训练得到前向传播结果
            outputs = model(trains)
            # 参数置0
            model.zero_grad()
            # 计算模型前向传播结果值与真实label间的差距
            loss = F.cross_entropy(outputs, labels)
            # 反向传播
            loss.backward(retain_graph=False)
            # 每吃进样本数据要吃一次信息，向前走一步，后面的epoch也是在基于之前信息进行逐步迭代
            optimizer.step()

            if total_batch % 100 == 0:  # 每100次输出在训练集和校验集上的效果
                true = labels.data.cpu()
                # print("outputs.data.cpu()", outputs.data.cpu())
                # print(outputs.data.cpu().shape)
                # print("outputs.data.cpu()[0]",outputs.data.cpu()[0])
                # print("outputs.data.cpu()[1]",outputs.data.cpu()[1])
                # print("outputs.data.cpu()[0]",outputs.data.cpu()[2])
                # print("torch.max(outputs.data, 1)[1]",torch.max(outputs.data, 1)[1],torch.max(outputs.data, 1)[1].shape)
                # print(torch.max(outputs.data, 1))
                # print("outputs.data.cpu()",outputs.data.cpu())
                predict = torch.max(outputs.data, 1)[1].cpu()  # 取出最大概率对应的类别index
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '**'
                    last_improve = total_batch
                else:
                    improve = '--'
                time_dif = utils.get_time_dif(start_time)
                print_msg = 'Iter: {0:>6}, TrainLoss: {1:>5.2}, TrainAcc: {2:>6.2}, ValLoss: {3:>5.2}, ValAcc: {4:>6.2%}, Time: {5} {6}'
                print(print_msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

                model.train()  # 在刚刚经过100倍数评估时被置为eval()，迭代训练时需要重新置为train()

            total_batch = total_batch + 1
            # 在验证集合上loss超过1000batch没有下降，结束训练
            if total_batch - last_improve > config.require_improvement:
                print('在校验数据集合上已经{}steps没有提升了，模型自动停止训练'.format(config.require_improvement + 1))
                flag = True
                break  # current epoch stop

        if flag:
            break  # train stop

    test(config, model, test_iter)


"""
模型评估
"""


def evaluate(config, model, dev_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    label_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels in dev_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total = loss_total + loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            label_all = np.append(label_all, labels)
            predict_all = np.append(predict_all, labels)

    acc = metrics.accuracy_score(label_all, predict_all)
    if test:
        report = metrics.classification_report(label_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(label_all, predict_all)
        return acc, loss_total / len(dev_iter), report, confusion

    return acc, loss_total / len(dev_iter)


"""
模型测试
"""


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score")
    print(test_report)
    print("Confusion Maxtrix")
    print(test_confusion)
    time_dif = utils.get_time_dif(start_time)
    print("使用时间：", time_dif)
