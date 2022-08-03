import torch
import time
import argparse

from importlib import import_module
import numpy as np
import utils
import train

parser = argparse.ArgumentParser(description='BertClassifier')
# parser.add_argument('--model', type=str, default='BertFc', help='choose a model')
# parser.add_argument('--model', type=str, default='BertCNN', help='choose a model')
# parser.add_argument('--model', type=str, default='BertRNN', help='choose a model')
# parser.add_argument('--model', type=str, default='BertDPCNN', help='choose a model')
# parser.add_argument('--model', type=str, default='ERNIE', help='choose a model')
parser.add_argument('--model', type=str, default='ERNIEDPCNN', help='choose a model')
args = parser.parse_args()

if __name__ == '__main__':
    print(torch.__version__)
    # 数据集地址
    dataset = 'THUCNews'
    model_name = args.model
    x = import_module(
        'models.' + model_name)  # <module 'models.BertFc' from '/home/hadoop/PycharmProjects/BertClassifier/models/BertFc.py'>
    config = x.Config(dataset)
    print(config.model_name)
    # print(config)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic=True   # 保证每次运行结果一样


    # 加载数据集
    start_time=time.time()
    print('加载数据集')
    train_data,dev_data,test_data=utils.build_dataset(config)
    train_iter=utils.build_iterator(train_data,config)
    test_iter=utils.build_iterator(test_data,config)
    dev_iter=utils.build_iterator(dev_data,config)

    time_dif=utils.get_time_dif(start_time)
    print("模型开始之前，准备数据时间：", time_dif)
    # for i,(train,label) in enumerate(dev_iter):
    #     if (i%10==0):
    #         print(i,label)  # dev contains 10000 items,10000/128=78.125,residue=True,79 batches,the batch 79st only has 16 items

    # 模型训练，评估与测试
    model=x.Model(config).to(config.device)

    train.train(config,model,train_iter,dev_iter,test_iter)
