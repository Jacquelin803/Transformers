
import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F

"""
关于数据格式与文本对应关系解释：
卷积吃进去需要[batch_size,channel,h,w],batch_size为128，一次batch里有128个句子，
每个句子的维度[1，32,768]，h就是32，也就是一个句子最多有32个字，多余的不要，少的做补，每个字的特征有768个维度
"""
class Config(object):

    def __init__(self, dataset):
        self.model_name = 'BertCNN'

        self.train_path = dataset + '/data/t80.txt'
        self.test_path = dataset + '/data/t80.txt'
        self.dev_path = dataset + '/data/t80.txt'
        self.datasetpkl = dataset + '/data/dataset.pkl'

        # 类别
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        # 模型训练结果
        self.save_path = dataset + '/ModelResPath/' + self.model_name + '.ckpt'
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 若超过1000bacth效果还没有提升，提前结束训练
        self.require_improvement = 1000

        # 类别数
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 3
        # batch_size
        self.batch_size = 128
        # 每句话处理的长度(短填，长切）
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # bert预训练模型位置
        self.bert_path = 'bert_pretrain'
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert隐层
        self.hidden_size = 768

        # 卷积核尺寸
        self.filter_sizes = (2, 3, 4)
        # 卷积核数量,每一种大小的卷积核各有256个
        self.num_filters = 256
        # dropout
        self.dropout = 0.5


class Model(nn.Module):

    def __init__(self,config):
        super(Model, self).__init__()
        self.bert=BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad=True

        # convolution
            # in_channels (int): Number of channels in the input image
            # out_channels (int): Number of channels produced by the convolution
            # kernel_size (int or tuple): Size of the convolving kernel  ->(h,w)
        self.convs=nn.ModuleList(
            [nn.Conv2d(in_channels=1,out_channels=config.num_filters,     # 文本为1通道，每种大小的filter使用了256个filters，故而输出通道为256,每个通道可以吃到不同信息
                       kernel_size=(k,config.hidden_size)) for k in config.filter_sizes]  # 文本卷积时，只对词上下滑动，不进行左右滑动，因为每一个横行为一个词的信息，左右滑动会使词的信息缺失,上下滑动的这个k,也就是n_gram
        )

        self.dropout=nn.Dropout(config.dropout)
        self.fc=nn.Linear(config.num_filters*len(config.filter_sizes),config.num_classes)  # 这个fc层针对的是每个样本数据的进出size，进来的是因为之前进行了torch.cat，对不同size的kernel卷积后的数据合并，也即256*3

    def conv_and_pool(self,x,conv):
        # conv  (这个函数里的注释针对kernel_size=2做分析)
        x=conv(x)  # x.shape [128,256,(32-2)/stride + 1=31,1] , [2,768]的kernel对[32,768]的数据进行文本卷积，得到的是[31,1]
        x=F.relu(x)  # relu不改变维度[128,256,31,1]
        x=x.squeeze(3)  # [128,256,31]  原本最后一维每个里就一个数据
        size=x.size(2)  # size=31
        x=F.max_pool1d(x,size)  # [128,256,1]  在横行代表的词向量里取最大的值
        x=x.squeeze(2)  # [128,256] 取最大值后就只剩一个数据啦
        return x

    def forward(self,x):
        # x [ids, seq_len, mask]
        context=x[0]  # 对应输入的句子 shape[128,32]
        mask=x[2]   # 对padding部分进行mask shape[128,32]

        # bert_pretrained model
        encoder_out,pooled=self.bert(context,attention_mask=mask,output_all_encoded_layers=False)  # pooled shape [128,768]

        # convolution
        out=encoder_out.unsqueeze(1)  # encoder_out[128,32,768],out[128,1,32,768]  卷积层接的数据维度为四维:[该batch内样本数,channel通道，高度，宽度]，所以需要对原三维数据在通道上扩充维度
        # out2=encoder_out.unsqueeze(0)  # out2[1,128,32,768]
        # print("CNN")
        out=torch.cat([self.conv_and_pool(out,conv) for conv in self.convs],1)  # out.shape [128,768] 每一种不同kernel_size都会卷积得到一个[128,256]，256个通道里的信息又是不一样的,然后将它们合并,其实类似于每个样本不同方面信息的横向拼接

        # dropout/fc
        out=self.dropout(out)
        out=self.fc(out)  # [128,10]

        return out







