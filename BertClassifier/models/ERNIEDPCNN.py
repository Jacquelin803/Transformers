




import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F


class Config(object):

    def __init__(self, dataset):
        self.model_name = 'ERNIEDPCNN'
        print("ERNIEDPCNN")
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
        self.bert_path = 'ERNIEPretrain'
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert隐层
        self.hidden_size = 768

        # 卷积核数量
        self.num_filters = 250
        # dropout
        self.dropout = 0.5


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # conv_region,文本为1通道，filter使用了250个filters，故而输出通道为250,每个通道可以吃到不同信息
        self.conv_region = nn.Conv2d(in_channels=1, out_channels=config.num_filters,
                                     kernel_size=(3, config.hidden_size))

        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1))

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

        self.padd1 = nn.ZeroPad2d((0, 0, 1, 1))  # 对top,bottom补0
        self.padd2 = nn.ZeroPad2d((0, 0, 0, 1))  # 比如左边添加1dim、右边添加2dim、上边添加3dim、下边添加4dim，即指定paddin参数为（1，2，3，4）

        self.relu = nn.ReLU()

        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self,x):
        context=x[0]
        mask=x[2]

        # bert_pretrained model
        encoder_out, pooled = self.bert(context, attention_mask=mask,
                                        output_all_encoded_layers=False)  # pooled shape [128,768]

        # deep pyramid CNN
        # conv_region
        out=encoder_out.unsqueeze(1)  # encoder_out[128,32,768],out[128,1,32,768]
        out=self.conv_region(out)  # [batch-size,250,seq_len - kernel_size_h + 1,1] -> [128,250,32-3+1=30,1]
        # first conv
        out=self.padd1(out)  # [128,250,32,1] 原本是[30,1],最前补[0.0],最后补[0.0],变成[32,1]
        out=self.relu(out)
        out=self.conv(out)  # [128,250,32-3+1,1]
        # second conv
        out=self.padd1(out)  # [128,250,32,1]
        out=self.relu(out)
        out=self.conv(out)  # [128,250,32-3+1,1]
        # block
        while out.size()[2]>2:
            out=self._block(out)
        # ->[128,250,1,1]
        out=out.squeeze()  # [128,250]
        out=self.fc(out)

        return out


    def _block(self,x):
        # 注释只针对第一轮
        x=self.padd2(x)  # [128,250,30,1]->[128,250,31,1]

        px=self.max_pool(x)  # [128,250,(31-3)/2 + 1=15,1]  roll2: [128,250,7,1]

        x=self.padd1(px)  # [128,250,17,1]
        x=self.relu(x)
        x=self.conv(x)  # [128,250,(17-3)+1=15,1]

        x = self.padd1(x)  # [128,250,17,1]
        x = self.relu(x)
        x = self.conv(x)  # [128,250,17-3+1=15,1]

        x=x+px

        return x





