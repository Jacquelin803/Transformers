import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel


class Config(object):

    def __init__(self, dataset):
        self.model_name = 'BertRNN'

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

        # RNN隐层size
        self.rnn_hidden = 256
        # rnn数量
        self.num_layers = 2
        # dropout
        self.dropout = 0.5


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # lstm
        #         input_size: The number of expected features in the input `x`  ->768
        #         hidden_size: The number of features in the hidden state `h`   ->256
        #         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
        #             would mean stacking two LSTMs together to form a `stacked LSTM`,
        #             with the second LSTM taking in outputs of the first LSTM and
        #             computing the final results. Default: 1
        #         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
        #             Default: ``True``
        #         batch_first: If ``True``, then the input and output tensors are provided
        #             as (batch, seq, feature). Default: ``False``
        #         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
        #             LSTM layer except the last layer, with dropout probability equal to
        #             :attr:`dropout`. Default: 0
        #         bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.rnn_hidden, num_layers=config.num_layers,
                            batch_first=True, dropout=config.dropout,
                            bidirectional=True)  # bidirectional=True出来的会是256*2的feature

        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask=mask,
                                          output_all_encoded_layers=False)  # encoder_out.shape [128,32,768]
        out, _ = self.lstm(encoder_out)  # lstm: Outputs: output, (h_n, c_n)   # out.shape [128,32,512]
        out = self.dropout(out)
        out = out[:, -1, :]  # [128,512]
        out = self.fc(out)
        return out
