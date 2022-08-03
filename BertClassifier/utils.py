
"""
数据处理
"""

import torch
import time
from tqdm import tqdm
from datetime import timedelta
import pickle as pkl
import os

PAD, CLS = '[PAD]', '[CLS]'

"""
返回结果 4个: list ids, lable, ids_len, mask
"""


def load_dataset(file_path, config):
    contents = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):    # 显示进度tqdm
            line = line.strip()
            if not line:
                continue
            content, lable = line.split('\t')
            print("content label:",content,lable)
            token = config.tokenizer.tokenize(content)  # 文本的字
            seq_len = len(token)  # 句子词的实际长度
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            print(token_ids)
            pad_size = config.pad_size  # 每句话处理的长度(短填，长切）

            if pad_size:
                if seq_len < pad_size:  # 短填
                    mask = [1] * seq_len + [0] * (pad_size - seq_len)
                    token_ids = token_ids + [0] * (pad_size - seq_len)
                else:  # 已经超过模型约定的句子长度了（pad_size=32）,长切
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(lable), seq_len, mask))
    return contents


"""
返回值 train, dev ,test
"""


def build_dataset(config):

    """
            with open(filename, 'rb') as input_file:
                try:
                        return pickle.load(input_file)
                except EOFError:
                        return None
    :param config:
    :return:
    """
    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        print("read pkl dataset")
        train = dataset['train']
        test = dataset['test']
        dev = dataset['dev']
    else:
        print("read txt",config.train_path)
        train = load_dataset(config.train_path, config)
        test = load_dataset(config.test_path, config)
        dev = load_dataset(config.dev_path, config)
        dataset = {}
        dataset['train'] = train
        dataset['test'] = test
        dataset['dev'] = dev
        pkl.dump(dataset, open(config.datasetpkl, 'wb',),protocol=4)
    return train, dev, test


class DatasetIterator(object):

    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_batches = len(dataset) // batch_size

        self.residue = False  # 记录batch数量是否为整数
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0  # 哪一个batch
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([item[0] for item in datas]).to(self.device)
        y = torch.LongTensor([item[1] for item in datas]).to(self.device)
        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)

        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            # 最后一个batch
            batches = self.dataset[self.index * self.batch_size:len(self.dataset)]
            self.index+=1
            batches=self._to_tensor(batches)
            return batches
        elif self.index>self.n_batches:
            # over
            self.index=0
            raise StopIteration
        else:
            # 中间batches
            batches=self.dataset[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index+=1
            batches=self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches+1
        else:
            return self.n_batches


def build_iterator(dataset,config):
    iter=DatasetIterator(dataset,config.batch_size,config.device)
    return iter


"""
获取已经使用的时间
"""
def get_time_dif(start_time):
    end_time=time.time()
    time_dif=end_time-start_time
    return timedelta(seconds=int(round(time_dif)))


