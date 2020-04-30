# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

'''
    bert模型的返回结果
    1. 参数output_all_encoded_layers=True，12层Transformer的结果全返回了，
    存在第一个列表中，每个encoder_output的大小为[batch_size, sequence_length, hidden_size];
    2. pool_out大小为[batch_size, hidden_size]，取了最后一层Transformer的输出结果的第一个单词[cls]的hidden states，
    其已经蕴含了整个input句子的信息了。
    3. 如果你用不上所有encoder层的输出，output_all_encoded_layers参数设置为Fasle，那么result中的第一个元素就不是列表了，
    只是encoder_11_output，大小为[batch_size, sequence_length, hidden_size]的张量，可以看作bert对于这句话的表示。
    result = (
    [encoder_0_output, encoder_1_output, ..., encoder_11_output], 
    pool_output
    )
'''



class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out



# class Model(nn.Module):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.bert = BertModel.from_pretrained(config.bert_path)
#         for param in self.bert.parameters():  # 调用bert模型参数
#             param.requires_grad = True
#         self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
#         self.final_dense = nn.Linear(config.hidden_size, config.num_classes)
#
#         self.classifer = nn.Linear(config.hidden_size*4, config.num_classes)
#         self.activation = nn.Softmax()
#
#     def forward(self, x):
#         context = x[0]  # 输入的句子
#         mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
#         encoded_layers, pooling = self.bert(context, attention_mask=mask,
#                                       output_all_encoded_layers=True)  # 输出隐藏层向量列表
#
#         '''
#             后三层与pool层进行拼接
#         '''
#         last_dense = torch.cat((pooling, encoded_layers[-1][:, 0], encoded_layers[-2][:, 0], encoded_layers[-3][:, 0]),
#                                dim=1)
#         outputs = self.classifer(last_dense)
#
#
#         return outputs