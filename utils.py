import json
import os
import glob
import string

import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords


# 数据读取
def read_data(file_path):
    """
       params:
           file_path: 文件路径
       return:
           data: 读取的数据列表，每行一条样本

    """
    data = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            # 过长的评论文本会超出io限制报错。 暂时忽略
            try:
                # 字符串预处理
                # 布尔替换成python的习惯
                str_text = line.replace("true", "True")
                str_text = str_text.replace("false", "False")
                # 转成字典形式
                raw_sample = eval(str_text)
                data.append([raw_sample['reviewerID'],
                             raw_sample['asin'],
                             raw_sample['overall'],
                             raw_sample['reviewText']])
            except:
                pass
    return data


def softmax(x):
    """Compute the softmax of vector x.
    """
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


# 加载停用词
stop_words = stopwords.words("english") + list(string.punctuation)
def word_segment(text):
    # word_seg = [i for i in word_tokenize(str(text).lower()) if i not in stop_words]
    # word_seg = text.split(" ")
    word_seg = [i for i in word_tokenize(str(text).lower())]
    return word_seg


def preprocessed(text):
    """ 3文本预处理
    """
    # 分句和词性还原， 目前只实现分句
    return text.split("\.")


if __name__ == "__main__":

    import nltk
    nltk.download('averaged_perceptron_tagger')
    # from nltk.corpus import sentiwordnet as swn
    # breakdown = swn.senti_synset('breakdown.n.03')
    # print(breakdown)