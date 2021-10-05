# -*- coding:utf-8 -*-

import pandas as pd
from gensim.models import word2vec, Word2Vec

from utils import read_data, softmax, word_segment


def get_word2vec_model(is_train, model_path, split_data=None, vector_size=None, min_count=None, window=None):
    """word2vec训练代码
    """
    if is_train:
        model = word2vec.Word2Vec(split_data, vector_size=vector_size, min_count=min_count, window=window)
        model.save(model_path)
    else:
        model = Word2Vec.load(model_path)
    return model


def get_coarse_simtiment_score(text, word2vec_model):
    word_seg = word_segment(text)
    sim_word = []
    sim_word_weight = []
    for e in word2vec_model.wv.most_similar(positive=word_seg, topn=10):
        # print(e[0], e[1])
        sim_word.append(e[0])
        sim_word_weight.append(e[1])

    return sim_word, softmax(sim_word_weight)


if __name__ == "__main__":
    # 数据读取
    data = read_data("./data/raw/All_Beauty_5.json")
    data_df = pd.DataFrame(data)
    data_df.columns = ['reviewerID', 'asin', 'overall', 'reviewText']

    #  step1: LDA
    num_topics = 50
    num_words = 100

    data = data_df["reviewText"].tolist()
    # 英文分词
    print("数据读取完成")
    split_data = []
    for i in data:
        split_data.append(word_segment(i))

    print("数据分词完成")
    # 构建模型

    window_size = 3
    min_count = 1
    vector_size = 200
    model_path = "./output/word2vec.model"
    model = get_word2vec_model(is_train=True,
                               model_path=model_path,
                               split_data=split_data,
                               vector_size=vector_size,
                               min_count=min_count,
                               window=window_size)

    sim_word, sim_word_weight = get_coarse_simtiment_score(data[1], model)
    print(sim_word, sim_word_weight, sum(sim_word_weight))