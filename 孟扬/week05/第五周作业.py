#!/usr/bin/env python3
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
#基于kmeans结果类内距离的排序
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    sentence_label_avg_distance_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        #kmeans.cluester_centers_ #每个聚类中心
        # print(kmeans.cluster_centers_)
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    # 计算每类的类内平均距离
    for label in range(n_clusters):
        cluster_points = vectors[kmeans.labels_ == label]
        if len(cluster_points) > 0:
            distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[label])
            avg_distance = np.mean(distances)
            sentence_label_avg_distance_dict[label].append(avg_distance)
        else:
            sentence_label_avg_distance_dict[label].append(0)
    # 将类内平均距离升序排序
    sentence_label_avg_distance_dict_desc = dict(sorted(sentence_label_avg_distance_dict.items(), key=lambda x : x[1]))
    for label,avg_dis in sentence_label_avg_distance_dict_desc.items():
        print(f"第{label}类的类内平均距离为{avg_dis}")
        for i in range(min(3,len(sentence_label_dict[label]))):
            print(sentence_label_dict[label][i].replace(" ",""))
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     cluster_points = vectors[label]
    #     for i in range(min(1, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))

        print("---------")

if __name__ == "__main__":
    main()

