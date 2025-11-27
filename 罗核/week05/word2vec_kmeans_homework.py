#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import operator
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
    # 将句向量和对应分配和label组成字典
    vector_label_dict = defaultdict(list)
    # 将每个聚类中心的向量和label组成字典
    center_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        #kmeans.cluester_centers_ #每个聚类中心 
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # 填充vector_label_dict字典
    for vector, label in zip(vectors, kmeans.labels_):
        vector_label_dict[label].append(vector)
    # 填充聚类中心向量
    for label in kmeans.labels_:
        center_label_dict[label].append(kmeans.cluster_centers_[label-1])
    # 计算类内距离
    distance_desc_dict = distance_center(center_label_dict, vector_label_dict)
    for key in distance_desc_dict:
        print("cluster %s , 类内距 %s:" % (key, distance_desc_dict[key]))
        sentences = sentence_label_dict[key]
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

# 计算每个聚类中心的向量和这一类中所有其他的句向量的距离
# 计算完成后，将label和计算的类内距组成字典，并按类内距降序返回
def distance_center(center_label_dict, vector_label_dict):
    distance_dict = {}
    for label, cluster_center in center_label_dict.items():
        total_distince = comp(cluster_center, vector_label_dict[label])
        distance_dict[label] = total_distince
    # 按类内距降序排序
    distance_desc_dict = dict(sorted(distance_dict.items(), key=operator.itemgetter(1), reverse=True))
    return distance_desc_dict


# 计算某一个聚类中心的向量和这一组其他句向量的距离和
def comp(cluster_center, vectors):
    total = 0
    for i in range(len(vectors)):
        total += distance_tow_p(cluster_center, vectors[i])
    return total

# 计算两个向量间的距离
def distance_tow_p(p1, p2):
    #计算两点间距
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)

if __name__ == "__main__":
    main()

