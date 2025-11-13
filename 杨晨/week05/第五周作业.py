#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法，并添加类内平均距离排序功能
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances  # 用于计算欧氏距离

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

    # 获取聚类标签和中心点
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 将句子和向量按标签分组
    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    for sentence, label, vector in zip(sentences, labels, vectors):
        sentence_label_dict[label].append(sentence)
        vector_label_dict[label].append(vector)

    # 计算每个聚类的类内平均距离
    cluster_avg_distances = {}
    for label in range(n_clusters):
        if label in vector_label_dict:
            vec_list = vector_label_dict[label]
            center = centers[label]
            # 计算每个向量到聚类中心的欧氏距离
            distances = [euclidean_distances([v], [center])[0][0] for v in vec_list]
            avg_distance = np.mean(distances)  # 求平均距离
            cluster_avg_distances[label] = avg_distance
        else:
            cluster_avg_distances[label] = float('inf')  # 如果聚类为空，设为无穷大（避免错误）

    # 按类内平均距离由小到大排序标签
    sorted_labels = sorted(cluster_avg_distances.keys(), key=lambda x: cluster_avg_distances[x])

    # 输出排序后的聚类结果
    print("按类内平均距离排序后的聚类结果（距离越小越紧致）：")
    for label in sorted_labels:
        avg_dist = cluster_avg_distances[label]
        print(f"cluster {label} (类内平均距离: {avg_dist:.4f}) :")
        sentences_in_cluster = sentence_label_dict[label]
        for i in range(min(10, len(sentences_in_cluster))):  # 每类最多显示10句
            print(sentences_in_cluster[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
