#!/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
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


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, random_state=42)  # 定义一个kmeans计算类，设置随机种子保证结果可重现
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_distance_dict = defaultdict(list)
    cluster_avg_distance = {}  # 存储每个聚类的平均距离
    # 计算每个样本到其所属聚类中心的距离
    for i, (sentence, vector) in enumerate(zip(sentences, vectors)):
        label = kmeans.labels_[i]
        cluster_center = kmeans.cluster_centers_[label]

        distance = np.linalg.norm(vector - cluster_center)
        sentence_distance_dict[label].append((sentence, distance))

    #计算每个聚类的平均距离
    for label in sentence_distance_dict:
        sentence_distance_dict[label].sort(key=lambda x: x[1])
        # 计算该聚类的平均距离
        distances = [item[1] for item in sentence_distance_dict[label]]
        cluster_avg_distance[label] = np.mean(distances)

    # 按类内平均距离升序排序聚类
    sorted_clusters = sorted(cluster_avg_distance.items(), key=lambda x: x[1])

    # 输出结果 - 按类内平均距离升序排列
    print("按类内平均距离升序排列的聚类结果：")
    print("=" * 60)

    for label, avg_distance in sorted_clusters:
        cluster_sentences = sentence_distance_dict[label]
        print(f"簇 {label} (共{len(cluster_sentences)}个句子, 平均距离: {avg_distance:.4f}):")
        print("距离中心最近的5个句子:")
        for i, (sentence, distance) in enumerate(cluster_sentences[:5]):
            print(f"  {i + 1}. {sentence.replace(' ', '')} (距离: {distance:.4f})")
        print("---------")

    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     #kmeans.cluester_centers_ #每个聚类中心
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")


if __name__ == "__main__":
    main()
