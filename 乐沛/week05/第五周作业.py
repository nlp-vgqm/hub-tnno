#!/usr/bin/env python3
#coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

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

#将文本向量化
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
    global avg_distance
    model = load_word2vec_model(r"D:/Desktop/资料/NLP/课程/第五周 词向量/week5 词向量及文本向量/model.w2v") # 加载词向量模型
    sentences = load_sentence("D:/Desktop/资料/NLP/课程/第五周 词向量/week5 词向量及文本向量/titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)   # 将所有标题向量化


    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算
    labels = kmeans.labels_      # 获取每个簇的数据向量
    sentence_label_dict = defaultdict(list)

    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        centers = kmeans.cluster_centers_  # 每个聚类中心
        sentence_label_dict[label].append(sentence)         # 同标签的放到一起

    cluster_stats = []
    for cluster_id in range(kmeans.n_clusters):
        # 获取属于当前簇的所有数据点
        cluster_vectors = vectors[labels == cluster_id]
        cluster_center = kmeans.cluster_centers_[cluster_id]

        # 计算每个点到簇中心的距离
        if len(cluster_vectors) > 0:
            distances = euclidean_distances(cluster_vectors, [cluster_center])
            distances = distances.flatten()

            # 计算平均距离
            avg_distance = np.mean(distances)

            # print(f"各点到中心的距离: {distances}")
            # print(f"平均距离: {avg_distance:.4f}")
        else:
            print("该簇没有数据点")
        # 存储簇的统计信息
        cluster_stats.append({
            'cluster_id': cluster_id,
            'avg_distance': avg_distance,
            'sentences': sentence_label_dict[cluster_id],
            'num_sentences': len(cluster_vectors)
        })
        # print("-" * 30)

    # 按平均距离从高到低排序
    cluster_stats_sorted = sorted(cluster_stats, key=lambda x: x['avg_distance'], reverse=False)

    # 按排序后的顺序打印每个簇的内容
    for cluster_info in cluster_stats_sorted:
        cluster_id = cluster_info['cluster_id']
        avg_distance = cluster_info['avg_distance']
        sentences = cluster_info['sentences']
        num_sentences = cluster_info['num_sentences']

        print(f"\n簇 {cluster_id} (平均距离: {avg_distance:.4f}, 句子数量: {num_sentences}):")


        for i in range(len(sentences)):
            print(f"  {sentences[i].replace(' ', '')}")
        print("-" * 60)

if __name__ == "__main__":
    main()

