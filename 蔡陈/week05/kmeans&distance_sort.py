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
from sklearn.metrics import pairwise_distances
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


def calculate_intra_cluster_distances(vectors, labels, cluster_centers):
    """计算每个样本到其所属聚类中心的距离"""
    distances = []
    for i, (vector, label) in enumerate(zip(vectors, labels)):
        cluster_center = cluster_centers[label]
        distance = np.linalg.norm(vector - cluster_center)  # 欧氏距离
        distances.append((i, distance, label))
    return distances


def sort_sentences_by_distance(sentences, vectors, labels, cluster_centers):
    """对每个聚类内的句子按到聚类中心的距离排序"""
    # 计算所有样本到聚类中心的距离
    distances = calculate_intra_cluster_distances(vectors, labels, cluster_centers)

    # 按聚类标签分组，并在每个聚类内按距离排序
    cluster_sorted_sentences = defaultdict(list)

    for idx, distance, label in distances:
        sentence = sentences[idx]
        cluster_sorted_sentences[label].append((sentence, distance, idx))

    # 对每个聚类内的句子按距离排序（从近到远）
    for label in cluster_sorted_sentences:
        cluster_sorted_sentences[label].sort(key=lambda x: x[1])

    return cluster_sorted_sentences


def main():
    model = load_word2vec_model(r"/Users/apple/Desktop/badouweek5assignment/model.w2v")  # 加载词向量模型
    sentences = list(load_sentence("/Users/apple/Desktop/badouweek5assignment/titles.txt"))  # 加载所有标题并转为list
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, random_state=42)  # 定义一个kmeans计算类，设置random_state保证结果可重现
    kmeans.fit(vectors)  # 进行聚类计算

    # 获取聚类结果
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # 按类内距离排序
    sorted_clusters = sort_sentences_by_distance(sentences, vectors, labels, cluster_centers)

    # 输出排序后的结果
    for label in sorted(range(n_clusters), key=lambda x: len(sorted_clusters[x]), reverse=True):
        cluster_sentences = sorted_clusters[label]
        print(f"\ncluster {label} (共{len(cluster_sentences)}个样本):")
        print("样本分布（按距离聚类中心从近到远）：")

        # 计算该聚类的平均距离
        avg_distance = np.mean([dist for _, dist, _ in cluster_sentences])
        print(f"平均类内距离: {avg_distance:.4f}")

        for i, (sentence, distance, original_idx) in enumerate(cluster_sentences[:15]):  # 显示前15个
            print(f"  {i + 1:2d}. [距离:{distance:.4f}] {sentence.replace(' ', '')}")

        if len(cluster_sentences) > 15:
            print(f"  ... 还有{len(cluster_sentences) - 15}个样本")
        print("---------")


if __name__ == "__main__":
    main()