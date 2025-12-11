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


def distance_1(p1, p2):
    # 计算两点间距欧氏距离
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)


def distance_2(p1, p2):
    # 计算两点间的余弦距离
    # 计算点积
    dot_product = sum(a * b for a, b in zip(p1, p2))

    # 计算模长
    norm1 = sum(a * a for a in p1) ** 0.5
    norm2 = sum(b * b for b in p2) ** 0.5

    # 避免除零错误
    if norm1 == 0 or norm2 == 0:
        return 1.0

    # 计算余弦相似度并转换为余弦距离
    cosine_similarity = dot_product / (norm1 * norm2)
    cosine_distance = 1 - cosine_similarity

    return cosine_distance


def distance_3(p1, p2):
    #  曼哈顿距离
    return sum(abs(a - b) for a, b in zip(p1, p2))


def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    # 初始化距离列表，每个聚类对应一个空列表来存储所有距离
    cluster_distances_1 = [[] for _ in range(n_clusters)]  # 欧氏距离
    cluster_distances_2 = [[] for _ in range(n_clusters)]  # 余弦距离
    cluster_distances_3 = [[] for _ in range(n_clusters)]  # 曼哈顿距离

    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):
        sentence_label_dict[label].append(sentence)
        # 计算每个样本到聚类中心的距离并保存
        cluster_distances_1[label].append(distance_1(vector, kmeans.cluster_centers_[label]))
        cluster_distances_2[label].append(distance_2(vector, kmeans.cluster_centers_[label]))
        cluster_distances_3[label].append(distance_3(vector, kmeans.cluster_centers_[label]))

    # 计算每个聚类的平均距离
    cluster_avg_distance_1 = [sum(distances) / len(distances) if distances else 0
                              for distances in cluster_distances_1]
    cluster_avg_distance_2 = [sum(distances) / len(distances) if distances else 0
                              for distances in cluster_distances_2]
    cluster_avg_distance_3 = [sum(distances) / len(distances) if distances else 0
                              for distances in cluster_distances_3]

    # 按不同距离度量对聚类进行排序（从小到大）
    sorted_indices_1 = sorted(range(n_clusters), key=lambda i: cluster_avg_distance_1[i])
    sorted_indices_2 = sorted(range(n_clusters), key=lambda i: cluster_avg_distance_2[i])
    sorted_indices_3 = sorted(range(n_clusters), key=lambda i: cluster_avg_distance_3[i])

    # 打印不同距离度量的排序结果
    print("\n=== 欧氏距离排序 ===")
    for i, idx in enumerate(sorted_indices_1):
        count = len(cluster_distances_1[idx])
        print(f"聚类 {idx}: 平均距离={cluster_avg_distance_1[idx]:.4f}, 样本数={count}")

    print("\n=== 余弦距离排序 ===")
    for i, idx in enumerate(sorted_indices_2):
        count = len(cluster_distances_2[idx])
        print(f"聚类 {idx}: 平均距离={cluster_avg_distance_2[idx]:.4f}, 样本数={count}")

    print("\n=== 曼哈顿距离排序 ===")
    for i, idx in enumerate(sorted_indices_3):
        count = len(cluster_distances_3[idx])
        print(f"聚类 {idx}: 平均距离={cluster_avg_distance_3[idx]:.4f}, 样本数={count}")

    # 选择余弦距离作为主要评判标准（适合词向量）
    selected_distances = cluster_avg_distance_2
    selected_sorted_indices = sorted_indices_2

    # 设置舍弃阈值：舍弃平均距离最大的30%的聚类
    discard_ratio = 0.3
    keep_count = int(n_clusters * (1 - discard_ratio))
    kept_clusters = selected_sorted_indices[:keep_count]
    discarded_clusters = selected_sorted_indices[keep_count:]

    print(f"\n=== 聚类筛选结果（基于余弦距离，舍弃后{discard_ratio * 100}%） ===")
    print(f"保留聚类数量: {len(kept_clusters)}")
    print(f"舍弃聚类数量: {len(discarded_clusters)}")

    # # 显示保留的聚类详情
    # print("\n=== 保留的聚类详情 ===")
    # for label in kept_clusters:
    #     sentences_in_cluster = sentence_label_dict[label]
    #     avg_distance = selected_distances[label]
    #
    #     print(f"\n聚类 {label} (平均余弦距离: {avg_distance:.4f}, 样本数: {len(sentences_in_cluster)}):")
    #     for i in range(min(5, len(sentences_in_cluster))):  # 每个聚类显示5个样本
    #         print(f"  {sentences_in_cluster[i].replace(' ', '')}")
    #
    # # 显示舍弃的聚类统计
    # if discarded_clusters:
    #     print(f"\n=== 舍弃的聚类统计 ===")
    #     for label in discarded_clusters:
    #         sentences_in_cluster = sentence_label_dict[label]
    #         avg_distance = selected_distances[label]
    #         print(f"聚类 {label}: 平均距离={avg_distance:.4f}, 样本数={len(sentences_in_cluster)}")

    # 计算整体聚类质量
    print(f"\n=== 聚类质量统计 ===")
    total_kept_samples = sum(len(sentence_label_dict[label]) for label in kept_clusters)
    total_discarded_samples = sum(len(sentence_label_dict[label]) for label in discarded_clusters)
    print(f"保留样本数: {total_kept_samples}")
    print(f"舍弃样本数: {total_discarded_samples}")
    print(f"保留比例: {total_kept_samples / len(sentences) * 100:.1f}%")

    # 计算保留聚类的平均距离
    if kept_clusters:
        avg_kept_distance = np.mean([selected_distances[label] for label in kept_clusters])
        avg_discarded_distance = np.mean(
            [selected_distances[label] for label in discarded_clusters]) if discarded_clusters else 0
        print(f"保留聚类的平均距离: {avg_kept_distance:.4f}")
        if discarded_clusters:
            print(f"舍弃聚类的平均距离: {avg_discarded_distance:.4f}")



if __name__ == "__main__":
    main()
