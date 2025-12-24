#!/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类，按类内距离来排序打印
# 聚类采用Kmeans算法
import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


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


def get_distance(p1, p2):
    # 计算两点间距
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)


def get_avg_distance(sentences, labels, cluster_centers, vectors):
    sentence_label_dict = defaultdict(list)
    label_total_distance_dict = defaultdict(float)
    label_count_dict = defaultdict(int)
    label_avg_distance_dict = {}
    for sentence, label, vector in zip(sentences, labels,
                                       vectors):  # 取出句子和标签
        # kmeans.cluster_centers_ #每个聚类中心
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
        distance = get_distance(vector, cluster_centers[label])
        label_total_distance_dict[label] += distance
        label_count_dict[label] += 1
    for label in label_total_distance_dict:
        if label_count_dict[label] > 0:
            label_avg_distance_dict[label] = label_total_distance_dict[label] / label_count_dict[label]
        else:
            label_avg_distance_dict[label] = 0.0

    return sentence_label_dict, label_avg_distance_dict, label_count_dict


def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算
    sentence_label_dict, label_avg_distance_dict, label_count_dict = get_avg_distance(sentences, kmeans.labels_,
                                                                                      kmeans.cluster_centers_, vectors)
    # 将标签按平均距离排序
    sorted_labels_avg_distance = sorted(label_avg_distance_dict.items(), key=lambda x: x[1])

    print("聚类结果（按平均距离从小到大）")
    print("=" * 20)
    for i, (label, avg_distance) in enumerate(sorted_labels_avg_distance, 1):

        print(f"簇 {i} (标签: {label})")
        print(f"  平均距离: {avg_distance:.4f}")
        print(f"  句子数量: {label_count_dict[label]}")
        print("  句子列表: ")

        for i in range(min(10, len(sentence_label_dict[label]))):  # 随便打印几个，太多了看不过来
            print('    ', sentence_label_dict[label][i].replace(" ", ""))

        print("-" * 50)


if __name__ == "__main__":
    main()
