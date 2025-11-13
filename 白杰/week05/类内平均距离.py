#!/usr/bin/env python3  
# coding: utf-8

# 基于词向量+KMeans聚类，按句子到聚类中心的距离（升序）排序输出
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 加载Word2Vec模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


# 加载句子（保留原始句子、分词句子和索引，便于后续匹配向量）
def load_sentence(path):
    sentences = []  # 格式：[(原始句子, 分词后句子), ...]
    seen = set()  # 去重：记录已出现的原始句子
    with open(path, encoding="utf8") as f:
        for line in f:
            raw_sent = line.strip()
            if not raw_sent or raw_sent in seen:
                continue
            seen.add(raw_sent)
            cut_sent = " ".join(jieba.cut(raw_sent))
            sentences.append((raw_sent, cut_sent))
    print("获取句子数量：", len(sentences))
    return sentences


# 句子向量化（基于Word2Vec词向量的平均值）
def sentences_to_vectors(cut_sentences, model):
    vectors = []
    for sentence in cut_sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        if len(words) > 0:
            vector /= len(words)
        vectors.append(vector)
    return np.array(vectors)


# 计算句子到聚类中心的距离（欧氏距离）
def calculate_distances(vectors, labels, centers):
    """
    vectors: 所有句子的向量数组
    labels: 每个句子的聚类标签
    centers: 聚类中心数组（kmeans.cluster_centers_）
    返回：每个句子到其所属聚类中心的距离列表
    """
    distances = []
    for i in range(len(vectors)):
        label = labels[i]  # 句子i的标签
        center = centers[label]  # 对应聚类中心
        # 计算欧氏距离：sqrt(sum((x-y)^2))，np.linalg.norm直接实现
        dist = np.linalg.norm(vectors[i] - center)
        distances.append(dist)
    return distances


def main():
    # 1. 加载模型和句子
    model = load_word2vec_model(r"model.w2v")
    sentences = load_sentence("titles.txt")  # 格式：[(原始句子, 分词句子), ...]
    raw_sentences = [s[0] for s in sentences]  # 原始句子列表
    cut_sentences = [s[1] for s in sentences]  # 分词句子列表

    # 2. 句子向量化
    vectors = sentences_to_vectors(cut_sentences, model)

    # 3. KMeans聚类（固定random_state确保结果可复现）
    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, random_state=42)  # 固定随机种子，聚类结果固定
    kmeans.fit(vectors)
    labels = kmeans.labels_  # 每个句子的聚类标签
    centers = kmeans.cluster_centers_  # 每个聚类的中心向量

    # 4. 计算每个句子到其所属聚类中心的距离
    distances = calculate_distances(vectors, labels, centers)

    # 5. 按聚类标签分组，每组包含（原始句子, 距离）
    cluster_dict = defaultdict(list)
    for i in range(len(sentences)):
        label = labels[i]
        raw_sent = raw_sentences[i]
        dist = distances[i]
        cluster_dict[label].append((raw_sent, dist))  # 存储句子和对应的距离

    # 6. 每个簇内按距离从小到大排序（距离越小越稳定）并输出
    for label, items in cluster_dict.items():
        # 按距离升序排序（key=lambda x: x[1]）
        sorted_items = sorted(items, key=lambda x: x[1])
        # 提取排序后的句子（距离仅用于排序，输出时可省略）
        sorted_sents = [item[0] for item in sorted_items]
        print(f"cluster {label}（共{len(sorted_sents)}个句子）:")
        # 输出前10个最稳定的句子（距离最小）
        for sent in sorted_sents[:10]:
            print(f"  - {sent}")
        print("-" * 50)


if __name__ == "__main__":
    main()
