#!/usr/bin/env python3
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict

# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    try:
        model = Word2Vec.load(path)
        return model
    except FileNotFoundError:
        print(f"错误：词向量模型文件 '{path}' 未找到。")
        exit()

def load_sentence(path):
    sentences = set()
    try:
        with open(path, encoding="utf8") as f:
            for line in f:
                sentence = line.strip()
                if sentence: # 跳过空行
                    sentences.add(" ".join(jieba.cut(sentence)))
    except FileNotFoundError:
        print(f"错误：句子文件 '{path}' 未找到。")
        exit()

    print("获取句子数量：", len(sentences))
    return sentences

# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        valid_word_count = 0
        for word in words:
            try:
                vector += model.wv[word]
                valid_word_count += 1
            except KeyError:
                continue
        if valid_word_count > 0:
            vector = vector / valid_word_count
        vectors.append(vector)
    return np.array(vectors)

def main():
    model = load_word2vec_model(r"model.w2v") # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)   # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, random_state=42)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    # --- 新增代码：计算每个簇的类内平均距离 ---
    cluster_distances = defaultdict(list)
    cluster_sentences = defaultdict(list)

    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):
        cluster_sentences[label].append(sentence)
        # 计算句子向量与对应簇中心的距离
        center = kmeans.cluster_centers_[label]
        distance = euclidean_distances([vector], [center])[0][0]
        cluster_distances[label].append(distance)

    # 计算每个簇的平均距离，并存储在字典中 {簇标签: 平均距离}
    cluster_avg_distance = {}
    for label, distances in cluster_distances.items():
        avg_dist = sum(distances) / len(distances)
        cluster_avg_distance[label] = avg_dist

    # --- 新增代码：根据类内平均距离对簇进行排序 ---
    # sorted_clusters 是一个列表，元素为 (簇标签, 平均距离)，按平均距离从小到大排序
    sorted_clusters = sorted(cluster_avg_distance.items(), key=lambda item: item[1])

    # --- 修改输出部分：按排序后的顺序输出 ---
    print("\n" + "="*50)
    print(f"聚类结果（按类内平均距离从小到大排序）")
    print("="*50)
    for label, avg_dist in sorted_clusters:
        print(f"\n【簇 {label}】 - 类内平均距离: {avg_dist:.4f}")
        print(f"簇内句子数量: {len(cluster_sentences[label])}")
        print("-" * 30)
        print("簇内前10个句子:")
        for i in range(min(10, len(cluster_sentences[label]))):
            print(cluster_sentences[label][i].replace(" ", ""))
        print("-" * 30)

if __name__ == "__main__":
    main()
