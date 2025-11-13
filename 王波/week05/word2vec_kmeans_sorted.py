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
from collections import defaultdict
from sklearn.metrics.pairwise import pairwise_distances

#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

#加载文件
def load_sentence(path):
    sentences = set()  # 集合是无序的，会导致每次运行顺序不同
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    # 将set转换为list，并排序，以确保每次顺序相同
    sentences_list = sorted(list(sentences))
    print("获取句子数量：", len(sentences_list))
    return sentences_list

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


# 计算每个句子到其聚类中心的距离并排序
def sort_sentences_by_distance(sentences, vectors, kmeans):
    sentence_label_dict = defaultdict(list)

    # 计算每个样本到其所属聚类中心的距离
    distances = pairwise_distances(vectors, kmeans.cluster_centers_, metric='euclidean')

    # 获取每个样本到其所属聚类中心的距离
    sample_distances = distances[np.arange(len(vectors)), kmeans.labels_]

    # 将句子、向量、标签和距离组合在一起
    for i, (sentence, label, distance) in enumerate(zip(sentences, kmeans.labels_, sample_distances)):
        sentence_label_dict[label].append((sentence, distance, vectors[i]))

    # 对每个聚类内的句子按距离排序（距离越小越靠前）
    for label in sentence_label_dict:
        sentence_label_dict[label].sort(key=lambda x: x[1])

    return sentence_label_dict


#关键词提取
def extract_cluster_keywords(sentence_label_dict, top_n=5):
    print("\n=== 各聚类关键词分析 ===")

    for label, sentence_data in sentence_label_dict.items():
        # 取距离中心最近的几个句子作为代表性样本
        representative_sentences = [data[0] for data in sentence_data[:10]]

        # 简单的词频统计
        word_freq = defaultdict(int)
        for sentence in representative_sentences:
            words = sentence.split()
            for word in words:
                if len(word) > 1:  # 过滤单字
                    word_freq[word] += 1

        # 按词频排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        print(f"聚类 {label} 的关键词:")
        keywords = [word for word, freq in sorted_words[:top_n]]
        print(f"  {', '.join(keywords)}")

def main():
    model = load_word2vec_model(r"word2vec_model.model") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    sentences_list = list(sentences)  # 转换为列表以便保持顺序
    vectors = sentences_to_vectors(sentences_list, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences_list)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)   # 聚类中心
    kmeans = KMeans(n_clusters, random_state=10)  #定义一个kmeans计算类,设置随机种子以便结果可重现
    kmeans.fit(vectors)          #进行聚类计算

    # 无序排列
    # sentence_label_dict = defaultdict(list)
    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     #kmeans.cluester_centers_ #每个聚类中心
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

    # 使用排序功能
    sentence_label_dict = sort_sentences_by_distance(sentences_list, vectors, kmeans)
    cluster_sizes = [len(sentences) for sentences in sentence_label_dict.values()]
    print("\n=== 聚类大小分布统计 ===")
    print(f"总聚类数: {len(cluster_sizes)}")
    print(f"总样本数: {sum(cluster_sizes)}")
    print(f"最大聚类大小: {max(cluster_sizes)}")
    print(f"最小聚类大小: {min(cluster_sizes)}")
    print(f"平均聚类大小: {np.mean(cluster_sizes):.2f}")
    print(f"聚类大小标准差: {np.std(cluster_sizes):.2f}")
    print("====================")

    # 按聚类标签排序
    sorted_labels = sorted(sentence_label_dict.keys())

    # 输出每个聚类的信息
    for label in sorted_labels:
        sentence_data = sentence_label_dict[label]

        print("=" * 50)
        # 取距离中心最近的几个句子作为代表性样本
        representative_sentences = [data[0] for data in sentence_data[:10]]

        # 简单的词频统计
        word_freq = defaultdict(int)
        for sentence in representative_sentences:
            words = sentence.split()
            for word in words:
                if len(word) > 1:  # 过滤单字
                    word_freq[word] += 1

        # 按词频排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        print(f"聚类 {label} 的关键词:")
        keywords = [word for word, freq in sorted_words[:5]]  # 前5项
        print(f"{', '.join(keywords)}")
        print(f"聚类中心坐标: {kmeans.cluster_centers_[label][:5]}...")  # 只显示前5维
        print(f"共{len(sentence_data)}个句子:")
        print("代表性句子（距离聚类中心最近的10个）:")
        for i, (sentence, distance, vector) in enumerate(sentence_data[:10]):
            print(f"  {i + 1:2d}. [距离: {distance:.4f}] {sentence.replace(' ', '')}")
        # 可选：显示一些统计信息
        distances = [data[1] for data in sentence_data]
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        print(f"类内距离统计 - 平均: {avg_distance:.4f}, 标准差: {std_distance:.4f}")
        print()

if __name__ == "__main__":
    main()

