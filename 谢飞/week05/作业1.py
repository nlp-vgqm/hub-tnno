import os

import numpy as np
from sklearn.cluster import KMeans
import matplotlib
from matplotlib import font_manager

matplotlib.use("TkAgg")

_font_candidates = [
    (r"C:\Windows\Fonts\msyh.ttc", "Microsoft YaHei"),
    (r"C:\Windows\Fonts\simhei.ttf", "SimHei"),
    (r"C:\Windows\Fonts\simkai.ttf", "KaiTi"),
]
for font_path, font_name in _font_candidates:
    if os.path.exists(font_path):
        try:
            font_manager.fontManager.addfont(font_path)
        except Exception:
            continue
        matplotlib.rcParams["font.family"] = font_name
        break
else:
    matplotlib.rcParams["font.family"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]

matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

class KMeansIntraDistanceSorter:
    """
    基于KMeans聚类结果进行类内距离排序的类
    """

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.cluster_centers_ = None
        self.labels_ = None
        self.intra_distances_ = None

    def fit(self, X):
        """
        训练KMeans模型
        """
        self.kmeans.fit(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.labels_ = self.kmeans.labels_
        self.intra_distances_ = None
        return self

    def calculate_intra_cluster_distances(self, X):
        """
        计算每个样本到其所属簇中心的距离
        """
        if self.labels_ is None:
            raise ValueError("请先调用fit方法训练模型")

        X = np.asarray(X)
        if X.shape[0] != self.labels_.shape[0]:
            raise ValueError("输入数据的样本数量与训练时不一致")

        assigned_centers = self.cluster_centers_[self.labels_]
        intra_distances = np.linalg.norm(X - assigned_centers, axis=1)

        self.intra_distances_ = intra_distances
        return intra_distances

    def sort_by_intra_distance(self, X, ascending=True):
        """
        基于类内距离对样本进行排序

        参数:
        - X: 输入数据
        - ascending: 是否升序排列，True表示距离小的在前

        返回:
        - sorted_indices: 排序后的索引
        - sorted_distances: 排序后的距离
        - cluster_info: 每个簇的统计信息
        """
        if self.intra_distances_ is None or len(self.intra_distances_) != len(X):
            intra_distances = self.calculate_intra_cluster_distances(X)
        else:
            intra_distances = self.intra_distances_

        # 按距离排序
        if ascending:
            sorted_indices = np.argsort(intra_distances)
        else:
            sorted_indices = np.argsort(intra_distances)[::-1]

        sorted_distances = intra_distances[sorted_indices]

        # 计算每个簇的统计信息
        cluster_info = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels_ == cluster_id
            cluster_distances = intra_distances[cluster_mask]
            if cluster_distances.size == 0:
                cluster_info[cluster_id] = {
                    'count': 0,
                    'mean_distance': None,
                    'max_distance': None,
                    'min_distance': None,
                    'std_distance': None
                }
                continue

            cluster_info[cluster_id] = {
                'count': int(np.sum(cluster_mask)),
                'mean_distance': float(np.mean(cluster_distances)),
                'max_distance': float(np.max(cluster_distances)),
                'min_distance': float(np.min(cluster_distances)),
                'std_distance': float(np.std(cluster_distances, ddof=0))
            }

        return sorted_indices, sorted_distances, cluster_info

    def get_cluster_sorted_data(self, X, ascending=True):
        """
        获取按簇分组并排序的数据

        返回:
        - cluster_sorted_data: 字典，键为簇ID，值为该簇内排序后的(索引, 距离)列表
        """
        if self.intra_distances_ is None or len(self.intra_distances_) != len(X):
            intra_distances = self.calculate_intra_cluster_distances(X)
        else:
            intra_distances = self.intra_distances_

        cluster_sorted_data = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels_ == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_distances = intra_distances[cluster_mask]

            # 对当前簇内的样本按距离排序
            if ascending:
                sorted_within_cluster = sorted(zip(cluster_indices, cluster_distances),
                                            key=lambda x: x[1])
            else:
                sorted_within_cluster = sorted(zip(cluster_indices, cluster_distances),
                                            key=lambda x: x[1], reverse=True)

            cluster_sorted_data[cluster_id] = sorted_within_cluster

        return cluster_sorted_data

def visualize_intra_cluster_distances(X, labels, cluster_centers, sorted_indices, sorted_distances):
    """
    可视化类内距离排序结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 原始聚类结果
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        axes[0,0].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                      markeredgecolor='k', markersize=8, label=f'簇 {k}')
        axes[0,0].plot(cluster_centers[k, 0], cluster_centers[k, 1],
                      marker="*", linestyle="", markerfacecolor=col,
                      markeredgecolor='k', markersize=14)

    axes[0,0].set_title('KMeans聚类结果')
    axes[0,0].legend()

    # 2. 类内距离分布
    for k in unique_labels:
        cluster_distances = sorted_distances[labels[sorted_indices] == k]
        axes[0,1].hist(cluster_distances, alpha=0.7, label=f'簇 {k}')

    axes[0,1].set_title('类内距离分布')
    axes[0,1].set_xlabel('到簇中心的距离')
    axes[0,1].set_ylabel('频数')
    axes[0,1].legend()

    # 3. 排序后的距离（全局）
    axes[1,0].plot(range(len(sorted_distances)), sorted_distances, 'b-', alpha=0.7)
    axes[1,0].set_title('类内距离排序（全局）')
    axes[1,0].set_xlabel('排序后的样本索引')
    axes[1,0].set_ylabel('到簇中心的距离')

    # 4. 每个簇内排序的距离
    for k in unique_labels:
        cluster_mask = labels[sorted_indices] == k
        cluster_sorted_distances = sorted_distances[cluster_mask]
        axes[1,1].plot(range(len(cluster_sorted_distances)), cluster_sorted_distances,
                      'o-', label=f'簇 {k}', alpha=0.7)

    axes[1,1].set_title('类内距离排序（按簇）')
    axes[1,1].set_xlabel('簇内样本序号')
    axes[1,1].set_ylabel('到簇中心的距离')
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    from sklearn.datasets import make_blobs

    # 创建测试数据
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0,
                          random_state=42, n_features=2)

    # 创建排序器并训练
    sorter = KMeansIntraDistanceSorter(n_clusters=4, random_state=42)
    sorter.fit(X)

    print("聚类中心:")
    print(sorter.cluster_centers_)
    print(f"\n每个簇的样本数量: {np.bincount(sorter.labels_)}")

    # 方法1: 全局排序
    print("\n=== 全局排序结果 ===")
    sorted_indices, sorted_distances, cluster_info = sorter.sort_by_intra_distance(X, ascending=True)

    print("前10个最近样本的索引:", sorted_indices[:10])
    print("前10个最近样本的距离:", sorted_distances[:10])

    print("\n簇统计信息:")
    for cluster_id, info in cluster_info.items():
        if info['mean_distance'] is None:
            print(f"簇 {cluster_id}: 样本数=0")
            continue

        print(
            f"簇 {cluster_id}: 样本数={info['count']}, "
            f"平均距离={info['mean_distance']:.3f}, "
            f"最大距离={info['max_distance']:.3f}, "
            f"最小距离={info['min_distance']:.3f}, "
            f"距离标准差={info['std_distance']:.3f}"
        )

    # 方法2: 按簇分组排序
    print("\n=== 按簇分组排序 ===")
    cluster_sorted_data = sorter.get_cluster_sorted_data(X, ascending=True)

    for cluster_id, sorted_data in cluster_sorted_data.items():
        print(f"\n簇 {cluster_id} (前5个最近样本):")
        for idx, (sample_idx, distance) in enumerate(sorted_data[:5]):
            print(f"  样本 {sample_idx}: 距离 = {distance:.3f}")

    # 可视化结果
    visualize_intra_cluster_distances(X, sorter.labels_, sorter.cluster_centers_,
                                    sorted_indices, sorted_distances)

    # 应用场景示例：异常检测（距离大的可能是异常点）
    print("\n=== 异常检测示例 ===")
    # 找出每个簇中距离最远的样本（可能的异常点）
    outlier_threshold_factor = 2.0
    potential_outliers = []

    for cluster_id, info in cluster_info.items():
        if info['mean_distance'] is None:
            continue

        cluster_mask = sorter.labels_ == cluster_id
        cluster_distances = sorter.intra_distances_[cluster_mask]
        threshold = info['mean_distance'] + outlier_threshold_factor * np.std(
            cluster_distances
        )
        cluster_outliers = []
        for sample_idx, distance in cluster_sorted_data[cluster_id]:
            if distance > threshold:
                cluster_outliers.append((sample_idx, distance))

        if cluster_outliers:
            print(f"簇 {cluster_id} 中可能的异常点:")
            for sample_idx, distance in cluster_outliers[-3:]:  # 显示距离最大的3个
                print(f"  样本 {sample_idx}: 距离 = {distance:.3f} (阈值 = {threshold:.3f})")
            potential_outliers.extend(cluster_outliers)