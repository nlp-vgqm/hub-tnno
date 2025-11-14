import numpy as np
import random
import sys
'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            return result, self.points, sum
        self.points = np.array(new_center)
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

    def sort_clusters_by_distance(self, result, centers):
        """
        基于类内距离对簇进行排序

        Args:
            result: 聚类结果，每个簇的数据点列表
            centers: 每个簇的中心点

        Returns:
            sorted_result: 按类内距离排序后的簇列表
            sorted_centers: 对应的中心点
            intra_distances: 每个簇的类内距离
        """
        intra_distances = []

        # 计算每个簇的类内距离（各点到中心点距离的平均值）
        for i in range(len(result)):
            cluster_points = np.array(result[i])
            center = np.array(centers[i])

            if len(cluster_points) == 0:
                intra_distances.append(0)
                continue

            # 计算簇内每个点到中心点的距离
            distances = [self.__distance(point, center) for point in cluster_points]
            # 计算平均距离作为类内距离
            avg_distance = np.mean(distances)
            intra_distances.append(avg_distance)

        # 根据类内距离排序（从大到小）
        sorted_indices = np.argsort(intra_distances)[::-1]
        print(f'sorted_indices: {sorted_indices}')

        sorted_result = [result[i] for i in sorted_indices]
        sorted_centers = [centers[i] for i in sorted_indices]
        sorted_distances = [intra_distances[i] for i in sorted_indices]

        return sorted_result, sorted_centers, sorted_distances


x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 10)
result, centers, distances = kmeans.cluster()
print(result)
print(centers)
print(distances)
print("*" * 50)
# 基于类内举例进行排序
sorted_result, sorted_centers, sorted_distances = kmeans.sort_clusters_by_distance(result, centers)
print(sorted_result)
print(sorted_centers)
print(sorted_distances)