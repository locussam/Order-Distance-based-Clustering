#pragma once
#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "global.h"

/*产生数据,
data:每行为一个数据,列数为其特征维数*/
void generate_data(cv::Mat& data,const int samples,const int DIM);

/*聚类
input:
data:输入数据矩阵
feature_name_v: 输入数据矩阵,每行向量对应的照片名称
k: 计算k近邻
K: 计算样本间K个近邻的平均(K<= k, K越大,聚类越紧凑)
output:
cluster_vec: 输出的聚类的类向量组,每行为一类,其中为照片序号
cluster_name_v: 输出的聚类的向量组,每行为一类,其中为照片名
*/
void cluster(const cv::Mat&data,const std::vector<std::string>& feature_name_v,
	int k,const int K, 
	std::vector<std::vector<int>>&cluster_vec, 
	std::vector<std::vector<std::string>>&cluster_name_v);

/*利用FLANN库得到数据的初始k近邻矩阵与距离矩阵
data: 输入的特征矩阵
point: 待求k近邻的点矩阵
indices: 输出的k近邻矩阵
return dists: 输出的k近邻矩阵对应的距离
*/
void flann_search(const cv::Mat& data, const cv::Mat& point, 
	cv::Mat& indices, cv::Mat& dists, const int k);

/*利用k近邻距离矩阵,求每一行的平均
dists_knn_m: 样本间的k近邻距离
return samples_knn_dists_average_m:样本间k近邻距离的平均*/
cv::Mat cal_knn_dists_average(const cv::Mat&dists_knn_m,const int K);

/*利用原始数据计算样本点间距离矩阵储存以便使用(d(a,b) 为L1距离)*/
cv::Mat cal_samples_dists_m(const cv::Mat& data);
/*计算两样本sample1,sample2间的L1距离*/
float distance(const int sample1, const int sample2, const cv::Mat&data);
/*计算两类间的距离 min(d(a,b))*/
float distance(const int cluster1, const int cluster2, const std::vector<std::vector<int>>& cluster_vec, const cv::Mat samples_dists_m);

/*根据输入的数据量N生成初始的分类向量组,即每个样本一行且仅包含它自己
N:数据量*/
std::vector<std::vector<int>> init_cluster_vector(const int N);

/*根据输入样本间距离矩阵,与类向量组,得到类间距离矩阵,
samples_dists_m: 输入的样本间距离矩阵
cluster_vec: 类向量组
return cluster_dists_m: 类间向量矩阵*/
cv::Mat cal_cluster_dists_matrix(const cv::Mat samples_dists_m,
	const std::vector<std::vector<int>>& cluster_vec);

/*根据输入的距离矩阵,得到类间的k近邻矩阵,
并其重载形式可输出相应的k近邻距离矩阵,以便以后统一接口,不使用flann库
dists_m: 距离矩阵
return knn_m: k近邻矩阵
可选参数return& knn_dists_m: 用于返回的knn距离矩阵*/
cv::Mat cal_knn_m(const cv::Mat& dists_m,int&k);
cv::Mat cal_knn_m(const cv::Mat& dists_m, cv::Mat& knn_dists_m,int&k);

/*分别计算 DN与DR 并由此建立 类间关联矩阵,用于合并类
即:i,j两类满足合并要求则 merge_m(i,j)=1 
cluster_dists_m: 类间距离矩阵 在DN的d(Ci,Cj)使用
cluster_knn_m: 类间k近邻矩阵 在DR计算时使用
samples_knn_dists_average_m: 图像的k近邻距离平均,在就是DN的Φ的最后一个求和项
cluster_vec: 本次的类向量组,在计算DN的 Φ(Ci,Cj)时使用
return merge_m:类间关联矩阵,用于合并类
*/
cv::Mat cal_merge_matrix(const cv::Mat&cluster_dists_m, 
	const cv::Mat&cluster_knn_m, const cv::Mat&samples_knn_dists_average_m,
	 const std::vector<std::vector<int>>& cluster_vec);

/*利用merge_m也就是类间关联矩阵,BFS合并所有关联类
merge_m: 类间关联矩阵
return& cluster_vec:新的类向量组*/
std::vector<std::vector<int>> renew_clusters(const cv::Mat merge_m,
	const std::vector<std::vector<int>>& cluster_vec);
/*判断x在不在cluster_vec中,并返回行号,-1为没有找到*/
int is_in_clusters_vec(const int x, std::vector<std::vector<int>>& cluster_vec);

/*计算DR  (eq.2)  根据k近邻 计算 a,b 之间的 rank-order距离
cluster_knn_m: 类间k近邻矩阵 在DR计算时使用
return DR
*/
float cal_DR(const int a, const int b, const cv::Mat&cluster_knn_m);
/*计算DR矩阵*/
cv::Mat cal_DR_m(const cv::Mat&cluster_knn_m);
/*返回 a中第i个邻近在b中的序号,没有则返回indices.cols(k)的值
*/
int find_ai_in_b(const int a, const int b, const int i, const cv::Mat&indices);
/*计算DN
cluster_dists_m: 类间距离矩阵 在DN的d(Ci,Cj)使用
samples_knn_dists_average_m: 图像的k近邻距离平均,在就是DN的Φ的最后一个求和项
cluster_vec: 本次的类向量组,在计算DN的 Φ(Ci,Cj)时使用
return DN
*/
float cal_DN(const int i, const int j,
	const cv::Mat&cluster_dists_m,
	const cv::Mat&samples_knn_dists_average_m,
	const std::vector<std::vector<int>>& cluster_vec);

/*计算DN矩阵*/
cv::Mat cal_DN_m(const cv::Mat&cluster_dists_m,
	const cv::Mat&samples_knn_dists_average_m,
	const std::vector<std::vector<int>>& cluster_vec);

/*根据file_name_v与聚类后的cluster_vec得到cluster_file_name_v*/
std::vector<std::vector<std::string>> get_cluster_file_name_v(
	const std::vector<std::string>&file_name_v,
	const std::vector<std::vector<int>>cluster_vec);