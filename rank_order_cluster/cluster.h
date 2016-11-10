#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "global.h"

/*��������,
data:ÿ��Ϊһ������,����Ϊ������ά��*/
void generate_data(cv::Mat& data);

/*����
data:�������ݾ���
return clusters:���������,������ÿ��Ϊһ����*/
std::vector<std::vector<int>> cluster(const cv::Mat&data);

/*����FLANN��õ����ݵĳ�ʼk���ھ�����������
data: �������������
point: ����k���ڵĵ����
indices: �����k���ھ���
return dists: �����k���ھ����Ӧ�ľ���
*/
void flann_search(const cv::Mat& data, const cv::Mat& point, 
	cv::Mat& indices, cv::Mat& dists, const int k);

/*����k���ھ������,��ÿһ�е�ƽ��
dists_knn_m: �������k���ھ���
return samples_knn_dists_average_m:������k���ھ����ƽ��*/
cv::Mat cal_knn_dists_average(const cv::Mat&dists_knn_m);

/*����ԭʼ���ݼ���������������󴢴��Ա�ʹ��(d(a,b) ΪL1����)*/
cv::Mat cal_samples_dists_m(const cv::Mat data);
/*����������sample1,sample2���L1����*/
float distance_L1(const int sample1, const int sample2, const cv::Mat&data);
/*���������ľ��� min(d(a,b))*/
float distance_L1(const int cluster1, const int cluster2, const std::vector<std::vector<int>>& cluster_vec, const cv::Mat samples_dists_m);

/*���������������N���ɳ�ʼ�ķ���������,��ÿ������һ���ҽ��������Լ�
N:������*/
std::vector<std::vector<int>> init_cluster_vector(const int N);

/*��������������������,����������,�õ����������,
samples_dists_m: �����������������
cluster_vec: ��������
return cluster_dists_m: �����������*/
cv::Mat cal_cluster_dists_matrix(const cv::Mat samples_dists_m,
	const std::vector<std::vector<int>>& cluster_vec);

/*��������ľ������,�õ�����k���ھ���,
����������ʽ�������Ӧ��k���ھ������,�Ա��Ժ�ͳһ�ӿ�,��ʹ��flann��
dists_m: �������
return knn_m: k���ھ���
��ѡ����return& knn_dists_m: ���ڷ��ص�knn�������*/
cv::Mat cal_knn_m(const cv::Mat& dists_m);
cv::Mat cal_knn_m(const cv::Mat& dists_m, cv::Mat& knn_dists_m);

/*�ֱ���� DN��DR ���ɴ˽��� ����������,���ںϲ���
��:i,j��������ϲ�Ҫ���� merge_m(i,j)=1 
cluster_dists_m: ��������� ��DN��d(Ci,Cj)ʹ��
cluster_knn_m: ���k���ھ��� ��DR����ʱʹ��
samples_knn_dists_average_m: ͼ���k���ھ���ƽ��,�ھ���DN�Ħ������һ�������
cluster_vec: ���ε���������,�ڼ���DN�� ��(Ci,Cj)ʱʹ��
return merge_m:����������,���ںϲ���
*/
cv::Mat cal_merge_matrix(const cv::Mat&cluster_dists_m, 
	const cv::Mat&cluster_knn_m, const cv::Mat&samples_knn_dists_average_m,
	 const std::vector<std::vector<int>>& cluster_vec);

/*����merge_mҲ��������������,BFS�ϲ����й�����
merge_m: ����������
return& cluster_vec:�µ���������*/
void renew_clusters(const cv::Mat merge_m, 
	std::vector<std::vector<int>>& cluster_vec);

/*����DR
cluster_knn_m: ���k���ھ��� ��DR����ʱʹ��
return DR
*/
float cal_DR(const cv::Mat&cluster_knn_m);

/*����DN
cluster_dists_m: ��������� ��DN��d(Ci,Cj)ʹ��
samples_knn_dists_average_m: ͼ���k���ھ���ƽ��,�ھ���DN�Ħ������һ�������
cluster_vec: ���ε���������,�ڼ���DN�� ��(Ci,Cj)ʱʹ��
return DN
*/
float cal_DN(const cv::Mat&cluster_dists_m, 
	const cv::Mat&samples_knn_dists_average_m,
	const std::vector<std::vector<int>>& cluster_vec);