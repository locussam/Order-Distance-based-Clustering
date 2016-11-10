#include"main.h"
using namespace std;
using namespace  cv;

void print_cluster_vec(vector<vector<int>>& cluster_vec)
{
	for (int i=0;i<cluster_vec.size();++i)
	{
		for (int j=0;j<cluster_vec[i].size();++j)
		{
			cout << cluster_vec[i][j] << " ";
		}
		cout << endl;
	}
}
//数据验证
int main()
{
	Mat data(7, 2, CV_32F);
	data.at<float>(0, 0) = 0; data.at<float>(0, 1) = 0;
	data.at<float>(1, 0) = 0; data.at<float>(1, 1) = 1;
	data.at<float>(2, 0) = 1; data.at<float>(2, 1) = 1;
	data.at<float>(3, 0) = 10; data.at<float>(3, 1) = 2;
	data.at<float>(4, 0) = 10; data.at<float>(4, 1) = 1;
	data.at<float>(5, 0) = 10; data.at<float>(5, 1) = -1;
	data.at<float>(6, 0) = 1; data.at<float>(6, 1) = 0;

	Mat samples_knn_m;//样本间的knn矩阵
	Mat samples_dists_knn_m;//样本的knn矩阵对应的距离
	Mat samples_dists_m = cal_samples_dists_m(data);//计算各个点间的L1距离,以便后续使用
	flann_search(data, data, samples_knn_m, samples_dists_knn_m, k);//这里由于已经计算了样本点两两间的距离可以也用cal_knn_m得到另外两矩阵

	vector<vector<int>>cluster_vec = init_cluster_vector(data.rows);//初始化类向量组
	Mat samples_knn_dists_average_m = cal_knn_dists_average(samples_dists_knn_m);//为每个样本的k个近邻的距离的平均值向量
																			 //以下四行可以放入循环体中,在这里是为了加速  直接拷贝
	Mat cluster_dists_m;//初始时每个样本独自为一类,类间距就是样本间距
	samples_dists_knn_m.copyTo(cluster_dists_m);
	Mat cluster_knn_m;
	samples_knn_m.copyTo(cluster_knn_m);



	cluster_dists_m = cal_cluster_dists_matrix(samples_dists_m, cluster_vec);
	cluster_knn_m = cal_knn_m(cluster_dists_m);
	cout << cluster_knn_m << endl;
	//Mat merge_m = cal_merge_matrix(cluster_dists_m, cluster_knn_m, samples_knn_dists_average_m, cluster_vec);
	//renew_clusters(merge_m, cluster_vec);

	//for (int i = 0; i < 50; ++i)
	//{
	//	cluster_dists_m = cal_cluster_dists_matrix(samples_dists_m, cluster_vec);
	//	cluster_knn_m = cal_knn_m(cluster_dists_m);
	//	merge_m = cal_merge_matrix(cluster_dists_m, cluster_knn_m, samples_knn_dists_average_m, cluster_vec);
	//	renew_clusters(merge_m, cluster_vec);
	//}
	return 0;
}
