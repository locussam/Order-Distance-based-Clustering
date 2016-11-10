#include"main.h"
using namespace std;
using namespace  cv;

void print_cluster_vec(vector<vector<int>>& cluster_vec)
{
	cout << "[";
	for (size_t i = 0; i < cluster_vec.size(); ++i)
	{
		for (size_t j = 0; j < cluster_vec[i].size(); ++j)
		{
			cout << cluster_vec[i][j] << " ";
		}
		cout << ";" << endl;
	}
	cout << "]" << endl;
}
//数据验证
//int main()
//{
//	Mat data(7, 2, CV_32F);
//	data.at<float>(0, 0) = 0; data.at<float>(0, 1) = 0;
//	data.at<float>(1, 0) = 0; data.at<float>(1, 1) = 1;
//	data.at<float>(2, 0) = 1; data.at<float>(2, 1) = 1;
//	data.at<float>(3, 0) = 10; data.at<float>(3, 1) = 2;
//	data.at<float>(4, 0) = 10; data.at<float>(4, 1) = 1;
//	data.at<float>(5, 0) = 10; data.at<float>(5, 1) = -1;
//	data.at<float>(6, 0) = 1; data.at<float>(6, 1) = 0;
//		int k = 4;//这里的k只是最开始计算全部N的数据时为了减轻运算量用的k,
//		//当类合并到 k以下时,会在cal_knn_m中改变k,使k=类数
//		const int K = 4;
//	Mat samples_knn_m;//样本间的knn矩阵
//	Mat samples_dists_knn_m;//样本的knn矩阵对应的距离
//	Mat samples_dists_m = cal_samples_dists_m(data);//计算各个点间的L1距离,以便后续使用
//	samples_knn_m = cal_knn_m(samples_dists_m, samples_dists_knn_m, k);//这里由于已经计算了样本点两两间的距离可以也用cal_knn_m得到另外两矩阵
//
//	vector<vector<int>>cluster_vec = init_cluster_vector(data.rows);//初始化类向量组
//	Mat samples_knn_dists_average_m = cal_knn_dists_average(samples_dists_knn_m,K);//为每个样本的k个近邻的距离的平均值向量
//																			 //以下四行可以放入循环体中,在这里是为了加速  直接拷贝
//	Mat cluster_dists_m;//初始时每个样本独自为一类,类间距就是样本间距
//	samples_dists_knn_m.copyTo(cluster_dists_m);
//	Mat cluster_knn_m;
//	samples_knn_m.copyTo(cluster_knn_m);
//
//
//
//	cluster_dists_m = cal_cluster_dists_matrix(samples_dists_m, cluster_vec);
//	cluster_knn_m = cal_knn_m(cluster_dists_m,k);
//	cout << samples_knn_dists_average_m << endl;
//	cout << cal_DR(0, 1, cluster_knn_m) << endl;
//
//	cout << cal_DN(0, 5, cluster_dists_m, samples_knn_dists_average_m, cluster_vec);
//	//fai = (1+3.75)/(1+1)   DN = 11/fai; DN = 4.631579
//	Mat merge_m = cal_merge_matrix(cluster_dists_m, cluster_knn_m, samples_knn_dists_average_m, cluster_vec);
//	cout << merge_m << endl;
//	print_cluster_vec(cluster_vec);
//	cluster_vec = renew_clusters(merge_m, cluster_vec);
//	print_cluster_vec(cluster_vec);
//	//renew_clusters(merge_m, cluster_vec);
//
//	//for (int i = 0; i < 50; ++i)
//	//{
//	//	cluster_dists_m = cal_cluster_dists_matrix(samples_dists_m, cluster_vec);
//	//	cluster_knn_m = cal_knn_m(cluster_dists_m);
//	//	merge_m = cal_merge_matrix(cluster_dists_m, cluster_knn_m, samples_knn_dists_average_m, cluster_vec);
//	//	renew_clusters(merge_m, cluster_vec);
//	//}
//	return 0;
//}
int main()
{

	Mat data;
	vector<string> feature_name_v;
	get_data_and_name(data, feature_name_v);
	cout << "finish data in" << endl;
	int k = 20;//这里的k只是最开始计算全部N的数据时为了减轻运算量用的k,
	//当类合并到 k以下时,会在cal_knn_m中改变k,使k=类数
	const int K = 9;


	Mat samples_knn_m;//样本间的knn矩阵
	Mat samples_dists_knn_m;//样本的knn矩阵对应的距离
	Mat samples_dists_m = cal_samples_dists_m(data);//计算各个点间的L1距离,以便后续使用
	cout << "finish cal samples_data" << endl;
	samples_knn_m = cal_knn_m(samples_dists_m, samples_dists_knn_m, k);//这里由于已经计算了样本点两两间的距离可以也用cal_knn_m得到另外两矩阵
	vector<vector<int>>cluster_vec = init_cluster_vector(data.rows);//初始化类向量组
	int pre_it_size = cluster_vec.size();

	Mat samples_knn_dists_average_m = cal_knn_dists_average(samples_dists_knn_m,K);//为每个样本的k个近邻的距离的平均值向量
																		 //以下四行可以放入循环体中,在这里是为了加速  直接拷贝
	Mat cluster_dists_m;//初始时每个样本独自为一类,类间距就是样本间距
	samples_dists_m.copyTo(cluster_dists_m);
	Mat cluster_knn_m;
	samples_knn_m.copyTo(cluster_knn_m);

	Mat merge_m = cal_merge_matrix(cluster_dists_m, cluster_knn_m, samples_knn_dists_average_m, cluster_vec);
	cluster_vec = renew_clusters(merge_m, cluster_vec);
	cout << "start merge" << endl;

	int temp_itor = 0;
	while (pre_it_size != cluster_vec.size()  || temp_itor < 10)
	{
		pre_it_size = cluster_vec.size();
		cluster_dists_m = cal_cluster_dists_matrix(samples_dists_m, cluster_vec);
		cluster_knn_m = cal_knn_m(cluster_dists_m, k);
		merge_m = cal_merge_matrix(cluster_dists_m, cluster_knn_m, samples_knn_dists_average_m, cluster_vec);
		cluster_vec = renew_clusters(merge_m, cluster_vec);
		cout << temp_itor++ << endl;
	}
	print_cluster_vec(cluster_vec);

	////这个为可以控制最后并几个类出来的版本
	//while (pre_it_size != cluster_vec.size())
	//{
	//	pre_it_size = cluster_vec.size();
	//	cluster_dists_m = cal_cluster_dists_matrix(samples_dists_m, cluster_vec);
	//	cluster_knn_m = cal_knn_m(cluster_dists_m, k);
	//	merge_m = cal_merge_matrix(cluster_dists_m, cluster_knn_m, samples_knn_dists_average_m, cluster_vec);
	//	auto temp_vec = renew_clusters(merge_m, cluster_vec);
	//	if (temp_vec.size() < 2)
	//	{
	//		break;
	//	}
	//	else
	//	{
	//		cluster_vec = temp_vec;
	//	}
	//	print_cluster_vec(cluster_vec);
	//}

	auto cluster_name_v = get_cluster_file_name_v(feature_name_v, cluster_vec);
	for (int i = 0; i < cluster_vec.size(); ++i)
	{
		for (int j = 0; j < cluster_vec[i].size(); ++j)
		{
			cout << cluster_name_v[i][j] <<" ";
		}
		cout << endl;
	}
}
