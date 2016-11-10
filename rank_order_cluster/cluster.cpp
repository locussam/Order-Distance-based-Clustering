#include "cluster.h"
using namespace cv;
using namespace std;

void generate_data(cv::Mat & data)
{
}

std::vector<std::vector<int>> cluster(const cv::Mat & data)
{
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
	Mat merge_m = cal_merge_matrix(cluster_dists_m, cluster_knn_m, samples_knn_dists_average_m, cluster_vec);
	renew_clusters(merge_m, cluster_vec);

	for (int i = 0; i < 50; ++i)
	{
		cluster_dists_m = cal_cluster_dists_matrix(samples_dists_m, cluster_vec);
		cluster_knn_m = cal_knn_m(cluster_dists_m);
		merge_m = cal_merge_matrix(cluster_dists_m, cluster_knn_m, samples_knn_dists_average_m, cluster_vec);
		renew_clusters(merge_m, cluster_vec);
	}
	return cluster_vec;
}

void flann_search(const cv::Mat & data, const cv::Mat & point, cv::Mat & indices, cv::Mat & dists, const int k)
{
	flann::Index flannIndex(data, flann::AutotunedIndexParams(), cvflann::FLANN_DIST_L1);
	flannIndex.knnSearch(point, indices, dists, k, flann::SearchParams(64));
}

cv::Mat cal_knn_dists_average(const cv::Mat& dists_knn_m)
{
	Mat samples_knn_dists_average_m = Mat::zeros(dists_knn_m.rows, 1, CV_32F);
	int k = dists_knn_m.cols;
	for (int i = 0;i<dists_knn_m.rows;++i)
	{
		for (int j = 0;j<dists_knn_m.cols;++j)
		{
			samples_knn_dists_average_m.at<float>(i,0) += dists_knn_m.at<float>(i, j) / k;
		}
	}
	return samples_knn_dists_average_m;
}

cv::Mat cal_samples_dists_m(const cv::Mat data)
{
	Mat samples_dists_m = Mat::zeros(data.rows, data.rows, CV_32F);
	for (int i=0;i<data.rows;++i)
	{
		for (int j=0;j<i;++j)//利用对称性,同时对角线已初始化为0
		{
			samples_dists_m.at<float>(j, i) = samples_dists_m.at<float>(i, j) = distance_L1(i, j, data);
		}
	}
	return samples_dists_m;
}

float distance_L1(const int sample1, const int sample2, const cv::Mat&data)
{
	float distance = 0;
	for (int col = 0;col<data.cols;++col)
	{
		distance += abs(data.at<float>(sample1, col) - data.at<float>(sample2, col));
	}
	return distance;
}

float distance_L1(const int cluster1, const int cluster2, const std::vector<std::vector<int>>& cluster_vec, const cv::Mat samples_dists_m)
{
	if (cluster_vec[cluster1].empty() || cluster_vec[cluster2].empty())
	{
		cout << "wrong cluster_vec\n" << endl;
		exit(-1);
	}
	float distance = samples_dists_m.at<float>(cluster_vec[cluster1][0], cluster_vec[cluster2][0]);
	for (size_t i = 0;i<cluster_vec[cluster1].size();++i)
	{
		for (size_t j = 0;j<cluster_vec[cluster2].size();++j)
		{
			if (samples_dists_m.at<float>(cluster_vec[cluster1][i], cluster_vec[cluster2][j]) < distance)
			{
				distance = samples_dists_m.at<float>(cluster_vec[cluster1][i], cluster_vec[cluster2][j]);
			}
		}
	}
	return distance;
}

std::vector<std::vector<int>> init_cluster_vector(const int N)
{
	vector<int> temp_vec;
	vector<vector<int>> init_vec;
	for (int i = 0; i < N; ++i)
	{
		temp_vec.push_back(i);
		init_vec.push_back(temp_vec);
		temp_vec.clear();
	}
	return init_vec;
}

cv::Mat cal_cluster_dists_matrix(const cv::Mat samples_dists_m, const std::vector<std::vector<int>>& cluster_vec)
{
	const int cluster_num = cluster_vec.size();
	Mat cluster_dists_m = Mat::zeros(cluster_num, cluster_num, CV_32F);

	for (int i = 0; i < cluster_num; ++i)
	{
		for (int j = 0; j < i; ++j)//利用对称性,同时对角线已初始化为0
		{
			cluster_dists_m.at<float>(j, i) = cluster_dists_m.at<float>(i, j) = distance_L1(i, j, cluster_vec, samples_dists_m);
		}
	}
	return cluster_dists_m;
}

cv::Mat cal_knn_m(const cv::Mat & dists_m)
{
	Mat knn_m(dists_m.rows, k, CV_32S);
	struct Node
	{
		int index;
		float distance;
		Node& operator =(Node& value)
		{
			distance = value.distance;
			index = value.index;
			return *this;
		}

		bool operator ==(Node& value)
		{
			return (distance == value.distance);
		}
		bool operator >(Node& value)
		{
			return (distance > value.distance);
		}
		bool operator <(Node& value)
		{
			return(distance < value.distance);
		}
	};
	Node temp_node;
	vector<Node> temp_vec;
	for (int i = 0; i < dists_m.rows; ++i)
	{
		for (int j = 0; j < dists_m.cols; ++j)
		{
			temp_node.index = j;
			temp_node.distance = dists_m.at<float>(i, j);
			temp_vec.push_back(temp_node);
		}
		sort(temp_vec.begin(), temp_vec.end());
		for (int j = 0; j < k; ++j)
		{
			knn_m.at<int>(i, j) = temp_vec[j].index;
		}
		temp_vec.clear();
	}
	return knn_m;
}

cv::Mat cal_knn_m(const cv::Mat& dists_m, cv::Mat & knn_dists_m)
{
	Mat knn_m(dists_m.rows, k, CV_32S);
	knn_dists_m = Mat::zeros(dists_m.rows, k, CV_32F);
	struct Node
	{
		int index;
		float distance;
		Node& operator =(Node& value)
		{
			distance = value.distance;
			index = value.index;
			return *this;
		}

		bool operator ==(Node& value)
		{
			return (distance == value.distance);
		}
		bool operator >(Node& value)
		{
			return (distance > value.distance);
		}
		bool operator <(Node& value)
		{
			return(distance < value.distance);
		}
	};
	Node temp_node;
	vector<Node> temp_vec;
	for (int i=0;i<dists_m.rows;++i)
	{
		for (int j=0;j <dists_m.cols;++j)
		{
			temp_node.index = j;
			temp_node.distance = dists_m.at<float>(i, j);
			temp_vec.push_back(temp_node);
		}
		sort(temp_vec.begin(), temp_vec.end());
		for (int j = 0;j<k;++j)
		{
			knn_m.at<int>(i, j) = temp_vec[j].index;
			knn_dists_m.at<float>(i, j) = temp_vec[j].distance;
		}
		temp_vec.clear();
	}
	return knn_m;
}


cv::Mat cal_merge_matrix(const cv::Mat & cluster_dists_m, const cv::Mat & cluster_knn_m, const cv::Mat & samples_knn_dists_average_m, const std::vector<std::vector<int>>& cluster_vec)
{
	Mat merge_m = Mat::zeros(cluster_vec.size(), cluster_vec.size(), CV_32SC1);
	for (int i = 0; i < merge_m.rows; ++i)
	{
		for (int j = 0; j < merge_m.cols; ++j)
		{
			if (cal_DR(cluster_knn_m) < t &&
				cal_DN(cluster_dists_m, samples_knn_dists_average_m,
					cluster_vec) < 1)
			{
				merge_m.at<int>(i, j) = 1;
				//merge_m.at<int>(j, i) = 1;
			}
		}
	}
	return cv::Mat();
}

void renew_clusters(const cv::Mat merge_m, std::vector<std::vector<int>>& cluster_vec)
{
}

float cal_DR(const cv::Mat&cluster_knn_m)
{
	float DR = 0;
	return DR;
}

float cal_DN(const cv::Mat&cluster_dists_m,
	const cv::Mat&samples_knn_dists_average_m,
	const std::vector<std::vector<int>>& cluster_vec)
{
	float DN = 0;
	return DN;
}

