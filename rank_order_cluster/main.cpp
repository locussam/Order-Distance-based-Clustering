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
//������֤
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

	Mat samples_knn_m;//�������knn����
	Mat samples_dists_knn_m;//������knn�����Ӧ�ľ���
	Mat samples_dists_m = cal_samples_dists_m(data);//�����������L1����,�Ա����ʹ��
	flann_search(data, data, samples_knn_m, samples_dists_knn_m, k);//���������Ѿ�������������������ľ������Ҳ��cal_knn_m�õ�����������

	vector<vector<int>>cluster_vec = init_cluster_vector(data.rows);//��ʼ����������
	Mat samples_knn_dists_average_m = cal_knn_dists_average(samples_dists_knn_m);//Ϊÿ��������k�����ڵľ����ƽ��ֵ����
																			 //�������п��Է���ѭ������,��������Ϊ�˼���  ֱ�ӿ���
	Mat cluster_dists_m;//��ʼʱÿ����������Ϊһ��,��������������
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
