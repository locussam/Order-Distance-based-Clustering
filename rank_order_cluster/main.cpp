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
int main()
{

	Mat data;
	vector<string> feature_name_v;
	get_data_and_name(data, feature_name_v);
	cout << "finish data in" << endl;
	int k = 20;//�����kֻ���ʼ����ȫ��N������ʱΪ�˼����������õ�k,
	//����ϲ��� k����ʱ,����cal_knn_m�иı�k,ʹk=����
	const int K = 20;
	std::vector<std::vector<int>> cluster_vec;
	std::vector<std::vector<std::string>> cluster_name_v;
	cluster(data, feature_name_v, k, K, cluster_vec, cluster_name_v);
	print_cluster_vec(cluster_vec);


}
