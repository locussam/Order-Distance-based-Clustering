#include "init_data.h"
using namespace std;
using namespace cv;


void get_data_and_name(cv::Mat&data, std::vector<std::string>&feature_name_v, const std::string path)
{
	int DIM = 1024;
	ifstream list_file(path + "LIST.TXT");
	string feature_name;
	ifstream feature_file;
	float temp;
	vector<float> temp_vec;
	feature_name_v.clear();
	while (getline(list_file, feature_name))
	{
		feature_name_v.push_back(feature_name);

		feature_file.open(path + feature_name);
		if (feature_file.is_open())
		{
			while (feature_file >> temp)
			{
				temp_vec.push_back(temp);
			}
			if (temp_vec.size() % DIM != 0)
			{
				cout << "wrong data" << feature_name << endl;
				exit(-1);
			}
		}
		feature_file.close();
	}

	int col = DIM;
	int row = temp_vec.size() / DIM;
	
	data = Mat::zeros(row, col, CV_32F);
	int temp_n = 0;
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j, ++temp_n)
		{
			data.at<float>(i, j) = temp_vec[temp_n];
		}
	}
}
