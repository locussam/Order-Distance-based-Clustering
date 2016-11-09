#include"main.h"
using namespace std;
using namespace  cv;

Mat f(void)
{
	return Mat();
}
int main()
{
	Mat a = Mat::zeros(10, 10, CV_32FC1);
	Mat v;
	
	a.copyTo(v);v.at <float>(5, 5) = 2.6;

	int x, y;
	x = y = v.at <float>(5, 5);
	cout << x << endl;
	cout << y << endl;
}
