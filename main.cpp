#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

struct submatrix // 子块的存储
{
	int subM[8][8];
};

// 亮度量化表
int light_dct[8][8] =
{
	{16, 11, 10, 16, 24, 40, 51, 61 },
	{12, 12, 14, 19, 26, 58, 60, 55 },
	{14, 13, 16, 24, 40, 57, 69, 56 },
	{14, 17, 22, 29, 51, 87, 80, 62 },
	{18, 22, 37, 56, 68, 109, 103, 77 },
	{24, 35, 55, 64, 81, 104, 113, 92 },
	{49, 64, 78, 87, 103, 121, 120, 101 },
	{72, 92, 95, 98, 112, 100, 103, 99 }

};

int main()
{
	Mat image = imread("1.jpg",0);
	if (image.empty()) { cout << "Empty image!" << endl; return -1; }
	// imshow("srcimage", image);

	cout << image.rows << endl;
	cout << image.cols << endl;

	int n = 8;

	int rlen = image.rows & -n;
	int clen = image.cols & -n;

	Mat srcimage = image(Rect(0, 0, clen, rlen)); // 保证能分成8*8
	// imshow("image", srcimage);

	int subRows = rlen / n;
	int subCols = clen / n;

	/*for (int i = 0; i < subRows; i++)
	{
		for (int j = 0; j < subCols; j++)
		{
			for (int x = i*n; x < i * n + n; x++)
			{
				uchar* srr = srcimage.ptr<uchar>(x);
				for (int y = j*n; y < n + j * n; y++)
				{
					cout << (int)srr[y] << ' ';
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}*/

	Mat subItem(n, n, CV_32F);
	for (int i = 0; i < n; i++)
	{
		uchar* img = srcimage.ptr<uchar>(i);
		float* subimg = subItem.ptr<float>(i);
		for (int j = 0; j < n; j++)
		{
			subimg[j] = img[j];
		}
	}

	cout << subItem << endl << endl;

	// resize(image, image, Size(512, 512));
	// image.convertTo(image, CV_32F, 1.0 / 255);
	Mat srcDCT;
	dct(subItem, srcDCT);
	// imshow("dct", srcDCT);
	cout << srcDCT << endl << endl;

	// 量化
	Mat subF(n, n, CV_8UC1);
	for (int i = 0; i < n; i++)
	{
		uchar* img = subF.ptr<uchar>(i);
		float* subimg = subItem.ptr<float>(i);
		for (int j = 0; j < n; j++)
		{
			float item;
			item = subimg[j] / light_dct[i][j];  // 量化
			int iitem;
			iitem = (int)item > 0 ? (item + 0.5) : (item - 0.5);  // 四舍五入
			img[j] = iitem;
		}
	}
	cout << subF << endl;



	waitKey();
	return 0;
}