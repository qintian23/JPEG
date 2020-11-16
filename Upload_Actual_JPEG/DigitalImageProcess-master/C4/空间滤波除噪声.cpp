#if 0
#include "opencv2/opencv.hpp"

//均值滤波器
//实现1：只能实现掩膜尺寸3x3的滤波，没有对图像边界处理，算法无优化
void meanBlur1(cv::Mat& src, cv::Mat& dst)
{
	dst = cv::Mat::zeros(src.size(), src.type());
	for (size_t i = 1; i < src.rows - 1; i++)
	{
		for (size_t j = 1; j < src.cols - 1; j++)
		{
			dst.at<uchar>(i - 1, j - 1) = (src.at<uchar>(i - 1, j) + src.at<uchar>(i - 1, j + 1) +
				src.at<uchar>(i, j - 1) + src.at<uchar>(i, j) + src.at<uchar>(i, j + 1) +
				src.at<uchar>(i + 1, j - 1) + src.at<uchar>(i + 1, j) + src.at<uchar>(i + 1, j + 1)) / 9;
		}
	}
}
//实现2
void meanBlur2(cv::Mat& src, cv::Mat& dst, int ksize)
{
	dst = cv::Mat::zeros(src.size(), src.type());
	//确保为奇数
	if (ksize % 2 == 0)
		ksize += 1;
	int halfKsize = (ksize - 1) / 2;
	//扩充边缘像素
	cv::Mat matBorder;
	cv::copyMakeBorder(src, matBorder, halfKsize, halfKsize, halfKsize, halfKsize, cv::BORDER_REFLECT_101);
	//遍历除边界像素外的像素
	int sum = 0;
	for (size_t r = halfKsize; r < src.rows + halfKsize; r++) 
	{
		for (size_t c = halfKsize; c < src.cols + halfKsize; c++)
		{
			//掩膜覆盖的所有像素值
			for (size_t i = r - halfKsize; i <= r + halfKsize; i++) {
				for (size_t j = c - halfKsize; j <= c + halfKsize; j++)
					sum += matBorder.at<uchar>(i, j);
			}
			dst.at<uchar>(r - halfKsize, c - halfKsize) = sum / (ksize * ksize);
		}
	}
}
int main()
{

}
#endif
