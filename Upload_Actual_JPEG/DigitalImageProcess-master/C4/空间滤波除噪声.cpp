#if 0
#include "opencv2/opencv.hpp"

//��ֵ�˲���
//ʵ��1��ֻ��ʵ����Ĥ�ߴ�3x3���˲���û�ж�ͼ��߽紦���㷨���Ż�
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
//ʵ��2
void meanBlur2(cv::Mat& src, cv::Mat& dst, int ksize)
{
	dst = cv::Mat::zeros(src.size(), src.type());
	//ȷ��Ϊ����
	if (ksize % 2 == 0)
		ksize += 1;
	int halfKsize = (ksize - 1) / 2;
	//�����Ե����
	cv::Mat matBorder;
	cv::copyMakeBorder(src, matBorder, halfKsize, halfKsize, halfKsize, halfKsize, cv::BORDER_REFLECT_101);
	//�������߽������������
	int sum = 0;
	for (size_t r = halfKsize; r < src.rows + halfKsize; r++) 
	{
		for (size_t c = halfKsize; c < src.cols + halfKsize; c++)
		{
			//��Ĥ���ǵ���������ֵ
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
