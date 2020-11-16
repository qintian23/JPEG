#if 0
#include "opencv2/opencv.hpp"

int main()
{
	cv::Mat input = cv::imread("2.JPG", cv::IMREAD_GRAYSCALE);
	cv::imshow("step0_ori", input);
	int w = cv::getOptimalDFTSize(input.cols);
	int h = cv::getOptimalDFTSize(input.rows);
	cv::Mat padded;
	cv::copyMakeBorder(input, padded, 0, h - input.rows, 0, w - input.cols,
		cv::BORDER_CONSTANT, cv::Scalar::all(0));
	padded.convertTo(padded, CV_32FC1);
	cv::imshow("step1_padded", padded);
	for (int i = 0; i < padded.rows; i++)
	{
		float* ptr = padded.ptr<float>(i);
		for (int j = 0; j < padded.cols; j++)
			ptr[j] *= pow(-1, i + j);
	}
	cv::imshow("step2_center", padded);
	cv::Mat plane[] = { padded,cv::Mat::zeros(padded.size(),CV_32F) };
	cv::Mat complexImg;
	cv::merge(plane, 2, complexImg);
	cv::dft(complexImg, complexImg);
	cv::split(complexImg, plane);
	cv::magnitude(plane[0], plane[1], plane[0]);
	plane[0] += cv::Scalar::all(1);
	cv::log(plane[0], plane[0]);
	cv::normalize(plane[0], plane[0], 1, 0, cv::NORM_MINMAX);
	cv::imshow("dft", plane[0]);

	/****************************************************************************************/
	//插入
		////1.理想低通滤波
	//cv::Mat idealBlur(padded.size(), CV_32FC2);
	//double D0 = 60;
	//for (int i = 0; i < padded.rows; i++) {
	//	float* p = idealBlur.ptr<float>(i);
	//	for (int j = 0; j < padded.cols; j++) {
	//		double d = sqrt(pow((i - padded.rows / 2), 2) + pow((j - padded.cols / 2), 2));//分子,计算pow必须为float型
	//		if (d <= D0) {
	//			p[2 * j + 1] = 1;
	//			p[2 * j] = 1;
	//		}
	//		else {
	//			p[2 * j] = 0;
	//			p[2 * j + 1] = 0;
	//		}
	//	}
	//}
	//multiply(complexImg, idealBlur, idealBlur);
	//cv::idft(idealBlur, idealBlur);
	//cv::split(idealBlur, plane);
	////2.巴特沃斯低通滤波
	//cv::Mat butterworthBlur(padded.size(), CV_32FC2);
	//double D0 = 60;
	//int n = 20;
	//for (int i = 0; i < padded.rows; i++) {
	//	float* p = butterworthBlur.ptr<float>(i);
	//	for (int j = 0; j < padded.cols; j++) {
	//		double d = sqrt(pow((i - padded.rows / 2), 2) + pow((j - padded.cols / 2), 2));//分子,计算pow必须为float型
	//		p[2*j] = 1.0 / (1 + pow(d / D0, 2 * n));
	//		p[2*j+1] = 1.0 / (1 + pow(d / D0, 2 * n));
	//	}
	//}
	//multiply(complexImg, butterworthBlur, butterworthBlur);
	//cv::idft(butterworthBlur, butterworthBlur);
	//cv::split(butterworthBlur, plane);
	////3.高斯低通滤波
	//cv::Mat gaussianBlur(padded.size(), CV_32FC2);
	//float D0 = 2 * 10 * 10 ;
	//for (int i = 0; i < padded.rows; i++)
	//{
	//	float* p = gaussianBlur.ptr<float>(i);
	//	for (int j = 0; j < padded.cols; j++)
	//	{
	//		float d = pow(i - padded.rows / 2, 2) + pow(j - padded.cols / 2, 2);
	//		p[2 * j] = expf(-d / D0);
	//		p[2 * j + 1] = expf(-d / D0);
	//	}
	//}
	//multiply(complexImg, gaussianBlur, gaussianBlur);
	//cv::idft(gaussianBlur, gaussianBlur);
	//cv::split(gaussianBlur, plane);
	////4.理想高通滤波
	//cv::Mat idealBlur(padded.size(), CV_32FC2);
	//double D0 = 20;
	//for (int i = 0; i < padded.rows; i++) {
	//	float* p = idealBlur.ptr<float>(i);
	//	for (int j = 0; j < padded.cols; j++) {
	//		double d = sqrt(pow((i - padded.rows / 2), 2) + pow((j - padded.cols / 2), 2));//分子,计算pow必须为float型
	//		if (d <= D0) {
	//			p[2 * j + 1] = 0;
	//			p[2 * j] = 0;
	//		}
	//		else {
	//			p[2 * j] = 1;
	//			p[2 * j + 1] = 1;
	//		}
	//	}
	//}
	//multiply(complexImg, idealBlur, idealBlur);
	//cv::idft(idealBlur, idealBlur);
	//cv::split(idealBlur, plane);
	////5.巴特沃斯高通滤波
	//cv::Mat butterworthBlur(padded.size(), CV_32FC2);
	//double D0 = 20;
	//int n = 1;
	//for (int i = 0; i < padded.rows; i++) {
	//	float* p = butterworthBlur.ptr<float>(i);
	//	for (int j = 0; j < padded.cols; j++) {
	//		double d = sqrt(pow((i - padded.rows / 2), 2) + pow((j - padded.cols / 2), 2));//分子,计算pow必须为float型
	//		p[2*j] = 1.0 / (1 + pow(D0 / d, 2 * n));
	//		p[2*j+1] = 1.0 / (1 + pow(D0 / d, 2 * n));
	//		
	//	}
	//}
	//multiply(complexImg, butterworthBlur, butterworthBlur);
	//cv::idft(butterworthBlur, butterworthBlur);
	//cv::split(butterworthBlur, plane);
	////6.高斯高通滤波
	//cv::Mat gaussianBlur(padded.size(), CV_32FC2);
	//float D0 = 2 * 10 * 10;
	//for (int i = 0; i < padded.rows; i++)
	//{
	//	float* p = gaussianBlur.ptr<float>(i);
	//	for (int j = 0; j < padded.cols; j++)
	//	{
	//		float d = pow(i - padded.rows / 2, 2) + pow(j - padded.cols / 2, 2);
	//		p[2 * j] = 1 - expf(-d / D0);
	//		p[2 * j + 1] = 1 - expf(-d / D0);
	//	}
	//}
	//multiply(complexImg, gaussianBlur, gaussianBlur);
	//cv::idft(gaussianBlur, gaussianBlur);
	//cv::split(gaussianBlur, plane);
	////7.频率域拉普拉斯算子
	//cv::Mat Laplace(padded.size(), CV_32FC2);
	//for (int i = 0; i < padded.rows; i++)
	//{
	//	float* p = Laplace.ptr<float>(i);
	//	for (int j = 0; j < padded.cols; j++)
	//	{
	//		float d = pow(i - padded.rows / 2, 2) + pow(j - padded.cols / 2, 2);
	//		p[2 * j] = 1 + 4 * pow(CV_PI, 2) * d;
	//		p[2 * j + 1] = 1 + 4 * pow(CV_PI, 2) * d;
	//	}
	//}
	//multiply(complexImg, Laplace, Laplace);
	//cv::idft(Laplace, Laplace);
	//cv::split(Laplace, plane);
	//8.高斯的高频强调滤波
	cv::Mat gaussianBlur(padded.size(), CV_32FC2);
	float D0 = 2 * 10 * 10;
	for (int i = 0; i < padded.rows; i++)
	{
		float* p = gaussianBlur.ptr<float>(i);
		for (int j = 0; j < padded.cols; j++)
		{
			float d = pow(i - padded.rows / 2, 2) + pow(j - padded.cols / 2, 2);
			p[2 * j] = 0.5 + 0.75 * (1 - expf(-d / D0));
			p[2 * j + 1] = 0.5 + 0.75 * (1 - expf(-d / D0));
		}
	}
	multiply(complexImg, gaussianBlur, gaussianBlur);
	cv::idft(gaussianBlur, gaussianBlur);
	cv::split(gaussianBlur, plane);
	/****************************************************************************************/
	cv::magnitude(plane[0], plane[1], plane[0]);
	cv::normalize(plane[0], plane[0], 1, 0, cv::NORM_MINMAX);
	cv::imshow("lpf", plane[0]);

	cv::waitKey();
	return 0;
}
#endif