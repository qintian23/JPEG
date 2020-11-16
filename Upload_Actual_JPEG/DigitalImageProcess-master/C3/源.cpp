#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
using namespace std;
using namespace cv;
/*���Ա仯֮�Ҷȼ���ת*/
void grayInv()
{
	Mat srcImg = imread("test.PNG", 0);
	if (!srcImg.data)
	{
		cout << "fail to load image" << endl;
		return;
	}
	int k = -1, b = 255;
	int rowNum = srcImg.rows;
	int colNum = srcImg.cols;
	Mat dstImg(srcImg.size(), srcImg.type());
	for (int i = 0; i < rowNum; i++)
	{
		uchar* srcData = srcImg.ptr<uchar>(i);
		for (int j = 0; j < colNum; j++) {
			dstImg.at<uchar>(i, j) = srcData[j] * k + b;
			//dstImg.at<uchar>(i, j)=log(double(1 + (double)srcImg.at<uchar>(i, j)));
		}
	}
	//// ��һ������
	//cv::normalize(dstImg, dstImg,
	//	0, 255, cv::NORM_MINMAX);
	//cv::convertScaleAbs(dstImg, dstImg);

	imshow("original", srcImg);
	imshow("grayInv", dstImg);
	waitKey(0);
}
void GammaCorr()
{
	Mat srcImg = imread("test.PNG");
	if (srcImg.empty())
	{
		cout << "fail to load" << endl;
		return;
	}
	float fGamma = 1 / 3.2;
	//����������ļ�LUT
	unsigned char lut[256];
	for (int i = 0; i < 256; i++) {
		//��ֹ��ɫ���������255����������Ϊ255��С��0����������Ϊ0
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}
	Mat dstImg = srcImg.clone();
	const int channels = dstImg.channels();
	switch (channels)
	{
	case 1://�Ҷ�ͼ
	{
		MatIterator_<uchar> it, end;
		for (it = dstImg.begin<uchar>(), end = dstImg.end<uchar>(); it != end; it++)
			//*it = pow((float)(((*it)) / 255.0), fGamma) * 255.0;
			*it = lut[(*it)];
		break;
	}
	case 3://��ɫͼ 
	{
		MatIterator_<Vec3b> it, end;
		for (it = dstImg.begin<Vec3b>(), end = dstImg.end<Vec3b>(); it != end; it++)
		{
			//(*it)[0] = pow((float)(((*it)[0])/255.0), fGamma) * 255.0; 
			//(*it)[1] = pow((float)(((*it)[1])/255.0), fGamma) * 255.0; 
			//(*it)[2] = pow((float)(((*it)[2])/255.0), fGamma) * 255.0; 
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}
		break;
	}
	}
	imshow("ori", srcImg);
	imshow("res", dstImg);
	waitKey(0);
}

void contrastStretch()
{
	Mat srcOri = imread("test1.JPG");
	Mat srcImage;
	cvtColor(srcOri, srcImage, CV_RGB2GRAY);
	//"=";"clone()";"copyTo"���ֿ�����ʽ��ǰ����ǳ�������������������
	Mat resultImage = srcImage.clone();
	int nRows = resultImage.rows;
	int nCols = resultImage.cols;
	//�ж�ͼ��洢�������ԣ����������Եõ����ظ���
	if (resultImage.isContinuous())
	{
		nCols = nCols * nRows;
		nRows = 1;
	}
	//ͼ��ָ�����
	uchar* pDataMat;
	int pixMax = 0, pixMin = 255;
	//����ͼ��������Сֵ
	for (int j = 0; j < nRows; j++)
	{
		//ptr<>()�õ�����һ��ָ�룬����ָ��+ģ����
		pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++)
		{
			if (pDataMat[i] > pixMax)
				pixMax = pDataMat[i];
			if (pDataMat[i] < pixMin)
				pixMin = pDataMat[i];
		}
	}
	//�Աȶ�����ӳ�䣬��ԭʼ��Χ���쵽num1~num2
	int num1 = 100, num2 = 200;
	for (int j = 0; j < nRows; j++)
	{
		pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++)
		{
			pDataMat[i] = (pDataMat[i] - pixMin) * (num2 - num1) / (pixMax - pixMin) + num1;
		}
	}
	imshow("ori", srcImage);
	imshow("dst", resultImage);
	waitKey(0);
}
void bitLevel()
{
	Mat srcImage = imread("test2.JPG", 0);
	Mat d[8];
	int b[8];
	for (int k = 0; k < 8; k++)
		//CV_8UC1������8�����������0~255��U�����޷������ͣ�F�������ȸ����ͣ�
		//C����ͨ������1����Ҷ�ͼ�񼴵�ͨ����2����RGB��ɫͼ����ͨ����3����
		//��Alphaͨ����͸���ȣ���RGBͼ�񣬼���ͨ��
		d[k].create(srcImage.size(), CV_8UC1);
	int rowNum = srcImage.rows, colNum = srcImage.cols;
	for (int i = 0; i < rowNum; i++)
		for (int j = 0; j < colNum; j++) {
			int num = srcImage.at<uchar>(i, j);
			//
			for (int p = 0; p < 8; p++)
				b[p] = 0;
			int q = 0;
			while (num != 0)
			{
				b[q] = num % 2;
				num = num / 2;
				q++;
			}
			//
			for (int k = 0; k < 8; k++)
				d[k].at<uchar>(i, j) = b[k] * 255;
		}
	imshow("ori", srcImage);
	for (int k = 0; k < 8; k++)
		imshow("bit" + to_string(1 + k), d[k]);
	waitKey(0);
}
// 1.���ÿ⺯��
// ��ɫͼ����Ҫ��ͨ�����⻯���Ҷ�ͼ��ֱ�ӵ��ü���
Mat equalizeHistMine(Mat srcImage);
void equalizeHistOpencv()
{
	Mat srcImage = imread("DJI_0221.JPG", 1);
	Mat dstImage;
	if (!srcImage.data)
	{
		cout << "fail to load" << endl;
		return;
	}
	Mat channels[3];
	split(srcImage, channels);
	for (int i = 0; i < 3; i++)
	{
		equalizeHist(channels[i], channels[i]);
		//channels[i] = equalizeHistMine(channels[i]);
	}
	merge(channels, 3, dstImage);
	imshow("ori", srcImage);
	imshow("dst", dstImage);
	waitKey(0);
}
// 2. �Լ�ʵ��
Mat equalizeHistMine(Mat srcImage)
{
	int gray[256] = { 0 };
	double gray_prob[256] = { 0 };
	double gray_disSum[256] = { 0 };
	int gray_equal[256] = { 0 };
	Mat dstImage = srcImage.clone();
	int gray_sum = srcImage.cols * srcImage.rows;
	//ͳ��ÿ���Ҷ��µ����ظ���
	for (int i = 0; i < srcImage.rows; i++)
	{
		uchar* p = srcImage.ptr<uchar>(i);
		for (int j = 0; j < srcImage.cols; j++)
		{
			int value = p[j];
			gray[value]++;
		}
	}
	//ͳ�ƻҶ�Ƶ��
	for (int i = 0; i < 256; i++)
	{
		gray_prob[i] = ((double)gray[i] / gray_sum);
	}
	//�����ۼ��ܶ�
	gray_disSum[0] = gray_prob[0];
	for (int i = 1; i < 256; i++)
	{
		gray_disSum[i] = gray_disSum[i - 1] + gray_prob[i];
	}
	//���¼�����⻯�ĻҶ�ֵ��
	for (int i = 0; i < 256; i++)
	{
		gray_equal[i] = (uchar)(255 * gray_disSum[i] + 0.5);
	}
	//����
	for (int i = 0; i < dstImage.rows; i++)
	{
		uchar* p = dstImage.ptr<uchar>(i);
		for (int j = 0; j < dstImage.cols; j++)
		{
			p[j] = gray_equal[p[j]];
		}
	}
	return dstImage;
}
void hisMatch()
{
	Mat srcImage = imread("test4.JPG", 0);
	float zHist[256];
	for (int i = 0; i < 256; i++)
	{
		if (i < 128)
			zHist[i] = 1.5 / 256;
		else
			zHist[i] = 0.5 / 256;
	}
	Mat dstImage;
	srcImage.copyTo(dstImage);
	int h = srcImage.rows;
	int w = srcImage.cols;
	int hist[256] = { 0 };//����������Ŀ
	int S[256] = { 0 };
	map<int, int> S2Z;//S�����⻯�Ҷȣ���Z�����ͼ��Ҷȣ���ӳ��
	map<int, int> R2Z;//R��ԭʼͼ��Ҷȣ���Z�����ͼ��Ҷȣ���ӳ��
	//ֱ��ͼͳ��
	for (int i = 0; i < h; i++)
	{
		uchar* p = srcImage.ptr<uchar>(i);
		for (int j = 0; j < w; j++)
		{
			int value = p[j];
			hist[value]++;
		}
	}
	//��һ���ۼ�ֱ��ͼ
	float sumHist[256] = { 0 };
	for (int i = 0; i < 256; i++)
	{
		int sum = 0;
		for (int j = 0; j <= i; j++)
			sum += hist[j];
		sumHist[i] = sum * 1.0 / (h * w);
	}
	//����sumHist�������⻯��ĻҶȼ�����S
	for (int i = 0; i < 256; i++)
		S[i] = 255 * sumHist[i] + 0.5;
	//����zSumHist�������⻯��Ҷȼ�����G
	int G[256] = { 0 };
	float zSumHist[256] = { 0.0 };
	for (int i = 0; i < 256; i++) {
		float sum = 0;
		for (int j = 0; j <= i; j++)
			sum += zHist[j];
		zSumHist[i] = sum;
	}
	for (int i = 0; i < 256; i++)
		G[i] = zSumHist[i] * 255 + 0.5;

	//��G(Z)=S ����S->Z��ӳ���
	for (int i = 0; i < 256; i++) {
		for (int j = 1; j < 256; j++) {
			//G[i]������ֻ������������ж���������Ϊ��ӽ�
			if (abs(S[i] - G[j - 1]) < abs(S[i] - G[j]))
			{
				S2Z[S[i]] = j - 1;
				break;
			}
		}
	}
	S2Z[S[255]] = 255;
	//����R->Z��ӳ��
	for (int i = 0; i < 256; i++)
		R2Z[i] = S2Z[S[i]];
	//�ؽ�ͼ��
	for (int i = 0; i < h; i++)
	{
		uchar* pdata = dstImage.ptr<uchar>(i);
		for (int j = 0; j < w; j++)
			*(pdata + j) = R2Z[*(pdata + j)];
	}
	imshow("ori", srcImage);
	imshow("dst", dstImage);
	waitKey(0);
}
void allFilters()
{
	Mat srcImage = imread("test5.JPG");
	Mat boxfliter, mblur, gaussianblur, medianblur, bilateral;
	/* �����˲������˲�������Ӧ��ģ����������ֵ��ƽ��ֵ��Ҳ���������ᵽ��ģ��ϵ��ȫΪ1�ĺ�װ�˲���
	C++: void boxFilter(InputArray src, OutputArray dst, int ddepth, Size ksize, Point anchor=Point(-1,-1), bool normalize=true, int borderType=BORDER_DEFAULT )
	* src������ͼ��
	* dst�����ͼ��������ͼ��ȴ�
	* ddepth�����ͼ�����ȣ���Ϊ-1���������ͼ����ͬ
	* ksize���˲���ģ��ߴ�
	* anchor��ê�㼴��ƽ�����Ǹ��㣬Ĭ�ϣ�-1��-1������ê�����˲��˵�����
	* normalize��Ĭ��Ϊtrue����ʾ���Ƿ��������һ��
	* borderType�������ƶ�ͻϮ�Ǹ��ⲿ���ص�ĳ�ֱ߽�ģʽ��һ�㲻ͬ��
	*/
	boxFilter(srcImage, boxfliter, -1, Size(7, 7));
	/*��ֵ�˲����൱�ڵ���normalize=true�ķ����˲�һ����ֻ����������Щ��Ĳ�ͬ��ģ��ϵ����Ϊ1
	* src������ͼ�񣬿����������ͨ������������ȱ���ΪCV_8U, CV_16U, CV_16S, CV_32F �� CV_64F.
	*/
	blur(srcImage, mblur, Size(7, 7));
	/* ��˹�˲����൱��ģ��ϵ���ɸ�˹�����������ɵļ�Ȩƽ���˲�����˹�˲�����һ������ƽ���˲�����
		���˲�����ģ���ǶԶ�ά��˹������ɢ�õ������ڸ�˹ģ�������ֵ��������𽥼�С�����˲���
		�Ľ������ھ�ֵ�˲�����˵���á���˹�˲�������Ҫ�Ĳ������Ǹ�˹�ֲ��ı�׼���׼��͸�˹
		�˲�����ƽ�������кܴ��������Խ�󣬸�˹�˲�����Ƶ���ͽϿ���ͼ���ƽ���̶Ⱦ�Խ�á�ͨ��
		���ڲ���������ƽ���ͼ������������ƺͶ�ͼ���ģ����
	* sigmaX����ʾ��˹�˺�����X����ĵı�׼ƫ��
	* sigmaY����ʾ��˹�˺�����Y����ĵı�׼ƫ���sigmaYΪ�㣬�ͽ�����ΪsigmaX��
		���sigmaX��sigmaY����0����ô����ksize.width��ksize.height���������
	*/
	GaussianBlur(srcImage, gaussianblur, Size(7, 7), 0, 0);
	/* ��ֵ�˲�
	* ֻ���������������һ������ksize�����Ǵ���1�����������������Ӧ���У����ǽ�����һ��7x7����
	* �ú������ڶ�ͨ��ͼ�����ͨ������
	*/
	medianBlur(srcImage, medianblur, 7);
	/* ˫���˲�����Ϊһ�ַ������˲��������Ա��ֱ�Ե����ƽ��������ü�Ȩƽ���ķ�����ϵ�����ڸ�˹�ֲ�������
		������Ҫ���ǣ�˫���˲���Ȩ�ز������������ص�ŷʽ���룬�����������ط�Χ���ڵķ������
		�����磺���Ƴ̶ȣ���ɫǿ�ȣ���Ⱦ���ȣ�����һ�ֱ�Ե�����˲�����
	* ˫���˲��ĺ˺����ǿռ���������ط�Χ��˵��ۺϽ������ͼ���ƽ̹��������ֵ�仯��С����Ӧ�����ط�
		Χ��Ȩ�ؽӽ���1����ʱ�ռ���Ȩ������Ҫ���ã��൱�ڽ��и�˹ģ������ͼ��ı�Ե��������ֵ�仯�ܴ�
		���ط�Χ��Ȩ�ر�󣬴Ӷ������˱�Ե����Ϣ��
	void bilateralFilter( InputArray src, OutputArray dst, int d,
								   double sigmaColor, double sigmaSpace,
								   int borderType = BORDER_DEFAULT );
	* src������ͼ��
	* dst: ���ͼ��
	* d���˲����ڵ�ֱ��������ע����ʹ�õ���Diameter����ô�ܿ��ܺ�����ѡȡ�Ĵ�����Բ�δ��ڣ�
	* sigmaColor������ֵ�򷽲�
	* sigmaSpace���ռ��򷽲�
	*/
	bilateralFilter(srcImage, bilateral, 20, 50, 50);
	imshow("ori", srcImage);
	imshow("box", boxfliter);
	imshow("blur", mblur);
	imshow("gaussian", gaussianblur);
	imshow("median", medianblur);
	imshow("bilateral", bilateral);
	waitKey(0);
}
void Sobel_Opencv()
{
	Mat srcImage = imread("test6.JPG");
	imshow("ori", srcImage);
	Mat xdstImage, ydstImage, dstImage;
	/* Sobel (
	* InputArray src,//����ͼ
	* OutputArray dst,//���ͼ
	* int ddepth,//���ͼ������
	* int dx,    // x �����ϵĲ�ֽ���
	* int dy,     // y�����ϵĲ�ֽ���
	* int ksize=3, // ��Ĭ��ֵ3����ʾSobel�˵Ĵ�С;����ȡ1��3��5��7
	* double scale=1,
	* double delta=0,
	* int borderType=BORDER_DEFAULT );
	*/
	Sobel(srcImage, xdstImage, -1, 1, 0);
	imshow("xdst", xdstImage);
	Sobel(srcImage, ydstImage, -1, 0, 1);
	imshow("ydst", ydstImage);
	addWeighted(xdstImage, 0.5, ydstImage, 0.5, 1, dstImage);
	imshow("dst", dstImage);
	imshow("res", srcImage - dstImage);
	imshow("res1", srcImage + dstImage);
	waitKey(0);
}
void Sobel_Mine()
{
	Mat m_img = imread("test6.JPG");
	Mat src(m_img.rows, m_img.cols, CV_8UC1, Scalar(0));
	cvtColor(m_img, src, CV_RGB2GRAY);

	Mat dstImage(src.rows, src.cols, CV_8UC1, Scalar(0));
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			dstImage.data[i * dstImage.step + j] = sqrt((src.data[(i - 1) * src.step + j + 1]
				+ 2 * src.data[i * src.step + j + 1]
				+ src.data[(i + 1) * src.step + j + 1]
				- src.data[(i - 1) * src.step + j - 1] - 2 * src.data[i * src.step + j - 1]
				- src.data[(i + 1) * src.step + j - 1]) * (src.data[(i - 1) * src.step + j + 1]
					+ 2 * src.data[i * src.step + j + 1] + src.data[(i + 1) * src.step + j + 1]
					- src.data[(i - 1) * src.step + j - 1] - 2 * src.data[i * src.step + j - 1]
					- src.data[(i + 1) * src.step + j - 1]) + (src.data[(i - 1) * src.step + j - 1] + 2 * src.data[(i - 1) * src.step + j]
						+ src.data[(i - 1) * src.step + j + 1] - src.data[(i + 1) * src.step + j - 1]
						- 2 * src.data[(i + 1) * src.step + j]
						- src.data[(i + 1) * src.step + j + 1]) * (src.data[(i - 1) * src.step + j - 1] + 2 * src.data[(i - 1) * src.step + j]
							+ src.data[(i - 1) * src.step + j + 1] - src.data[(i + 1) * src.step + j - 1]
							- 2 * src.data[(i + 1) * src.step + j]
							- src.data[(i + 1) * src.step + j + 1]));

		}

	}
	Mat grad_y(src.rows, src.cols, CV_8UC1, Scalar(0));
	{
		for (int i = 1; i < src.rows - 1; i++)
		{
			for (int j = 1; j < src.cols - 1; j++)
			{
				grad_y.data[i * grad_y.step + j] = abs((src.data[(i - 1) * src.step + j + 1]
					+ 2 * src.data[i * src.step + j + 1]
					+ src.data[(i + 1) * src.step + j + 1]
					- src.data[(i - 1) * src.step + j - 1] - 2 * src.data[i * src.step + j - 1]
					- src.data[(i + 1) * src.step + j - 1]));
			}
		}
	}
	Mat grad_x(src.rows, src.cols, CV_8UC1, Scalar(0));
	{
		for (int i = 1; i < src.rows - 1; i++)
		{
			for (int j = 1; j < src.cols - 1; j++)
			{
				grad_x.data[i * grad_x.step + j] = sqrt((src.data[(i - 1) * src.step + j - 1] + 2 * src.data[(i - 1) * src.step + j]
					+ src.data[(i - 1) * src.step + j + 1] - src.data[(i + 1) * src.step + j - 1]
					- 2 * src.data[(i + 1) * src.step + j]
					- src.data[(i + 1) * src.step + j + 1]) * (src.data[(i - 1) * src.step + j - 1] + 2 * src.data[(i - 1) * src.step + j]
						+ src.data[(i - 1) * src.step + j + 1] - src.data[(i + 1) * src.step + j - 1]
						- 2 * src.data[(i + 1) * src.step + j]
						- src.data[(i + 1) * src.step + j + 1]));
			}
		}
	}
	imshow("ԭͼ", src);
	imshow("gradient", dstImage);
	imshow("Vertical gradient", grad_y);
	imshow("Horizontal gradient", grad_x);
	imshow("res", src + dstImage);
	imshow("resx", src + grad_x);
	imshow("resy", src + grad_y);
	waitKey(0);
}
void Laplacian_Opencv()
{
	Mat srcImage = imread("test6.JPG");
	Mat dstImage;
	////�˲���������
	//GaussianBlur(srcImage, srcImage, Size(3, 3), 0, 0, BORDER_DEFAULT);
	// �ú���Ĭ��ksize=1����ʱ���˲���ģ��Ϊ{0}{1}{0}{1}{-4}{1}{0}{1}{0}
	Laplacian(srcImage, dstImage, srcImage.depth());
	imshow("ori", srcImage);
	imshow("La", dstImage);
	dstImage = srcImage - dstImage;
	imshow("dst", dstImage);
	waitKey(0);
}

void Laplacian_Mine()
{
	Mat srcImage = imread("test6.JPG", 0);
	GaussianBlur(srcImage, srcImage, Size(3, 3), 0, 0, BORDER_DEFAULT);
	imshow("ori", srcImage);
	int mask[9] = { -1,-1,-1,-1,8,-1,-1,-1,-1 };
	//int mask[9] = { 0,1,0,1,-4,1,0,1,0 };

	int nr = srcImage.rows;
	int nc = srcImage.cols;
	int n = nr * nc;
	int arr[9] = { 0 };

	int* table_lap = new int[n];
	int* table_orig = new int[n];
	int l;
	for (int i = 0; i < n; i++)
	{
		table_lap[i] = 0;
		table_orig[i] = 0;
	}
	for (int i = 1; i < nr - 1; i++)
	{
		const uchar* previous = srcImage.ptr<uchar>(i - 1);
		const uchar* current = srcImage.ptr<uchar>(i);
		const uchar* next = srcImage.ptr<uchar>(i + 1);
		for (int j = 1; j < nc - 1; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				arr[k] = previous[j + k - 1];
				arr[k + 3] = current[j + k - 1];
				arr[k + 6] = next[j + k - 1];
			}
			l = nc * i + j;        //calculate the location in the table of current pixel
			for (int mf = 0; mf < 9; mf++)
			{
				table_lap[l] = table_lap[l] + mask[mf] * arr[mf];
			}
			table_orig[l] = arr[4];
		}
	}
	//�궨���Ҷ���������0,255��
	uchar* La_scaled = new uchar[n];
	int min = table_lap[0];
	int max = table_lap[0];
	for (int i = 0; i < n; i++)
	{
		if (min > table_lap[i])
			min = table_lap[i];
		if (max < table_lap[i])
			max = table_lap[i];
	}
	for (int i = 0; i < n; i++)
	{
		La_scaled[i] = (uchar)(255 * (table_lap[i] - min) / (max - min));
	}
	//����궨���������˹���
	Mat LaRes;
	LaRes.create(srcImage.size(), srcImage.type());
	for (int i = 0; i < nr; i++)
	{
		uchar* p = LaRes.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{
			l = nc * i + j;
			//p[j] = La_scaled[l];
			p[j] = table_orig[l] + table_lap[l];
			if (p[j] > 255)p[j] = 255;
			if (p[j] < 0)p[j] = 0;
		}
	}
	imshow("LaRes", LaRes);
	waitKey();
}
int main()
{


	//Sobel_Mine();
	//Sobel_Opencv();
	//Laplacian_Mine();
	//Laplacian_Opencv();
	//allFilters();
	//hisMatch();
	equalizeHistOpencv();
	//bitLevel();
	//contrastStretch();
	//GammaCorr();
	//grayInv();
}