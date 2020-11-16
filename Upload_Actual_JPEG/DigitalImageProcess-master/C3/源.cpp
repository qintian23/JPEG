#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
using namespace std;
using namespace cv;
/*线性变化之灰度级反转*/
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
	//// 归一化处理
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
	//建立待查表文件LUT
	unsigned char lut[256];
	for (int i = 0; i < 256; i++) {
		//防止彩色溢出，大于255的像素令其为255，小于0的像素令其为0
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}
	Mat dstImg = srcImg.clone();
	const int channels = dstImg.channels();
	switch (channels)
	{
	case 1://灰度图
	{
		MatIterator_<uchar> it, end;
		for (it = dstImg.begin<uchar>(), end = dstImg.end<uchar>(); it != end; it++)
			//*it = pow((float)(((*it)) / 255.0), fGamma) * 255.0;
			*it = lut[(*it)];
		break;
	}
	case 3://彩色图 
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
	//"=";"clone()";"copyTo"三种拷贝方式，前者是浅拷贝，后两者是深拷贝。
	Mat resultImage = srcImage.clone();
	int nRows = resultImage.rows;
	int nCols = resultImage.cols;
	//判断图像存储的连续性，若连续可以得到像素个数
	if (resultImage.isContinuous())
	{
		nCols = nCols * nRows;
		nRows = 1;
	}
	//图像指针操作
	uchar* pDataMat;
	int pixMax = 0, pixMin = 255;
	//计算图像的最大最小值
	for (int j = 0; j < nRows; j++)
	{
		//ptr<>()得到的是一行指针，智能指针+模板类
		pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++)
		{
			if (pDataMat[i] > pixMax)
				pixMax = pDataMat[i];
			if (pDataMat[i] < pixMin)
				pixMin = pDataMat[i];
		}
	}
	//对比度拉伸映射，从原始范围拉伸到num1~num2
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
		//CV_8UC1：其中8代表比特数，0~255；U代表无符号整型，F代表单精度浮点型；
		//C代表通道数；1代表灰度图像即单通道，2代表RGB彩色图像即三通道，3代表
		//带Alpha通道（透明度）的RGB图像，即四通道
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
// 1.调用库函数
// 彩色图像需要分通道均衡化，灰度图像直接调用即可
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
// 2. 自己实现
Mat equalizeHistMine(Mat srcImage)
{
	int gray[256] = { 0 };
	double gray_prob[256] = { 0 };
	double gray_disSum[256] = { 0 };
	int gray_equal[256] = { 0 };
	Mat dstImage = srcImage.clone();
	int gray_sum = srcImage.cols * srcImage.rows;
	//统计每个灰度下的像素个数
	for (int i = 0; i < srcImage.rows; i++)
	{
		uchar* p = srcImage.ptr<uchar>(i);
		for (int j = 0; j < srcImage.cols; j++)
		{
			int value = p[j];
			gray[value]++;
		}
	}
	//统计灰度频率
	for (int i = 0; i < 256; i++)
	{
		gray_prob[i] = ((double)gray[i] / gray_sum);
	}
	//计算累计密度
	gray_disSum[0] = gray_prob[0];
	for (int i = 1; i < 256; i++)
	{
		gray_disSum[i] = gray_disSum[i - 1] + gray_prob[i];
	}
	//重新计算均衡化的灰度值，
	for (int i = 0; i < 256; i++)
	{
		gray_equal[i] = (uchar)(255 * gray_disSum[i] + 0.5);
	}
	//更新
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
	int hist[256] = { 0 };//各级像素数目
	int S[256] = { 0 };
	map<int, int> S2Z;//S（均衡化灰度）到Z（输出图像灰度）的映射
	map<int, int> R2Z;//R（原始图像灰度）到Z（输出图像灰度）的映射
	//直方图统计
	for (int i = 0; i < h; i++)
	{
		uchar* p = srcImage.ptr<uchar>(i);
		for (int j = 0; j < w; j++)
		{
			int value = p[j];
			hist[value]++;
		}
	}
	//归一化累加直方图
	float sumHist[256] = { 0 };
	for (int i = 0; i < 256; i++)
	{
		int sum = 0;
		for (int j = 0; j <= i; j++)
			sum += hist[j];
		sumHist[i] = sum * 1.0 / (h * w);
	}
	//根据sumHist建立均衡化后的灰度级数组S
	for (int i = 0; i < 256; i++)
		S[i] = 255 * sumHist[i] + 0.5;
	//根据zSumHist建立均衡化后灰度级数组G
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

	//令G(Z)=S 建立S->Z的映射表
	for (int i = 0; i < 256; i++) {
		for (int j = 1; j < 256; j++) {
			//G[i]递增，只需满足下面的判断条件，即为最接近
			if (abs(S[i] - G[j - 1]) < abs(S[i] - G[j]))
			{
				S2Z[S[i]] = j - 1;
				break;
			}
		}
	}
	S2Z[S[255]] = 255;
	//建立R->Z的映射
	for (int i = 0; i < 256; i++)
		R2Z[i] = S2Z[S[i]];
	//重建图像
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
	/* 方框滤波：该滤波器的响应是模板邻域像素值得平均值，也就是上面提到的模板系数全为1的盒装滤波器
	C++: void boxFilter(InputArray src, OutputArray dst, int ddepth, Size ksize, Point anchor=Point(-1,-1), bool normalize=true, int borderType=BORDER_DEFAULT )
	* src：输入图像
	* dst：输出图像（与输入图像等大）
	* ddepth：输出图像的深度，若为-1，则和输入图像相同
	* ksize：滤波器模板尺寸
	* anchor：锚点即被平滑的那个点，默认（-1，-1）代表锚点在滤波核的中心
	* normalize：默认为true，表示核是否被其区域归一化
	* borderType：用于推断突袭那个外部像素的某种边界模式，一般不同管
	*/
	boxFilter(srcImage, boxfliter, -1, Size(7, 7));
	/*均值滤波：相当于调用normalize=true的方框滤波一样，只不过参数有些许的不同，模板系数均为1
	* src：输入图像，可以有任意的通道数，但是深度必须为CV_8U, CV_16U, CV_16S, CV_32F 或 CV_64F.
	*/
	blur(srcImage, mblur, Size(7, 7));
	/* 高斯滤波：相当于模板系数由高斯函数计算生成的加权平均滤波，高斯滤波器是一种线性平滑滤波器，
		其滤波器的模板是对二维高斯函数离散得到。由于高斯模板的中心值最大，四周逐渐减小，其滤波后
		的结果相对于均值滤波器来说更好。高斯滤波器最重要的参数就是高斯分布的标准差，标准差和高斯
		滤波器的平滑能力有很大的能力，越大，高斯滤波器的频带就较宽，对图像的平滑程度就越好。通过
		调节参数，可以平衡对图像的噪声的抑制和对图像的模糊。
	* sigmaX，表示高斯核函数在X方向的的标准偏差
	* sigmaY，表示高斯核函数在Y方向的的标准偏差。若sigmaY为零，就将它设为sigmaX，
		如果sigmaX和sigmaY都是0，那么就由ksize.width和ksize.height计算出来。
	*/
	GaussianBlur(srcImage, gaussianblur, Size(7, 7), 0, 0);
	/* 中值滤波
	* 只有三个参数，最后一个参数ksize必须是大于1的奇数，比如下面的应用中，我们将计算一个7x7邻域
	* 该函数对于多通道图像会逐通道处理。
	*/
	medianBlur(srcImage, medianblur, 7);
	/* 双边滤波：作为一种非线性滤波器，可以保持边缘降噪平滑，其采用加权平均的方法，系数基于高斯分布产生，
		但是重要的是，双边滤波的权重不仅考虑了像素的欧式距离，还考虑了像素范围域内的辐射差异
		（例如：相似程度，颜色强度，深度距离等），是一种边缘保护滤波方法
	* 双边滤波的核函数是空间域核与像素范围域核的综合结果：在图像的平坦区域，像素值变化很小，对应的像素范
		围域权重接近于1，此时空间域权重起主要作用，相当于进行高斯模糊；在图像的边缘区域，像素值变化很大，
		像素范围域权重变大，从而保持了边缘的信息。
	void bilateralFilter( InputArray src, OutputArray dst, int d,
								   double sigmaColor, double sigmaSpace,
								   int borderType = BORDER_DEFAULT );
	* src：输入图像
	* dst: 输出图像
	* d：滤波窗口的直径（函数注释中使用的是Diameter，那么很可能函数中选取的窗口是圆形窗口）
	* sigmaColor：像素值域方差
	* sigmaSpace：空间域方差
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
	* InputArray src,//输入图
	* OutputArray dst,//输出图
	* int ddepth,//输出图像的深度
	* int dx,    // x 方向上的差分阶数
	* int dy,     // y方向上的差分阶数
	* int ksize=3, // 有默认值3，表示Sobel核的大小;必须取1，3，5或7
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
	imshow("原图", src);
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
	////滤波消除噪声
	//GaussianBlur(srcImage, srcImage, Size(3, 3), 0, 0, BORDER_DEFAULT);
	// 该函数默认ksize=1，此时的滤波器模板为{0}{1}{0}{1}{-4}{1}{0}{1}{0}
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
	//标定，灰度拉伸至（0,255）
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
	//储存标定后的拉普拉斯结果
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