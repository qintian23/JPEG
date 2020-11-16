﻿#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void calcPSF(Mat& outputImg, Size filterSize, int len, double theta);
void fftshift(const Mat& inputImg, Mat& outputImg);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);
void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma = 5.0, double beta = 0.2);

int LEN = 50;
int THETA = 360;
int snr = 8000;
Mat imgIn;
Rect roi;
static void onChange(int pos, void* userInput);

int main(int argc, char* argv[])
{
	string strInFileName = "1.JPG";

	imgIn = imread(strInFileName, IMREAD_GRAYSCALE);
	if (imgIn.empty()) //check whether the image is loaded or not
	{
		cout << "ERROR : Image cannot be loaded..!!" << endl;
		return -1;
	}
	imshow("src", imgIn);

	// it needs to process even image only
	roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);
	imgIn = imgIn(roi);
	cv::namedWindow("inverse");

	createTrackbar("LEN", "inverse", &LEN, 200, onChange, &imgIn);
	onChange(0, 0);
	createTrackbar("THETA", "inverse", &THETA, 360, onChange, &imgIn);
	onChange(0, 0);
	createTrackbar("snr", "inverse", &snr, 10000, onChange, &imgIn);
	onChange(0, 0);
	imshow("inverse", imgIn);
	cv::waitKey(0);

	return 0;
}

void calcPSF(Mat& outputImg, Size filterSize, int len, double theta)
{
	Mat h(filterSize, CV_32F, Scalar(0));
	Point point(filterSize.width / 2, filterSize.height / 2);
	ellipse(h, point, Size(0, cvRound(float(len) / 2.0)), 90.0 - theta,
		0, 360, Scalar(255), FILLED);
	Scalar summa = sum(h);
	outputImg = h / summa[0];
	Mat tmp;
	normalize(outputImg, tmp, 1, 0, NORM_MINMAX);
	imshow("psf", tmp);
}
void fftshift(const Mat& inputImg, Mat& outputImg)
{
	outputImg = inputImg.clone();
	int cx = outputImg.cols / 2;
	int cy = outputImg.rows / 2;
	Mat q0(outputImg, Rect(0, 0, cx, cy));
	Mat q1(outputImg, Rect(cx, 0, cx, cy));
	Mat q2(outputImg, Rect(0, cy, cx, cy));
	Mat q3(outputImg, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
	Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI, DFT_SCALE);
	Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
	Mat complexH;
	merge(planesH, 2, complexH);
	Mat complexIH;
	mulSpectrums(complexI, complexH, complexIH, 0);
	idft(complexIH, complexIH);
	split(complexIH, planes);
	outputImg = planes[0];
}
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
{
	Mat h_PSF_shifted;
	fftshift(input_h_PSF, h_PSF_shifted);
	Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);
	Mat denom;
	pow(abs(planes[0]), 2, denom);
	denom += nsr;
	divide(planes[0], denom, output_G);
}
void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
{
	int Nx = inputImg.cols;
	int Ny = inputImg.rows;
	Mat w1(1, Nx, CV_32F, Scalar(0));
	Mat w2(Ny, 1, CV_32F, Scalar(0));
	float* p1 = w1.ptr<float>(0);
	float* p2 = w2.ptr<float>(0);
	float dx = float(2.0 * CV_PI / Nx);
	float x = float(-CV_PI);
	for (int i = 0; i < Nx; i++)
	{
		p1[i] = float(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
		x += dx;
	}
	float dy = float(2.0 * CV_PI / Ny);
	float y = float(-CV_PI);
	for (int i = 0; i < Ny; i++)
	{
		p2[i] = float(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
		y += dy;
	}
	Mat w = w2 * w1;
	multiply(inputImg, w, outputImg);
}

// Trackbar call back function
static void onChange(int, void* userInput)
{
	Mat imgOut;
	//Hw calculation (start)
	Mat Hw, h;
	calcPSF(h, roi.size(), LEN, (double)THETA);
	calcWnrFilter(h, Hw, 1.0 / double(snr));
	//Hw calculation (stop)
	imgIn.convertTo(imgIn, CV_32F);
	edgetaper(imgIn, imgIn);
	// filtering (start)
	filter2DFreq(imgIn(roi), imgOut, Hw);
	// filtering (stop)
	imgOut.convertTo(imgOut, CV_8U);
	normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
	//    imwrite("result.jpg", imgOut);
	imshow("inverse", imgOut);
}