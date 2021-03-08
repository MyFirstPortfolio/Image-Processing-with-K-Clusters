////// ippr.cpp : this file contains the 'main' function. program execution begins and ends there.
////// 
//////-----------------------------------------------------------------  TILL LINE  640
////#include <opencv2\opencv.hpp>
////#include <sstream>
////#include <iostream>
////#include <opencv2/saliency.hpp>
////#include "opencv2/core.hpp"
////#include <opencv2/imgcodecs.hpp>
////#include <opencv2/imgproc.hpp>
////#include <opencv2/videoio.hpp>
////#include <opencv2/highgui.hpp>
////
////using namespace cv;
////using namespace std;
////using namespace saliency;
////
/////*void fftshift(const mat& input_img, mat& output_img)
////{
////	output_img = input_img.clone();
////	int cx = output_img.cols / 2;
////	int cy = output_img.rows / 2;
////	mat q1(output_img, rect(0, 0, cx, cy));
////	mat q2(output_img, rect(cx, 0, cx, cy));
////	mat q3(output_img, rect(0, cy, cx, cy));
////	mat q4(output_img, rect(cx, cy, cx, cy));
////
////	mat temp;
////	q1.copyto(temp);
////	q4.copyto(q1);
////	temp.copyto(q4);
////	q2.copyto(temp);
////	q3.copyto(q2);
////	temp.copyto(q3);
////}
////void calculatedft(mat& scr, mat& dst)
////{
////	// define mat consists of two mat, one for real values and the other for complex values
////	mat planes[] = { scr, mat::zeros(scr.size(), cv_32f) };
////	mat compleximg;
////	merge(planes, 2, compleximg);
////
////	dft(compleximg, compleximg);
////	dst = compleximg;
////}
////mat construct_h(mat& scr, string type, float d0)
////{
////	mat h(scr.size(), cv_32f, scalar(1));
////	float d = 0;
////	if (type == "i")
////	{
////		for (int u = 0; u < h.rows; u++)
////		{
////			for (int v = 0; v < h.cols; v++)
////			{
////				d = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
////				if (d > d0)
////				{
////					h.at<float>(u, v) = 0;
////				}
////			}
////		}
////		return h;
////	}
////	else if (type == "g")
////	{
////		for (int u = 0; u < h.rows; u++)
////		{
////			for (int v = 0; v < h.cols; v++)
////			{
////				d = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
////				h.at<float>(u, v) = exp(-d * d / (2 * d0 * d0));
////			}
////		}
////		return h;
////	}
////}
////void filtering(mat& scr, mat& dst, mat& h)
////{
////	fftshift(h, h);
////	mat planesh[] = { mat_<float>(h.clone()), mat_<float>(h.clone()) };
////
////	mat planes_dft[] = { scr, mat::zeros(scr.size(), cv_32f) };
////	split(scr, planes_dft);
////
////	mat planes_out[] = { mat::zeros(scr.size(), cv_32f), mat::zeros(scr.size(), cv_32f) };
////	planes_out[0] = planesh[0].mul(planes_dft[0]);
////	planes_out[1] = planesh[1].mul(planes_dft[1]);
////
////	merge(planes_out, 2, dst);
////
////}
////int main()
////{
////	mat imgin = imread("d://pictures//images(3).png", 0);
////	imgin.convertto(imgin, cv_32f);
////
////	// dft	
////	mat dft_image;
////	calculatedft(imgin, dft_image);
////
////	// construct h
////	mat h;
////	h = construct_h(imgin, "i", 85);
////
////	// filtering
////	mat complexih;
////	filtering(dft_image, complexih, h);
////
////	// idft
////	mat imgout;
////	dft(complexih, imgout, dft_inverse | dft_real_output);
////
////	normalize(imgout, imgout, 0, 1, norm_minmax);
////
////	imshow("img", imgin);
////	imshow("dft", imgout);
////	waitkey(0);
////	return 0;
////}*/
////
//////hf anf lf
/////*
////mat converttogrey(mat rgbimg)
////{
////	int iheight = rgbimg.rows;
////	int iwidth = rgbimg.cols;
////	mat greyimg = mat::zeros(rgbimg.size(), cv_8uc1);
////	for (int i = 0; i < iheight; i++) {
////		for (int j = 0; j < iwidth * 3; j += 3) {
////			greyimg.at<uchar>(i, j / 3) = (rgbimg.at<uchar>(i, j) + rgbimg.at<uchar>(i, j + 1) + rgbimg.at<uchar>(i, j + 2)) / 3;
////		}
////	}
////	return greyimg;
////}
////
////mat converttobw(mat greyimg)
////{
////	int iheight = greyimg.rows;
////	int iwidth = greyimg.cols;
////	mat bwimg = mat::zeros(greyimg.size(), cv_8uc1);
////	for (int i = 0; i < iheight; i++) {
////		for (int j = 0; j < iwidth; j++) {
////			if (greyimg.at<uchar>(i, j) > 60) {
////				bwimg.at<uchar>(i, j) = 255;
////			}
////			else {
////				bwimg.at<uchar>(i, j) = 0;
////			}
////		}
////	}
////	return bwimg;
////}
////
////mat edge(mat greyimg)
////{
////	int iheight = greyimg.rows;
////	int iwidth = greyimg.cols;
////	mat convimg = mat::zeros(greyimg.size(), cv_8uc1);
////	for (int i = 1; i < iheight - 1; i++) {
////		for (int j = 1; j < iwidth - 1; j++) {
////			int avg1 = (greyimg.at<uchar>(i - 1, j - 1) + greyimg.at<uchar>(i, j - 1) + greyimg.at<uchar>(i + 1, j - 1)) / 3;
////			int avg2 = (greyimg.at<uchar>(i - 1, j + 1) + greyimg.at<uchar>(i, j + 1) + greyimg.at<uchar>(i + 1, j + 1)) / 3;
////			int diff = abs(avg1 - avg2);
////			if (diff > 40) {
////				convimg.at<uchar>(i, j) = 255;
////			}
////		}
////	}
////	return convimg;
////
////}
////
////mat blur(mat greyimg, int winsize) {
////	int iheight = greyimg.rows;
////	int iwidth = greyimg.cols;
////	mat blurimg = mat::zeros(greyimg.size(), cv_8uc1);
////	int neighbor = (winsize - 1) / 2;
////	for (int i = neighbor; i < iheight - neighbor; i++) {
////		for (int j = neighbor; j < iwidth - neighbor; j++) {
////			int sum = 0;
////			for (int ii = -neighbor; ii < neighbor + 1; ii++) {
////				for (int jj = -neighbor; jj < neighbor + 1; jj++) {
////					sum += greyimg.at<uchar>(i + ii, j + jj);
////				}
////			}
////			blurimg.at<uchar>(i, j) = sum / (winsize * winsize);
////		}
////	}
////	return blurimg;
////}
////
////mat min(mat greyimg, int winsize) {
////	int iheight = greyimg.rows;
////	int iwidth = greyimg.cols;
////	mat blurimg = mat::zeros(greyimg.size(), cv_8uc1);
////	int neighbor = (winsize - 1) / 2;
////	for (int i = neighbor; i < iheight - neighbor; i++) {
////		for (int j = neighbor; j < iwidth - neighbor; j++) {
////			int min = 256;
////			for (int ii = -neighbor; ii < neighbor + 1; ii++) {
////				for (int jj = -neighbor; jj < neighbor + 1; jj++) {
////					if (min > greyimg.at<uchar>(i + ii, j + jj)) {
////						min = greyimg.at<uchar>(i + ii, j + jj);
////						if (min == 0) break;
////					}
////				}
////				if (min == 0) break;
////			}
////			blurimg.at<uchar>(i, j) = min;
////		}
////	}
////	return blurimg;
////}
////
////mat max(mat greyimg, int winsize) {
////	int iheight = greyimg.rows;
////	int iwidth = greyimg.cols;
////	mat blurimg = mat::zeros(greyimg.size(), cv_8uc1);
////	int neighbor = (winsize - 1) / 2;
////	for (int i = neighbor; i < iheight - neighbor; i++) {
////		for (int j = neighbor; j < iwidth - neighbor; j++) {
////			int max = -1;
////			for (int ii = -neighbor; ii < neighbor + 1; ii++) {
////				for (int jj = -neighbor; jj < neighbor + 1; jj++) {
////					if (max < greyimg.at<uchar>(i + ii, j + jj)) {
////						max = greyimg.at<uchar>(i + ii, j + jj);
////						if (max == 255) break;
////					}
////				}
////				if (max == 255) break;
////			}
////			blurimg.at<uchar>(i, j) = max;
////		}
////	}
////	return blurimg;
////}
////
////mat median(mat greyimg, int winsize) {
////	int iheight = greyimg.rows;
////	int iwidth = greyimg.cols;
////	mat blurimg = mat::zeros(greyimg.size(), cv_8uc1);
////	int neighbor = (winsize - 1) / 2;
////	for (int i = neighbor; i < iheight - neighbor; i++) {
////		for (int j = neighbor; j < iwidth - neighbor; j++) {
////			int* arr = new int[winsize * winsize];
////			int index = 0;
////			for (int ii = -neighbor; ii < neighbor + 1; ii++) {
////				for (int jj = -neighbor; jj < neighbor + 1; jj++) {
////					arr[index++] = greyimg.at<uchar>(i + ii, j + jj);
////				}
////			}
////			for (int a = 0; a <= winsize * winsize - 2; a++) {
////				for (int b = a + 1; b <= winsize * winsize - 1; b++) {
////					if (arr[a] > arr[b]) {
////						int temp = arr[a];
////						arr[a] = arr[b];
////						arr[b] = arr[a];
////					}
////				}
////			}
////			blurimg.at<uchar>(i, j) = arr[(winsize * winsize) / 2];
////		}
////	}
////	return blurimg;
////}
////
////void movedft(mat& fimage)
////{
////	mat tmp, q0, q1, q2, q3;
////
////
////	fimage = fimage(rect(0, 0, fimage.cols & -2, fimage.rows & -2));
////
////	int cx = fimage.cols / 2;
////	int cy = fimage.rows / 2;
////
////
////	q0 = fimage(rect(0, 0, cx, cy));
////	q1 = fimage(rect(cx, 0, cx, cy));
////	q2 = fimage(rect(0, cy, cx, cy));
////	q3 = fimage(rect(cx, cy, cx, cy));
////
////	q0.copyto(tmp);
////	q3.copyto(q0);
////	tmp.copyto(q3);
////
////	q1.copyto(tmp);
////	q2.copyto(q1);
////	tmp.copyto(q2);
////}
////
////void convtocomplexandmagnitude(mat greyimg, mat& complexnumberoutput, mat& dftmagnitudeoutput)
////{
////	mat originalfloat;
////	greyimg.convertto(originalfloat, cv_32fc1, 1.0 / 255);
////	mat originalcomplex[2] = { originalfloat, mat::zeros(originalfloat.size(), cv_32f) };
////
////	mat complexnumbers;
////	merge(originalcomplex, 2, complexnumbers);
////
////	dft(complexnumbers, complexnumbers);
////
////	mat splitarray[2] = { mat::zeros(originalfloat.size(), cv_32f),
////		mat::zeros(originalfloat.size(), cv_32f) };
////
////	split(complexnumbers, splitarray);
////
////	mat dftmagnitude;
////
////	magnitude(splitarray[0], splitarray[1], dftmagnitude);
////
////	dftmagnitude += scalar::all(1);
////
////
////	log(dftmagnitude, dftmagnitude);
////
////	normalize(dftmagnitude, dftmagnitude, 0, 1, norm_minmax);
////
////	movedft(dftmagnitude);
////
////	complexnumberoutput = complexnumbers;
////	dftmagnitudeoutput = dftmagnitude;
////}
////
////void createlowpassgaussianfilter(size size, mat& gaussianfilter, mat& gaussianimage, int ux, int uy, float sigmax, float sigmay, float amplitude = 1.0f)
////{
////	mat temp = mat(size, cv_32f);
////
////	for (int r = 0; r < size.height; r++)
////	{
////		for (int c = 0; c < size.width; c++)
////		{
////			float x = ((c - ux) * ((float)c - ux)) / (2.0f * sigmax * sigmax);
////			float y = ((r - uy) * ((float)r - uy)) / (2.0f * sigmay * sigmay);
////			float value = amplitude * exp(-(x + y));
////			temp.at<float>(r, c) = value;
////		}
////	}
////	gaussianimage = temp;
////
////	mat tomerge[] = { temp, temp };
////	merge(tomerge, 2, gaussianfilter);
////}
////
////void createhighpassgaussianfilter(size size, mat& gaussianfilter, mat& gaussianimage, int ux, int uy, float sigmax, float sigmay, float amplitude = 1.0f)
////{
////	mat temp = mat(size, cv_32f);
////
////	for (int r = 0; r < size.height; r++)
////	{
////		for (int c = 0; c < size.width; c++)
////		{
////			float x = ((c - ux) * ((float)c - ux)) / (2.0f * sigmax * sigmax);
////			float y = ((r - uy) * ((float)r - uy)) / (2.0f * sigmay * sigmay);
////			float value = amplitude * exp(-(x + y));
////			temp.at<float>(r, c) = 1 - value;
////		}
////	}
////	gaussianimage = temp;
////
////	mat tomerge[] = { temp, temp };
////	merge(tomerge, 2, gaussianfilter);
////}
////
////
////void invert(mat filteredcomplex, mat greyimg, mat& output)
////{
////	idft(filteredcomplex, filteredcomplex);
////
////	mat originalfloat;
////	greyimg.convertto(originalfloat, cv_32fc1, 1.0 / 255);
////	mat originalcomplex[2] = { originalfloat, mat::zeros(originalfloat.size(), cv_32f) };
////
////	mat imgoutput;
////	split(filteredcomplex, originalcomplex);
////	normalize(originalcomplex[0], imgoutput, 0, 1, norm_minmax);
////	output = imgoutput;
////}
////
////void treshold(mat source, mat& output)
////{
////	int iheight = source.rows;
////	int iwidth = source.cols;
////
////	mat outputimage = mat::zeros(source.size(), cv_32f);
////
////	for (int i = 0; i < iheight - 1; i++)
////	{
////		for (int j = 1; j < iwidth - 1; j++)
////		{
////			if (source.at<float>(i, j) > 240)
////			{
////				outputimage.at<float>(i, j) = 255;
////			}
////
////		}
////	}
////	output = outputimage;
////}
////
////////image
////int main() {
////
////	mat rgbimg = imread("d:\\pictures\\testing.jpg");
////	resize(rgbimg, rgbimg, size(rgbimg.cols / 2, rgbimg.rows / 2));
////	imshow("original", rgbimg);
////	waitkey();
////	mat greyimg = converttogrey(rgbimg);
////	imshow("grey", greyimg);
////	waitkey();
////	//mat bwimg = converttobw(greyimg);
////	//imshow("bnw", bwimg);
////	//waitkey();
////	//mat convimg = edge(greyimg);
////	//imshow("edge", convimg);
////	//waitkey();
////	//mat blurimg = blur(greyimg, 5);
////	//imshow("blur", blurimg);
////	//waitkey();
////	//mat minimg = min(greyimg, 3);
////	//imshow("min", minimg);
////	//waitkey();
////	//mat maximg = max(greyimg, 3);
////	//imshow("max", maximg);
////	//waitkey();
////	//mat medimg = max(greyimg, 3);
////	//imshow("med", medimg);
////	//waitkey();
////
////	mat complexnumber;
////	mat dftmagnitude;
////	convtocomplexandmagnitude(greyimg, complexnumber, dftmagnitude);
////	imshow("dft magnitude", dftmagnitude);
////	waitkey();
////
////	//creating filter and image
////	mat gaussiancomplexfilter, gaussiancomplexfilter1;
////	mat gaussianimage, gaussianimage1;
////	createlowpassgaussianfilter(size(dftmagnitude.cols, dftmagnitude.rows),
////		gaussiancomplexfilter, gaussianimage, dftmagnitude.cols / 2,
////		dftmagnitude.rows / 2, 50, 50);
////	imshow("lpf - masking", gaussianimage);
////	waitkey();
////
////	createhighpassgaussianfilter(size(dftmagnitude.cols, dftmagnitude.rows),
////		gaussiancomplexfilter1, gaussianimage1, dftmagnitude.cols / 2,
////		dftmagnitude.rows / 2, 50, 50);
////	imshow("hpf - masking", gaussianimage1);
////	waitkey();
////
////	//creating dft & gaussian result
////	mat dftxgaussian, dftxgaussian1;
////	mulspectrums(dftmagnitude, gaussianimage, dftxgaussian, dft_rows);
////	imshow("lpf - dft & gaussian", dftxgaussian);
////	waitkey();
////	mulspectrums(dftmagnitude, gaussianimage1, dftxgaussian1, dft_rows);
////	imshow("hpf - dft & gaussian", dftxgaussian1);
////	waitkey();
////
////	//filter applied
////	mat filteredcomplex, filteredcomplex1;
////	movedft(complexnumber);
////	mulspectrums(complexnumber, gaussiancomplexfilter, filteredcomplex, dft_rows);
////	movedft(filteredcomplex);
////	mulspectrums(complexnumber, gaussiancomplexfilter1, filteredcomplex1, dft_rows);
////	movedft(filteredcomplex1);
////
////	mat invertedimage, invertedimage1;
////	invert(filteredcomplex, greyimg, invertedimage);
////	invert(filteredcomplex1, greyimg, invertedimage1);
////
////	imshow(" lpf - gaussian result", invertedimage);
////	waitkey();
////	imshow(" hpf - gaussian result", invertedimage1);
////
////	waitkey();
////
////	return 0;
////}
////*/
////
//////saliency
////
////void spectralresidual(mat source, mat& outputsm, mat& outputbm) {
////	mat image;
////	source.copyto(image);
////	mat saliencymap;
////	ptr<saliency> saliencyalgorithm = staticsaliencyspectralresidual::create();
////	saliencyalgorithm->computesaliency(image, saliencymap);
////	staticsaliencyspectralresidual spec;
////	mat binarymap;
////	spec.computesaliency(saliencymap, binarymap);
////	outputsm = saliencymap;
////	outputbm = binarymap;
////}
////
////void backgroundsubtraction(mat source, mat& output) {
////	//update background model
////	mat greyvid;
////	mat fgmask;
////	cvtcolor(source, greyvid, color_bgr2gray);
////	mat float_mat;
////	greyvid.convertto(float_mat, cv_32f);
////	cvtcolor(source, source, color_bgr2gray);
////	ptr<backgroundsubtractor> pbacksub = createbackgroundsubtractormog2();
////	pbacksub->apply(source, fgmask);
////	output = fgmask;
////
////
////	//get the frame number and write it on the current frame
//////    rectangle(source, cv::point(10,2), cv::point(100,20), cv::scalar(255,255,255), -1);
//////    stringstream ss;
//////    ss << capture.get(cap_prop_pos_frames);
//////    string framenumberstring = ss.str();
//////    puttext(source, framenumberstring.c_str(), cv::point(15,15), font_hershey_duplex, 0.5, cv::scalar(0,0,0));
////
////}
////
////
////int main(int argc, char* argv[])
////{
////	string filename = ("d:\\car.mp4");
////	mat greyvid;
////	mat frame, fgmask;
////	//int keyboard = waitkey(30);
////	videocapture capture(filename);
////	bool playvideo = true;
////
////	ptr<backgroundsubtractor> pbacksub;
////
////
////	if (!capture.isopened())
////		throw "error when reading steam_avi";
////
////	for (; ; )
////	{
////		capture >> frame;
////		if (frame.empty())
////			break;
////		mat backsub;
////		backgroundsubtraction(frame, backsub);
////		imshow("background subtraction", backsub);
////		mat saliencymap, binarymap;
////		imshow("original", frame);
////		spectralresidual(frame, saliencymap, binarymap);
////		imshow("spectral residual saliency map", saliencymap);
////		imshow("spectral redisual binary map", binarymap);
////		while (1)
////		{
////			mat frame;
////
////			bool bsuccess = capture.read(frame); // read a new frame from video
////
////			if (!bsuccess)
////			{
////				cout << "cannot read a frame from video stream" << endl;
////				break;
////			}
////
////			//mat grayscale;
////			mat grayscale(500, 500, cv_8uc3);
////			cvtcolor(frame, grayscale, cv::color_rgb2gray);
////
////			imshow("greyscale video", grayscale);
////
////			const int max_clusters = 5;
////			scalar colortab[] =
////			{
////				scalar(0, 0, 255),
////				scalar(0,255,0),
////				scalar(255,100,100),
////				scalar(255,0,255),
////				scalar(0,255,255)
////			};
////
////
////			rng rng(12345);
////
////			for (;;)
////			{
////				int k, clustercount = rng.uniform(2, max_clusters + 1);
////				int i, samplecount = rng.uniform(1, 1001);
////				mat points(samplecount, 1, cv_32fc2), labels;
////
////				clustercount = min(clustercount, samplecount);
////				std::vector<point2f> centers;
////
////				// generate random sample from multigaussian distribution 
////				for (k = 0; k < clustercount; k++)
////				{
////					point center;
////					center.x = rng.uniform(0, grayscale.cols);
////					center.y = rng.uniform(0, grayscale.rows);
////					mat pointchunk = points.rowrange(k * samplecount / clustercount,
////						k == clustercount - 1 ? samplecount :
////						(k + 1) * samplecount / clustercount);
////					rng.fill(pointchunk, rng::normal, scalar(center.x, center.y), scalar(grayscale.cols * 0.05, grayscale.rows * 0.05));
////				}
////
////				randshuffle(points, 1, &rng);
////
////				double compactness = kmeans(points, clustercount, labels,
////					termcriteria(termcriteria::eps + termcriteria::count, 10, 1.0),
////					3, kmeans_pp_centers, centers);
////
////				grayscale = scalar::all(0);
////
////				for (i = 0; i < samplecount; i++)
////				{
////					int clusteridx = labels.at<int>(i);
////					point ipt = points.at<point2f>(i);
////					circle(grayscale, ipt, 2, colortab[clusteridx], filled, line_aa);
////				}
////				for (i = 0; i < (int)centers.size(); ++i)
////				{
////					point2f c = centers[i];
////					circle(grayscale, c, 40, colortab[i], 1, line_aa);
////				}
////				cout << "compactness: " << compactness << endl;
////
////				imshow("clusters", grayscale);
////
////				char key = (char)waitkey();
////				if (key == 27 || key == 'q' || key == 'q') // 'esc'
////					break;
////			}
////
////			return 0;
////		}
////	}
////}
////
////
////
////
////
////
////
//// ----------------------------------------------------TILL LINE 960  
////https://github.com/aditya1601/kmeans-clustering-cpp/blob/master/kmeans.cpp
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <fstream>
//#include <sstream>
//#include <algorithm>
//
//using namespace std;
//
//class Point {
//
//private:
//    int pointId, clusterId;
//    int dimensions;
//    vector<double> values;
//
//public:
//    Point(int id, string line) {
//        dimensions = 0;
//        pointId = id;
//        stringstream is(line);
//        double val;
//        while (is >> val) {
//            values.push_back(val);
//            dimensions++;
//        }
//        clusterId = 0; //Initially not assigned to any cluster
//    }
//
//    int getDimensions() {
//        return dimensions;
//    }
//
//    int getCluster() {
//        return clusterId;
//    }
//
//    int getID() {
//        return pointId;
//    }
//
//    void setCluster(int val) {
//        clusterId = val;
//    }
//
//    double getVal(int pos) {
//        return values[pos];
//    }
//};
//
//class Cluster {
//
//private:
//    int clusterId;
//    vector<double> centroid;
//    vector<Point> points;
//
//public:
//    Cluster(int clusterId, Point centroid) {
//        this->clusterId = clusterId;
//        for (int i = 0; i < centroid.getDimensions(); i++) {
//            this->centroid.push_back(centroid.getVal(i));
//        }
//        this->addPoint(centroid);
//    }
//
//    void addPoint(Point p) {
//        p.setCluster(this->clusterId);
//        points.push_back(p);
//    }
//
//    bool removePoint(int pointId) {
//        int size = points.size();
//
//        for (int i = 0; i < size; i++)
//        {
//            if (points[i].getID() == pointId)
//            {
//                points.erase(points.begin() + i);
//                return true;
//            }
//        }
//        return false;
//    }
//
//    int getId() {
//        return clusterId;
//    }
//
//    Point getPoint(int pos) {
//        return points[pos];
//    }
//
//    int getSize() {
//        return points.size();
//    }
//
//    double getCentroidByPos(int pos) {
//        return centroid[pos];
//    }
//
//    void setCentroidByPos(int pos, double val) {
//        this->centroid[pos] = val;
//    }
//};
//
//class KMeans {
//private:
//    int K, iters, dimensions, total_points;
//    vector<Cluster> clusters;
//
//    int getNearestClusterId(Point point) {
//        double sum = 0.0, min_dist;
//        int NearestClusterId;
//
//        for (int i = 0; i < dimensions; i++)
//        {
//            sum += pow(clusters[0].getCentroidByPos(i) - point.getVal(i), 2.0);
//        }
//
//        min_dist = sqrt(sum);
//        NearestClusterId = clusters[0].getId();
//
//        for (int i = 1; i < K; i++)
//        {
//            double dist;
//            sum = 0.0;
//
//            for (int j = 0; j < dimensions; j++)
//            {
//                sum += pow(clusters[i].getCentroidByPos(j) - point.getVal(j), 2.0);
//            }
//
//            dist = sqrt(sum);
//
//            if (dist < min_dist)
//            {
//                min_dist = dist;
//                NearestClusterId = clusters[i].getId();
//            }
//        }
//
//        return NearestClusterId;
//    }
//
//public:
//    KMeans(int K, int iterations) {
//        this->K = K;
//        this->iters = iterations;
//    }
//
//    void run(vector<Point>& all_points) {
//
//        total_points = all_points.size();
//        dimensions = all_points[0].getDimensions();
//
//
//        //Initializing Clusters
//        vector<int> used_pointIds;
//
//        for (int i = 1; i <= K; i++)
//        {
//            while (true)
//            {
//                int index = rand() % total_points;
//
//                if (find(used_pointIds.begin(), used_pointIds.end(), index) == used_pointIds.end())
//                {
//                    used_pointIds.push_back(index);
//                    all_points[index].setCluster(i);
//                    Cluster cluster(i, all_points[index]);
//                    clusters.push_back(cluster);
//                    break;
//                }
//            }
//        }
//        cout << "Clusters initialized = " << clusters.size() << endl << endl;
//
//
//        cout << "Running K-Means Clustering.." << endl;
//
//        int iter = 1;
//        while (true)
//        {
//            cout << "Iter - " << iter << "/" << iters << endl;
//            bool done = true;
//
//            // Add all points to their nearest cluster
//            for (int i = 0; i < total_points; i++)
//            {
//                int currentClusterId = all_points[i].getCluster();
//                int nearestClusterId = getNearestClusterId(all_points[i]);
//
//                if (currentClusterId != nearestClusterId)
//                {
//                    if (currentClusterId != 0) {
//                        for (int j = 0; j < K; j++) {
//                            if (clusters[j].getId() == currentClusterId) {
//                                clusters[j].removePoint(all_points[i].getID());
//                            }
//                        }
//                    }
//
//                    for (int j = 0; j < K; j++) {
//                        if (clusters[j].getId() == nearestClusterId) {
//                            clusters[j].addPoint(all_points[i]);
//                        }
//                    }
//                    all_points[i].setCluster(nearestClusterId);
//                    done = false;
//                }
//            }
//
//            // Recalculating the center of each cluster
//            for (int i = 0; i < K; i++)
//            {
//                int ClusterSize = clusters[i].getSize();
//
//                for (int j = 0; j < dimensions; j++)
//                {
//                    double sum = 0.0;
//                    if (ClusterSize > 0)
//                    {
//                        for (int p = 0; p < ClusterSize; p++)
//                            sum += clusters[i].getPoint(p).getVal(j);
//                        clusters[i].setCentroidByPos(j, sum / ClusterSize);
//                    }
//                }
//            }
//
//            if (done || iter >= iters)
//            {
//                cout << "Clustering completed in iteration : " << iter << endl << endl;
//                break;
//            }
//            iter++;
//        }
//
//
//        //Print pointIds in each cluster
//        for (int i = 0; i < K; i++) {
//            cout << "Points in cluster " << clusters[i].getId() << " : ";
//            for (int j = 0; j < clusters[i].getSize(); j++) {
//                cout << clusters[i].getPoint(j).getID() << " ";
//            }
//            cout << endl << endl;
//        }
//        cout << "========================" << endl << endl;
//
//        //Write cluster centers to file
//        ofstream outfile;
//        outfile.open("clusters.txt");
//        if (outfile.is_open()) {
//            for (int i = 0; i < K; i++) {
//                cout << "Cluster " << clusters[i].getId() << " centroid : ";
//                for (int j = 0; j < dimensions; j++) {
//                    cout << clusters[i].getCentroidByPos(j) << " ";     //Output to console
//                    outfile << clusters[i].getCentroidByPos(j) << " ";  //Output to file
//                }
//                cout << endl;
//                outfile << endl;
//            }
//            outfile.close();
//        }
//        else {
//            cout << "Error: Unable to write to clusters.txt";
//        }
//
//    }
//};
// 
//int main(int argc, char** argv) {
//
//    //Need 2 arguments (except filename) to run, else exit
//    if (argc != 3) {
//        cout << "Error: command-line argument count mismatch.";
//        return 1;
//    }
//
//    //Fetching number of clusters
//    int K = atoi(argv[2]);
//
//    //Open file for fetching points
//    string filename = argv[1];
//    ifstream infile(filename.c_str());
//
//    if (!infile.is_open()) {
//        cout << "Error: Failed to open file." << endl;
//        return 1;
//    }
//
//    //Fetching points from file
//    int pointId = 1;
//    vector<Point> all_points;
//    string line;
//
//    while (getline(infile, line)) {
//        Point point(pointId, line);
//        all_points.push_back(point);
//        pointId++;
//    }
//    infile.close();
//    cout << "\nData fetched successfully!" << endl << endl;
//
//    //Return if number of clusters > number of points
//    if (all_points.size() < K) {
//        cout << "Error: Number of clusters greater than number of points." << endl;
//        return 1;
//    }
//
//    //Running K-Means Clustering
//    int iters = 100;
//
//    KMeans kmeans(K, iters);
//    kmeans.run(all_points);
//
//    return 0;
//}
//
// -----------------------------------TILL LINE 

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video.hpp>


using namespace cv;
using namespace dnn;
using namespace saliency;
using namespace std;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image


vector<string> classes;
vector<Mat> Segmentation(Mat src, Mat rgb);
vector<String> getOutputsNames(const Net& net);

Mat KMeans(Mat src, int clusterCount);

Mat EqualizeHisto(Mat frame);

Mat Blur(Mat frame, int winsize);

Mat dilation(Mat frame, int winsize);

Mat erosion(Mat frame, int winsize);

void FinalFrame(Mat frame);

void postprocess(Mat& frame, const vector<Mat>& out);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);


void FinalOutput(Mat frame) {
    Mat output = frame.clone();
    EqualizeHisto(output);
    imshow ("",output);
    waitKey();
    Blur(output, 1);
    Ptr<StaticSaliencySpectralResidual> Saliency = StaticSaliencySpectralResidual::create();
    Mat srMask, fgMask;
    //(input img, output img)
    Saliency->computeSaliency(output, fgMask);
    fgMask.convertTo(fgMask, CV_8U, 255);

    cvtColor(fgMask, fgMask, COLOR_GRAY2BGR);
    vector<Mat> segment = Segmentation(KMeans(fgMask, 4), frame); //k = 4

    for (Mat image : segment) {
        FinalFrame(image);
    }

}

Mat KMeans(Mat src, int clusterCount) {
    Mat means(src.rows * src.cols, 3, CV_32F);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
            for (int k = 0; k < 3; k++)
                means.at<float>(i + j * src.rows, k) = src.at<Vec3b>(i, j)[k];

    Mat labels;
    Mat centers;
    int attempts = 5;
    kmeans(means, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

    Mat image = Mat::zeros(src.size(), src.type());
    Vec2i pointVal = { 0, 0 };

    //Get highest intensity
    for (int i = 0; i < centers.rows; i++) {
        int sum = 0;
        for (int j = 0; j < centers.cols; j++) {
            sum += centers.at<float>(i, j);
        }
        if (sum / 3 > pointVal[1]) {
            pointVal[0] = i;
            pointVal[1] = sum / 3;
        }
    }

    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
        {
            int cluster = labels.at<int>(i + j * src.rows, 0);
            if (cluster == pointVal[0]) {
                image.at<Vec3b>(i, j)[0] = centers.at<float>(cluster, 0);
                image.at<Vec3b>(i, j)[1] = centers.at<float>(cluster, 1);
                image.at<Vec3b>(i, j)[2] = centers.at<float>(cluster, 2);
            }
        }

    cvtColor(image, image, COLOR_BGR2GRAY);
    imshow("Kmeans", image);
    waitKey();
    return image;

}

vector<Mat> Segmentation(Mat src, Mat original) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Mat> segment;

    Mat src2 = dilation(src, 20);
    Mat src3 = erosion(src2, 5);

    findContours(src3, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(src3.size(), CV_8UC3);
    // Original image clone
    RNG rng(12345);

    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        Rect rec = boundingRect(contours.at(i));
        double ratio = rec.width / rec.height;

        if (rec.width < original.cols * 0.1 || rec.height < original.rows * 0.15 || rec.y < original.rows * 0.1 || (rec.x < original.cols * 0.1) || ratio > 5.0) {
            continue;
        }

        segment.push_back(original(rec));
    }
    return segment;

}


int main() {
   // VideoCapture capture("D:\\year 3\\IPPR\\test.mp4");
    VideoCapture capture("C:\\Users\\Mufaddal Murtaza\\Downloads\\Car.mp4");

   // Mat temp = imread("D:\\School\\SEM 5\\IPPS\\Assignment Picture\\Greyscale.jpg");
   
 

    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    int fps = capture.get(CAP_PROP_FPS) / 10;
    VideoWriter video("output.mp4v", VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(frame_width, frame_height));

    while (true)
    {
        Mat frame;
        for (int i = 0; i < 10; i++)
            capture >> frame;

        if (frame.empty())
            break;

        FinalOutput(frame);
        video.write(frame);
        imshow("YOLO Final Output", frame);
        char c = (char)waitKey(30);
        if (c == 27)
            break;
    }
    capture.release();
    video.release();

}

void FinalFrame(Mat frame) {
    //imshow("processing frame", frame);
    //waitKey();
    // Load names of classes    
    string classesFile = "D:\\year 3\\IPPR\\darknet-master\\data\\coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    String modelConfiguration = "D:\\year 3\\IPPR\\darknet-master\\cfg\\yolov3.cfg";
    String modelWeights = "D:\\year 3\\IPPR\\darknet-master\\yolov3.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Create a 4D blob from a frame.
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

    //Set the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);

}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

Mat EqualizeHisto(Mat eqgrey)
{
    int count[256] = { 0 };
    float prob[256] = { 0.0 };
    float accprob[256] = { 0.0 };
    int newpixel[256] = { 0 };

    Mat eqhisto = Mat::zeros(eqgrey.size(), CV_8UC1);

    for (int i = 0; i < eqgrey.rows; i++)//count, adding into array
    {
        for (int j = 0; j < eqgrey.cols; j++)
        {
            count[eqgrey.at<uchar>(i, j)]++;
        }
    }

    for (int k = 0; k < 256; k++)//probablity
    {
        prob[k] = (float)count[k] / (float)(eqgrey.rows * eqgrey.cols);
    }

    accprob[0] = prob[0];

    for (int a = 1; a < 256; a++)//acc probability
    {
        accprob[a] = (prob[a] + accprob[a - 1]);
    }

    for (int i = 0; i < 256; i++)//Multiply to find NEW PIXEL
    {
        newpixel[i] = accprob[i] * 255;
    }

    for (int i = 0; i < eqgrey.rows; i++)//Change Old image to new pixel image
    {
        for (int j = 0; j < eqgrey.cols; j++)
        {
            eqhisto.at<uchar>(i, j) = newpixel[eqgrey.at<uchar>(i, j)];
        }
    }
    //return
    return eqhisto;
}

Mat Blur(Mat grey, int winsize)
{
    Mat blurimg = Mat::zeros(grey.size(), CV_8UC1);
    if (winsize == 1)
    {
        for (int i = winsize; i < (grey.rows - winsize); i++)
        {
            for (int j = winsize; j < (grey.cols - winsize); j++)
            {
                int sum = 0;

                for (int ii = (-winsize); ii <= (+winsize); ii++)
                {
                    for (int jj = (-winsize); jj <= (+winsize); jj++)
                    {
                        sum += grey.at<uchar>(i + ii, j + jj);
                    }
                    blurimg.at<uchar>(i, j) = sum / (((winsize * 2) + 1) * ((winsize * 2) + 1)); // ((winsize*2)+1)^2;

                }
            }
        }
    }
    else if (winsize == 0)
    {
        for (int i = winsize; i < (grey.rows - winsize); i++)
        {
            for (int j = winsize; j < (grey.cols - winsize); j++)
            {
                blurimg.at<uchar>(i, j) = grey.at<uchar>(i, j);
            }
        }
    }

    return blurimg;
}


Mat dilation(Mat binary, int winsize)
{
    Mat dilate_img = Mat::zeros(binary.size(), CV_8UC1);
    for (int i = winsize; i < (binary.rows - winsize); i++)
    {
        for (int j = winsize; j < (binary.cols - winsize); j++)
        {
            for (int ii = (-winsize); ii <= (+winsize); ii++)
            {
                for (int jj = (-winsize); jj <= (+winsize); jj++)
                {
                    if (binary.at<uchar>(i + ii, j + jj) > 0)
                    {
                        dilate_img.at<uchar>(i, j) = 255;
                    }
                }
            }
        }

    }
    return dilate_img;
}

Mat erosion(Mat Dilation, int winsize)
{
    Mat erosion_img = Mat::zeros(Dilation.size(), CV_8UC1);

    for (int i = winsize; i < (Dilation.rows - winsize); i++)
    {
        for (int j = winsize; j < (Dilation.cols - winsize); j++)
        {
            int newwhite = 0;
            for (int ii = (-winsize); ii <= (+winsize); ii = ii++)
            {
                for (int jj = (-winsize); jj <= (+winsize); jj = jj++)
                {

                    if (Dilation.at<uchar>(i + ii, j + jj) > 0)
                    {
                        newwhite++;
                    }
                }
            }
            int newwinsize = ((1 + (winsize * 2)) * (1 + (winsize * 2)));
            if (newwhite == newwinsize)
            {
                erosion_img.at<uchar>(i, j) = 255;
            }
        }

    }
    return erosion_img;
}
