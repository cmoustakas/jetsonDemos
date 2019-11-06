#include <stdio.h>
#include <iostream>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std::chrono;

void readme();

/* @function main */
int main( int argc, char** argv )
{

		//Load the images
		Mat image1 = imread("pano2.jpg");
		Mat image2 = imread( "pano1.jpg");
    
    
    

		Size size(1024,780);


		resize(image1,image1,size);
		resize(image2,image2,size);

		Mat gray_image1;
		Mat gray_image2;

		//Covert to Grayscale
		cvtColor( image1, gray_image1, COLOR_RGB2GRAY );
		cvtColor( image2, gray_image2, COLOR_RGB2GRAY );
/**
		imshow( "First Image", image2 );
		imshow( "Second Image", image1 );
**/



		if ( !gray_image1.data || !gray_image2.data )
		{
				std::cout << " --(!) Error reading images " << std::endl;
				return -1;
		}


		//--Step 1 : Detect the keypoints using SURF Detector

		int minHessian = 400;
		auto start = high_resolution_clock::now(); 

    Ptr<SURF> detector = SURF::create( minHessian );
		std::vector< KeyPoint > keypoints_object, keypoints_scene;

		detector->detect( gray_image1, keypoints_object );
		detector->detect( gray_image2, keypoints_scene );

		//--Step 2 : Calculate Descriptors (feature vectors)
		Ptr<SURF> extractor = SURF::create();

		Mat descriptors_object,descriptors_scene;

		extractor->compute( gray_image1, keypoints_object, descriptors_object );
		extractor->compute( gray_image2, keypoints_scene, descriptors_scene );

		//--Step 3 : Matching descriptor vectors using FLANN matcher
		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match( descriptors_object, descriptors_scene, matches );

		double max_dist = 0;
		double min_dist = 100;

		cout<<"Rows :"<<descriptors_object.rows<<endl;
		//--Quick calculation of min-max distances between keypoints
		for(int i =0; i < descriptors_object.rows ; i++)
		{
				double dist = matches[i].distance;
				if( dist < min_dist ) min_dist = dist;
				if( dist > max_dist ) max_dist = dist;
		}

		printf("-- Max dist : %f \n", max_dist );
		printf("-- Min dist : %f \n", min_dist );

		//--Use only "good" matches (i.e. whose distance is less than 3 X min_dist )
		std::vector< DMatch > good_matches;




		for(int i =0 ; i < descriptors_object.rows ; i++)
		{
				if( matches[i].distance < 3*min_dist )
				{
						good_matches.push_back( matches[i] );
				}
		}
		
		
		std::vector< Point2f > obj;
		std::vector< Point2f > scene;


		for( int i = 0; i < good_matches.size(); i++)
		{
				//--Get the keypoints from the good matches
				obj.push_back( keypoints_object[good_matches[i].queryIdx].pt );
				scene.push_back( keypoints_scene[good_matches[i].trainIdx].pt );
		}



		//Find the Homography Matrix
		Mat H = findHomography( obj, scene, RANSAC );

		// Use the homography Matrix to warp the images
		cv::Mat result;

		warpPerspective( image1, result, H, cv::Size( image1.cols+image2.cols, image1.rows) );
		cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows) );
		image2.copyTo(half);

		/* To remove the black portion after stitching, and confine in a rectangular region*/

		// vector with all non-black point positions
		std::vector<cv::Point> nonBlackList;
		nonBlackList.reserve(result.rows*result.cols);

		// add all non-black points to the vector
		// there are more efficient ways to iterate through the image
		for(int j=0; j<result.rows; ++j)
				for(int i=0; i<result.cols; ++i)
				{
						// if not black: add to the list
						if(result.at<cv::Vec3b>(j,i) != cv::Vec3b(0,0,0))
						{
								nonBlackList.push_back(cv::Point(i,j));
						}
				}

		// create bounding rect around those points
		cv::Rect bb = cv::boundingRect(nonBlackList);
		
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start); 
	    cout<<"Total Serial Time microsecs : "<<duration.count()<<endl;

		// display result and save it
		cv::imshow("Reult", result(bb));
		
		imwrite( "panoRESULT.jpg", result );


		waitKey(0);

		return 0;
}

/** function readme */
void readme()
{
		std::cout << " Usage: pano < img1 > < img2 > " <<std::endl;
}
