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
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <chrono>
//#include <cuda.h>
//#include <cuda_runtime.h>


using namespace std ;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::cuda;
using namespace std::chrono;

void readme();
 





/* @function main */
int main( int argc, char** argv )
{

		
		
		
		Mat image1 = imread("pano2.jpg");  //	RIGHT IMAGE 
		Mat image2 = imread( "pano1.jpg"); //	LEFT IMAGE 

		GpuMat gpu_img1;
		GpuMat gpu_img2;
    
		Size size(1024,780);

		resize(image1,image1,size);
		resize(image2,image2,size);
		
		gpu_img1.upload(image1);
		gpu_img2.upload(image2);

		Mat gray_image1;
		Mat gray_image2;

        GpuMat gpu_gray1;
		GpuMat gpu_gray2;

		gpu_gray1.upload(gray_image1);
		gpu_gray2.upload(gray_image2);
		
		cv::cuda::Stream colorStream[2];
		

		//Covert to Grayscale
		cv::cuda::cvtColor( gpu_img1, gpu_gray1, COLOR_RGB2GRAY,1, colorStream[0] );
		cv::cuda::cvtColor( gpu_img2, gpu_gray2, COLOR_RGB2GRAY,1, colorStream[1] );

		// Start counting from here : 
		
		auto start = high_resolution_clock::now(); 
		//--Step 1 : Detect the keypoints using SURF Detector
		SURF_CUDA surf;

        GpuMat keypoints1GPU, keypoints2GPU;
        GpuMat descriptors1GPU, descriptors2GPU;

		surf(gpu_gray1,GpuMat(),keypoints1GPU,descriptors1GPU);
        surf(gpu_gray2,GpuMat(),keypoints2GPU,descriptors2GPU);

        Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
        vector<DMatch> matches;
        matcher->match(descriptors1GPU, descriptors2GPU, matches);


		double max_dist = 0;
		double min_dist = 100;


        vector<KeyPoint> keypoints1, keypoints2;
        vector<float> descriptors1, descriptors2;
        surf.downloadKeypoints(keypoints1GPU, keypoints1);
        surf.downloadKeypoints(keypoints2GPU, keypoints2);
        


        Mat descr1;
        descriptors1GPU.download(descr1);

    // Here Stops GPU and Starts CPU work 


		//--Quick calculation of min-max distances between keypoints

        cout<<"Rows : "<<descr1.rows<<endl;


        for(int i =0; i < descr1.rows ; i++)
		{

                double dist = matches[i].distance;
				if( dist < min_dist ) min_dist = dist;
				else if( dist > max_dist ) max_dist = dist;
		}


		
		
		printf("-- Max dist : %f \n", max_dist );
		printf("-- Min dist : %f \n", min_dist );

		//--Use only "good" matches (i.e. whose distance is less than 3 X min_dist )
		std::vector< DMatch > good_matches;
		



		for(int i =0 ; i < descr1.rows ; i++)
		{
				if( matches[i].distance < 2*max_dist )
				{
						good_matches.push_back( matches[i] );
				}
		}
		
        vector< Point2f > obj;
		vector< Point2f > scene;


		for( int i = 0; i < good_matches.size(); i++)
		{
				//--Get the keypoints from the good matches
				obj.push_back( keypoints1[good_matches[i].queryIdx].pt );
				scene.push_back( keypoints2[good_matches[i].trainIdx].pt );
		}

		//Find the Homography Matrix
		Mat H = findHomography( obj, scene, RANSAC );

		// Use the homography Matrix to warp the images
		Mat result;
        
		warpPerspective( image1, result, H, cv::Size( image1.cols+image2.cols, image1.rows));
		Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows) );

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
		
						}else break;
				}

		


		
		// create bounding rect around those points
		cv::Rect bb = cv::boundingRect(nonBlackList);
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start); 
	    cout<<"Total CUDA Time microsecs : "<<duration.count()<<endl;
		// display result and save it
		//imshow("Reult", result(bb));
		//imshow("Img2",image2);
        imwrite( "panoCUDARESULT.jpg", result );


		waitKey(0);

		return 0;
}

/** function readme */
void readme()
{
		std::cout << " Usage: pano < img1 > < img2 > " <<std::endl;
}
