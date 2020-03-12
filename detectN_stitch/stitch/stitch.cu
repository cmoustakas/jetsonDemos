
/** 
    Test Code in order to catch detectNStich.cu code's concept 
**/


#include <iostream>
#include <vector>
#include <chrono>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

/** OpenCV host Libraries **/

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

/** OpenCV CUDA Libraries **/
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>

using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace cv::cuda;
using namespace cv::xfeatures2d;

#define RIGHT 0
#define LEFT  2
#define UP -1
#define DOWN 1


/**
    This Kernel Implemented in order to set a more efficient mask 
    to keypoints and descriptor detection  in an Region Of Interest
    
**/ 


__global__ void mask_kernel(unsigned char* image,unsigned char* mask,int cornerX,int cornerY,int width,int height,int colorStep,int maskStep){
    
    const int xIndex = blockIdx.x*blockDim.x+threadIdx.x;
    const int yIndex = blockIdx.y*blockDim.y+threadIdx.y;
    bool inside = ((xIndex > cornerX) && (xIndex < cornerX +width) && (yIndex < cornerY + height) && (yIndex > cornerY));
    
    const int maskIdx = yIndex*maskStep + xIndex;

  

    if(inside){
    
        const int colorIdx = yIndex*colorStep + 3*xIndex;
        
        unsigned char b = image[colorIdx];
        unsigned char g = image[colorIdx+1];
        unsigned char r = image[colorIdx+2];

        float gray = r*0.3f + g*0.59f + b*0.11f;
        mask[maskIdx] = static_cast<unsigned char>(gray);

    }
    else
       mask[maskIdx] = 0.0f;



}


cv::Mat stitchFrames(Mat& frame1,Mat& frame2,int overlap){
    
/**

	Size size(1024,780);

	cv::resize(frame1,frame1,size);
	cv::resize(frame2,frame2,size);


    Mat mask1(frame1.rows,frame1.cols,CV_8UC1);
    Mat mask2(frame2.rows,frame2.cols,CV_8UC1);
    
    const int bytes_f1  = frame1.step*frame1.rows;
    const int bytes_f2  = frame2.step*frame2.rows;
    const int bytes_m1  = mask1.step*mask1.rows;
    const int bytes_m2  = mask2.step*mask2.rows;


    unsigned char *device_mask1,*device_mask2,*d_frame1,*d_frame2;
    
    cudaMalloc(&device_mask1,bytes_m1);
    cudaMalloc(&device_mask2,bytes_m2);

    cudaMalloc(&d_frame1,bytes_f1);
    cudaMalloc(&d_frame2,bytes_f2);

    cudaMemcpy(d_frame1,frame1.ptr(),bytes_f1,cudaMemcpyHostToDevice);
    cudaMemcpy(d_frame2,frame2.ptr(),bytes_f2,cudaMemcpyHostToDevice);
    
    const dim3 block(16,16);
    const dim3 grid(((frame1.cols)+block.x-1)/block.x,((frame2.rows)+block.y-1)/block.y);
    
    
    int cornerY = 0;
    mask_kernel<<<grid,block>>>(d_frame1,device_mask1,overlap,cornerY,3*frame1.cols/4,frame1.rows,frame1.step,mask1.step);
    cudaDeviceSynchronize();

    mask_kernel<<<grid,block>>>(d_frame2,device_mask2,0,cornerY,3*frame2.cols/4,frame2.rows,frame2.step,mask2.step);
    cudaDeviceSynchronize();

    cudaMemcpy(mask1.ptr(),device_mask1,bytes_m1,cudaMemcpyDeviceToHost);
    cudaMemcpy(mask2.ptr(),device_mask2,bytes_m2,cudaMemcpyDeviceToHost);

    imshow("mask1",mask1);
    imshow("mask2",mask2);
    **/

    GpuMat gpu_mask1,gpu_mask2;
    GpuMat d_frame1(frame1),d_frame2(frame2);

    cuda::cvtColor(d_frame1,gpu_mask1,COLOR_BGR2GRAY,1);
    cuda::cvtColor(d_frame2,gpu_mask2,COLOR_BGR2GRAY,1);

    Mat mask1(gpu_mask1);
    Mat mask2(gpu_mask2);
/**
    imshow("mask1",mask1);
    imshow("mask2",mask2);
**/
    imshow("fr1",frame1);
    imshow("fr2",frame2);
    waitKey();
 

    // Start counting from here : 

    auto start = high_resolution_clock::now(); 
    //--Step 1 : Detect the keypoints using SURF Detector
    SURF_CUDA surf;

    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;

    surf(gpu_mask1,GpuMat(),keypoints1GPU,descriptors1GPU);
    surf(gpu_mask2,GpuMat(),keypoints2GPU,descriptors2GPU);


    /** Match Descriptors **/
    Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
    vector<DMatch> matches;
    matcher->match(descriptors1GPU, descriptors2GPU,matches);


    double max_dist = 0;
    double min_dist = 100;


    vector<KeyPoint> keypoints1, keypoints2;
    vector<float> descriptors1, descriptors2;
    surf.downloadKeypoints(keypoints1GPU, keypoints1);
    surf.downloadKeypoints(keypoints2GPU, keypoints2);
    
    /** Download gpu keypoints and descriptors **/
    /** Because of the small size of each operation, gpu use is not worth **/
    /** ---------   **/

    Mat descr1;
    descriptors1GPU.download(descr1);

    // Here Stops GPU and Starts CPU job 

    cout<<"Rows : "<<descr1.rows<<endl;
    
    
    int r = descr1.rows;

    for(int i =0; i < descr1.rows ; i++)
    {
        double dist = matches[i].distance;
        if(dist == 0.f){
            r=i;
            break;
        }

        if( dist < min_dist && dist > 0 ) min_dist = dist;
        else if( dist > max_dist ) max_dist = dist;
    }


    Mat img_matches;
    drawMatches(Mat(frame1), keypoints1, Mat(frame2), keypoints2, matches, img_matches);

    namedWindow("matches", 0);
    
    imwrite("/home/robin/Desktop/droneFootages/matches.jpg", img_matches);




    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //--Use only "good" matches (i.e. whose distance is less than 3 X min_dist )
    std::vector< DMatch > good_matches;


/** Keep Only linear matches,because of vertical footage ,non-linear matches act like noise in the system **/

    for(int i =0 ; i < r ; i++)
    {
        
        int idx2 = matches[i].trainIdx;
        int idx1 = matches[i].queryIdx;
       
        int theta = atan2((double)(keypoints2[idx2].pt.y- keypoints1[idx1].pt.y),(double)(keypoints2[idx2].pt.x-keypoints1[idx1].pt.x + frame1.cols))*180/CV_PI ;
        //printf("%d degr \n",theta);

        if(  abs(theta)<10 && matches[i].distance < 4*min_dist) {
            good_matches.push_back( matches[i] );
        }
        
    }


    Mat img_Gmatches;
    drawMatches(Mat(frame1), keypoints1, Mat(frame2), keypoints2, good_matches, img_Gmatches);

    namedWindow("Gmatches", 0);
    
    imwrite("/home/robin/Desktop/droneFootages/gmatches.jpg", img_Gmatches);
    //waitKey(0);



    vector< Point2f > obj;
    vector< Point2f > scene;

    for( int i = 0; i < good_matches.size(); i++)
    {
        //--Get the keypoints from the good matches
        obj.push_back( keypoints1[good_matches[i].queryIdx].pt );
        scene.push_back( keypoints2[good_matches[i].trainIdx].pt );
    }

    // Homography Matrix
    Mat H = findHomography(scene,obj,RANSAC);
    
    /** Use the homography Matrix to warp the images **/
    Mat result;
    cv::warpPerspective(frame2, result, H, cv::Size(frame1.cols+overlap,frame1.rows));
    
    /**
    imshow("afterWarp",result);
    waitKey();
    **/
    
    Mat half(result, cv::Rect(0, 0, frame2.cols, frame2.rows) );
    
    /**
    imshow("half1",half);
    waitKey();
    **/

    frame1.copyTo(half);

    /**
    imshow("half2",half);
    waitKey();
    **/

   
    auto stop = high_resolution_clock::now();
    double elapsed_time_ms = duration<double, std::milli>(stop-start).count();
    cout<<"Total CUDA Time millisec : "<<elapsed_time_ms<<endl;
    
    imshow("PANO",result);

    waitKey();
    
 
    return result;
    

}

int nearestTo(double direction,int theta1,int theta2){
    
    int distance1 = abs(direction);
    int distance2 = abs(abs(direction) - abs(theta1));
    int distance3 = abs(abs(direction) - abs(theta2));
    int min = distance1;
    int ret = 0;

    if(distance2<min){
        min = distance2;
        ret = 1;
    }
    if(distance3 < min){
        min = distance3;
        ret = 2;
    }
    return ret ;

}

static void downloadP(const GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void downloadU(const GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}



static void drawArrows(Mat& frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status, Scalar line_color = Scalar(0, 0, 255))
{
    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i])
        {
            int line_thickness = 1;

            Point p = prevPts[i];
            Point q = nextPts[i];

            double angle = atan2((double) p.y - q.y, (double) p.x - q.x);

            double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );

            if (hypotenuse < 1.0)
                continue;

            // Here we lengthen the arrow by a factor of three.
            q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
            q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

            // Now we draw the main line of the arrow.
            line(frame, p, q, line_color, line_thickness);

            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.

            p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);

            p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}



int* decideDirection(const vector<Point2f>& prevPts,const vector<Point2f>& nextPts,const vector<uchar>&status){
    double direction,host=0,theta,hypotenuse ;
    int* retPtr = (int*)malloc(2*sizeof(int));
    for(size_t i =0;i<prevPts.size();i++){
        if(status[i]){
            host++;
            Point p = prevPts[i];
            Point n = nextPts[i];
            
            // Because of 1:1 tangent's nature there is no possible duplicate scenario 
            theta = atan2((double)p.y - n.y,(double)p.x-n.x);
            theta = theta*180/CV_PI ;  
            direction += theta;
            hypotenuse += sqrt( (double)(p.y - n.y)*(p.y - n.y) + (double)(p.x - n.x)*(p.x - n.x) );

            
        }
    }
    
    direction = direction/host;

    //Average Displacement Calculation 
    double averageDispl = hypotenuse/host;
    printf("Average Displacement : %d \n",(int)averageDispl);

    int quant;


    // Quantize Angles as {0,90,180,-90} [+][+][+][+][+]

    if(direction < 0){
        quant = nearestTo(direction,-90,-180);
        if(quant == 1)
            quant = -quant;    
    }
    else if(direction >0)
        quant = nearestTo(direction,90,180);
    
   
    
    printf("[+][+] Direction decided, camera is moving :  ");
    
    if(quant == 1)printf("DOWN \n");
    else if(quant == -1)printf("UP \n");
    else if(quant == 0)printf("RIGHT \n");
    else printf("LEFT \n");
    
    retPtr[0] = quant;
    retPtr[1] = averageDispl;
    
    return retPtr;

}




int main(int argc,char *argv[]){
    
    string videoFile;
    if(argc>1) videoFile = argv[1];
    else return -1;

    VideoCapture cap(videoFile);
    if(!cap.isOpened())return -1;
    

    Mat frame1;
    Mat frame2;

    // Captute two respective Frames 
    for(int i =0;i<20;i++)
        cap >> frame1 ; //Drop three frames 
    cap >> frame2;
    
    
    
    GpuMat dev_frame1(frame1);
    GpuMat dev_frame2(frame2);
    GpuMat dev_frame1_Gray;
    GpuMat dev_prevPts;
    
    
    //Start counting time [+][+]
    
    
    
    // Fast convertion from BGR to Gray
    cuda::cvtColor(dev_frame1,dev_frame1_Gray,COLOR_BGR2GRAY,1);


    //Detect only good corners 
    auto t_start = high_resolution_clock::now();
    //GpuMat d_ROI = dev_frame1_Gray(Rect(0,0,frame1.cols/2,frame1.rows/2));
    cv::cuda::Stream strm;

    Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(dev_frame1_Gray.type(), 500, 0.01, 1);
    detector->detect(dev_frame1_Gray, dev_prevPts,cv::noArray(),strm);
    

    GpuMat dev_nextPts;
    GpuMat dev_status;

    Ptr<cuda::SparsePyrLKOpticalFlow> opticalFlowObj = cuda::SparsePyrLKOpticalFlow::create(Size(21,21),3,30,false);
    opticalFlowObj->calc(dev_frame1,dev_frame2,dev_prevPts,dev_nextPts,dev_status);

    vector<Point2f> prevPts(dev_prevPts.cols);
    downloadP(dev_prevPts, prevPts);

    vector<Point2f> nextPts(dev_nextPts.cols);
    downloadP(dev_nextPts, nextPts);

    vector<uchar> status(dev_status.cols);
    downloadU(dev_status, status);
     
    
    int* directionPtr = (int*)malloc(2*sizeof(int));
    int frameBound;
    directionPtr = decideDirection(prevPts,nextPts,status);
    if(directionPtr[1] == 0){
        cout<<"Drone's velocity shouldn't be equal to 0 [-][-][-] \n"<<endl;
        return 0;
    }
    
    auto t_dir_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = duration<double, std::milli>(t_dir_end-t_start).count();
    cout << "Optical FLow total milisec. time : "<<elapsed_time_ms<<endl;

    cv::cuda::Stream rotStream[2];

    // Number of Frames should be passed in order  to catch the right frame for stitching   
    if(directionPtr[0] == UP || directionPtr[0] == DOWN){
        
        Mat fr1,fr2;

        cv::cuda::rotate( dev_frame1, dev_frame1, cv::Size( frame1.rows, frame1.cols ), -90, frame1.rows, 0, cv::INTER_LINEAR);
        dev_frame1.download(fr1);
        
        frameBound = (frame1.rows)/(directionPtr[1])  - 2;

        for(int i=0;i<(int)frameBound/4;i++){
            cap>>frame2;
            
        }
        GpuMat d_frame2(frame2);
        cv::cuda::rotate( d_frame2, d_frame2, cv::Size( frame1.rows, frame1.cols ), -90, frame1.rows,0, cv::INTER_LINEAR);
        
        d_frame2.download(fr2);
        fr1 = stitchFrames(fr1,fr2,frame1.cols/4);
        
    
    }
    else{
        frameBound = (frame1.cols)/(directionPtr[1])  - 2;
        

        for(int i=0;i<frameBound/4;i++){
            cap>>frame2;
        }  
        frame1 = stitchFrames(frame1,frame2,frame1.cols/3);
        
    }
    printf("frameBound : %d \n",frameBound);

    
   drawArrows(frame1, prevPts,nextPts,status,Scalar(0, 0, 255));
   imshow("frame1",frame1);
   waitKey();
    

    return 0;

}

