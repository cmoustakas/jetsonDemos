/** 
	


	[+] Author : Chares Moustakas.
	[+] E-mail : cmoustakas@ece.auth.gr , charesmoustakas@gmail.com 
	[+] Professor : Nikolaos Pitsianis,Dimitrios Floros. 
	[+] University : Aristotle's University Of Thessaloniki 
	[+] Department : Electrical and Computer Engineering 


	[+] Project's Description : 
		Based On detect-net Nvidia's Algorithm the below code loads a pretrained  neural network and renders frames 
	with overlay boxes, in adittion also provides stitched images for further in-depth image recognition on a parallel 
	computing concept.

**/



#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <malloc.h>
#include <sched.h>
#include <string.h>
#include <sys/mman.h>


/** CUDA Libs **/
#include <cuda.h>
#include <cuda_runtime.h>

/** Jetson Inference & Utils Includes **/
#include <jetson-utils/gstCamera.h>
#include <jetson-utils/glDisplay.h>
#include <jetson-utils/Thread.h>
#include <jetson-inference/detectNet.h>
#include <jetson-utils/commandLine.h>
#include <jetson-utils/cudaRGB.h>


/** OpenCV host Libraries **/
#include <opencv2/core/version.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>


/** OpenCV CUDA Libraries **/
#include <opencv2/core/cuda.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>




/** Namespaces **/
using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace cv::cuda;
using namespace cv::xfeatures2d;


/** Constants **/
#define RIGHT 0
#define LEFT  2
#define UP -1
#define DOWN 1


/** Function's Decl. **/
void panoramaThread(float*leftRGBA,float*rightRGBA,bool rotate,int rows,int cols,bool penalty);
void predictDirection(float* img1,float* img2,int rows,int cols);
void decideDirection(const vector<Point2f>& prevPts,const vector<Point2f>& nextPts,const vector<uchar>&status);
int nearestTo(double direction,int theta1,int theta2);
void downloadP(const GpuMat& d_mat, vector<Point2f>& vec);
void downloadU(const GpuMat& d_mat, vector<uchar>& vec);


/** Global Variables **/
int dir,displ;

const int FPS_PENALTY=70; // Decide direction total time = 2.5 seconds , FPS = 25 => frame penalty =~ 60


bool signal_recieved = false;
bool predictDir = false;
bool spy = false;
bool threadFlag = false;

GpuMat d_frame1;


void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: detectnet-camera [-h] [--network NETWORK] [--threshold THRESHOLD]\n");
	printf("                        [--camera CAMERA] [--width WIDTH] [--height HEIGHT]\n\n");
	printf("Locate objects in a live camera stream using an object detection DNN.\n\n");
	printf("optional arguments:\n");
	printf("  --help            show this help message and exit\n");
	printf("  --network NETWORK pre-trained model to load (see below for options)\n");
	printf("  --overlay OVERLAY detection overlay flags (e.g. --overlay=box,labels,conf)\n");
	printf("                    valid combinations are:  'box', 'labels', 'conf', 'none'\n");
     printf("  --alpha ALPHA     overlay alpha blending value, range 0-255 (default: 120)\n");
	printf("  --camera CAMERA   index of the MIPI CSI camera to use (e.g. CSI camera 0),\n");
	printf("                    or for VL42 cameras the /dev/video device to use.\n");
     printf("                    by default, MIPI CSI camera 0 will be used.\n");
	printf("  --width WIDTH     desired width of camera stream (default is 1280 pixels)\n");
	printf("  --height HEIGHT   desired height of camera stream (default is 720 pixels)\n");
	printf("  --threshold VALUE minimum threshold for detection (default is 0.5)\n\n");

	printf("%s\n", detectNet::Usage());

	return 0;
}





int main( int argc, char** argv )
{
	
	static int PREALLOC_SIZE     = 200 * 1024 * 1024; // Preallocate 200MB for our Process [+]

	/** Disable paging for the current process **/
	mlockall(MCL_CURRENT | MCL_FUTURE);				// forgetting munlockall() when done!

	/** Turn off malloc trimming AKA hey teacher leave the heap alone . **/
	mallopt(M_TRIM_THRESHOLD, -1);

	/** Turn off mmap usage AKA virtual RAM . **/
	mallopt(M_MMAP_MAX, 0);

	unsigned int page_size = sysconf(_SC_PAGESIZE);
	unsigned char * buffer = (unsigned char *)malloc(PREALLOC_SIZE);

	/** Touch each page in this piece of memory to get it mapped into RAM **/
	for(int i = 0; i < PREALLOC_SIZE; i += page_size)
		buffer[i] = 0; /** This will generate pagefaults that will provide larger portions of RAM to our Process by given one and only large Page**/
	
	
	free(buffer);




	/** Parse CMD **/
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetFlag("help") )
		return usage();


	/** Attach signal handler **/
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	
	/** Define Manually width and height **/ 
	int height = 338;
	int width = 640;
 
	/**

	gstCamera* camera = gstCamera::Create(cmdLine.GetInt("width", gstCamera::DefaultWidth),
								   cmdLine.GetInt("height", gstCamera::DefaultHeight),
								   cmdLine.GetString("camera"));
	
	**/
	gstCamera* camera = gstCamera::Create(width,height,cmdLine.GetString("camera"));

	if( !camera )
	{
		printf("\ndetectnet-camera:  failed to initialize camera device\n");
		return 0;
	}
	
	printf("\ndetectnet-camera:  successfully initialized camera device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	
	/*
	 * create detection network
	 */
	detectNet* net = detectNet::Create(argc, argv);
	
	if( !net )
	{
		printf("detectnet-camera:   failed to load detectNet model\n");
		return 0;
	}

	// parse overlay flags
	const uint32_t overlaySpyFlags = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "box,labels,conf"));
	const uint32_t flagsForStitch = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "none"));
	const uint32_t overlayFlags = overlaySpyFlags;
	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();

	if( !display ) 
		printf("detectnet-camera:  failed to create openGL display\n");


	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("detectnet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("detectnet-camera:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	
	float *leftImgRGBA = NULL;
	float *rightImgRGBA = NULL ;
	float *cyrcleFrameBuffer[6];

	int  frameBound;
	int frameCount=0,numDetections=0,bandwidth = 0,frameBuffCnt=0,counter=0;
	
	int* directionPtr = (int*)malloc(2*sizeof(int));
	bool rotate = false,penalty = false;
	


	while( !signal_recieved )
	{
		// capture RGBA image
		float *imgRGBA = NULL;
		
		if( !camera->CaptureRGBA(&imgRGBA,1000,true) ){
			printf("detectnet-camera:  failed to capture RGBA frame from camera \n");
			break ;
		}
		
		frameBuffCnt++;
		
		/** Save 1 Image per 10 frames (for 60 frames = FPS_PENALTY) overlap **/ 
		if(frameBuffCnt == 10){
			cyrcleFrameBuffer[counter] = imgRGBA;
			counter++;
			frameBuffCnt = 0;

			if(counter == 6)
				counter = 0;
		}


		// If Network Recognized Somethin Start Stitching Process 
		if( numDetections>0 !spy && !predictDir && !threadFlag ){
			
			/** capture two respective frames **/
			camera->CaptureRGBA(&imgRGBA,1000,true);
			leftImgRGBA = imgRGBA;
			float *imgRGBA = NULL;
			camera->CaptureRGBA(&imgRGBA,1000,true);
			rightImgRGBA = imgRGBA;			
			predictDir = true;
				
		}
		else if(predictDir){
								
			std::thread thrDir(predictDirection,leftImgRGBA,rightImgRGBA,height,width);
			thrDir.detach();
			
			spy = true;
			predictDir = false;

		}
		else if(spy && !threadFlag){
			
			if(frameCount == 0){ //Init frameBound and rotate, if its first time in the Club 
				
				if(dir == UP || dir == DOWN){
					frameBound =  (height)/(displ)  - 2;
					rotate = true;
					
					
				}
				else 
					frameBound =  (width)/(displ)  - 2;
					
				

				printf(" FrameBound : %d \n ",frameBound);
				frameBound = frameBound/2;  /** I need A Good Enough Overlap Area **/

				if(frameBound > FPS_PENALTY){
					
					bandwidth = (frameBound - FPS_PENALTY)/4 ;
					penalty = false;
					
				}
				else if(frameBound < FPS_PENALTY){
					
					penalty = true;
					bandwidth = (int)((frameBound)/10); 
					leftImgRGBA  = cyrcleFrameBuffer[counter-1];
					

				}
				
			}
			
			/** Bandwidth == Overlap Area Between Frames **/

			if(frameCount == bandwidth ){
				
				rightImgRGBA = imgRGBA;
				
				if(dir == UP || dir == RIGHT){
					std::thread thrPano(panoramaThread,leftImgRGBA,rightImgRGBA,rotate,height,width,penalty);
					thrPano.detach();
				}
				else{
					std::thread thrPano(panoramaThread,rightImgRGBA,leftImgRGBA,rotate,height,width,penalty); /** Swap Right and Left Image **/
					thrPano.detach();
				}
				
				spy = false ;
				frameCount = 0;
			}
				

			frameCount++;
		}

		detectNet::Detection* detections = NULL;
	
		numDetections = net->Detect(imgRGBA, width, height, &detections, flagsForStitch);
		
		// update display
		
		if( display != NULL )
		{
			// render the image
			display->RenderOnce(imgRGBA, width,height);
			

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
			display->SetTitle(str);

			// check if the user quit
			if( display->IsClosed() )
				signal_recieved = true;
		}
		
		// print out timing info
		//net->PrintProfilerTimes();
	}
	

	printf("detectnet-camera:  shutting down...\n");
	
	SAFE_DELETE(camera);
	SAFE_DELETE(display);
	SAFE_DELETE(net);

	printf("detectnet-camera:  shutdown complete.\n");
	
	

	return 0;
}





void decideDirection(const vector<Point2f>& prevPts,const vector<Point2f>& nextPts,const vector<uchar>&status){
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


    /** Quantize Angles as {0,90,180,-90} [+][+][+][+][+] **/

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
    
    dir = quant;
	
	if(averageDispl < 1)
		displ = 1; /** Give Some Displacement For Pretty Slow Velocity **/
	else
		displ = averageDispl;
    
    
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

void downloadP(const GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

void downloadU(const GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

/** 
	My thread Functions [+][+]
		1 stitching function 
		2 decide direction via optical flow function
**/


void panoramaThread(float *leftRGBA,float* rightRGBA,bool rotate,int rows,int cols,bool penalty){
	
	threadFlag = true;
	printf("[+][+][+][+][+][+][+][+][+]  Stitching Process Just Started [+][+][+][+][+][+][+][+][+] \n");

	uchar3 *rightBGR;
	cudaMalloc((void**)&rightBGR,rows*cols*sizeof(uchar3));
	cudaRGBA32ToBGR8( (float4*)rightRGBA, rightBGR,(size_t)cols,(size_t)rows);

	
	if(penalty){
		uchar3 *leftBGR;
		cudaMalloc((void**)&leftBGR,rows*cols*sizeof(uchar3));
		cudaRGBA32ToBGR8( (float4*)leftRGBA, leftBGR,(size_t)cols,(size_t)rows);
		GpuMat temp(rows,cols,CV_8UC3,(void*)leftBGR);
		temp.copyTo(d_frame1);
	}

	GpuMat gpu_mask1,gpu_mask2;
	
	//GpuMat d_frame1(rows,cols,CV_8UC3,(void*)leftBGR);
	GpuMat d_frame2(rows,cols,CV_8UC3,(void*)rightBGR);
	

	cv::cuda::Stream rot[2];
	if(rotate){
		cv::cuda::rotate( d_frame1, d_frame1, cv::Size(rows,cols ), -90, rows, 0, cv::INTER_LINEAR);
		cv::cuda::rotate( d_frame2, d_frame2, cv::Size(rows,cols ), -90, rows, 0, cv::INTER_LINEAR);
	}
	
	cuda::cvtColor(d_frame1,gpu_mask1,COLOR_BGR2GRAY,1);
    cuda::cvtColor(d_frame2,gpu_mask2,COLOR_BGR2GRAY,1);
	
	Mat frame1(d_frame1);
	Mat frame2(d_frame2);
	
	/**
	imshow("fr1",frame1);
	imshow("fr2",frame2);
	waitKey();
	**/

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
    /** Because of the small size of each operation, gpu use is not worth anymore**/
    /** ---------   **/

    Mat descr1;
    descriptors1GPU.download(descr1);

    // Here Stops GPU and Starts CPU job 

    
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



    //--Use only "good" matches (i.e. whose distance is less than 3 X min_dist and generally linear )
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
    cv::warpPerspective(frame2, result, H, cv::Size(frame1.cols+frame2.cols,frame1.rows));
   
    Mat half(result, cv::Rect(0, 0, frame2.cols, frame2.rows) );
  
    frame1.copyTo(half);

   
    auto stop = high_resolution_clock::now();
    double elapsed_time_ms = duration<double, std::milli>(stop-start).count();
    cout<<"Total Stitching Time millisec : "<<elapsed_time_ms<<endl;
    
    imwrite("pano.jpg",result);

    
 
   
    
	
	threadFlag = false;
	
}


void predictDirection(float* img1,float* img2,int rows,int cols){
	
	

	threadFlag = true ;

	auto t_start = std::chrono::high_resolution_clock::now();

	cv::cuda::Stream stream1,stream2,stream3;

	/** Transform RGBA 32F image to BGR 8U **/
	uchar3* output1 = NULL,*output2 = NULL;
	cudaMalloc((void**)&output1,rows*cols*sizeof(uchar3));
	cudaMalloc((void**)&output2,rows*cols*sizeof(uchar3));

	
	cudaRGBA32ToBGR8( (float4*)img1, output1,(size_t)cols,(size_t)rows);
	cudaRGBA32ToBGR8( (float4*)img2, output2,(size_t)cols,(size_t)rows);
	

	GpuMat d_gray;
	GpuMat dev_frame1(rows,cols,CV_8UC3,(void*)output1);
	GpuMat dev_frame2(rows,cols,CV_8UC3,(void*)output2);
	GpuMat dev_prevPts;
	
	/** Create a copy for left frame of image stitching : d_frame1 **/
	dev_frame2.copyTo(d_frame1);


	cv::cuda::cvtColor(dev_frame1,d_gray,COLOR_BGR2GRAY,1,stream3);
	
	/** Detect Good Corners to Track **/
	Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(d_gray.type(), 500, 0.01, 1);
    detector->detect(d_gray, dev_prevPts,cv::noArray(),stream1);
	
	

    GpuMat dev_nextPts;
    GpuMat dev_status;
	
	

	/** Execute sparse optical Flow calculation **/
    Ptr<cuda::SparsePyrLKOpticalFlow> opticalFlowObj = cuda::SparsePyrLKOpticalFlow::create(Size(21,21),3,30,false);
    opticalFlowObj->calc(dev_frame1,dev_frame2,dev_prevPts,dev_nextPts,dev_status,cv::noArray(),stream2);

	
	/** Download previous, next Points and status for success or not **/
    vector<Point2f> prevPts(dev_prevPts.cols);
    downloadP(dev_prevPts, prevPts);

    vector<Point2f> nextPts(dev_nextPts.cols);
    downloadP(dev_nextPts, nextPts);

    vector<uchar> status(dev_status.cols);
    downloadU(dev_status, status);

	decideDirection(prevPts,nextPts,status);
	
	
	auto t_dir_end = std::chrono::high_resolution_clock::now();
	double elapsed_time_ms = duration<double, std::milli>(t_dir_end-t_start).count();
	printf("Optical FLow total milisec. time : %f \n",elapsed_time_ms);


	threadFlag = false;

}

