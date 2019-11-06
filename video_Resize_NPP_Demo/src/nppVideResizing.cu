#include <iostream>
#include <cuda_runtime.h>
#include <nppi_geometry_transforms.h>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"


//#include <opencv2/videoio/videoio_c.h>


using namespace std;
using namespace cv;

int main()
{    
    

    //Create an 8 bit single channel image
  
  
  /**
    Mat img = imread("download.jpeg",1);
    Mat out(img.size().height*3/4,img.size().width*3/4,CV_8UC1,Scalar(0));
    **/
    
    /** SET UP ATTRIBUTES **/
    int stepSrc; // Step : number of bytes between succesive rows  
    int stepDst;

    int bytesSrc;
    int bytesDst;
    
    int inWidth;
    int inHeight;
    int outWidth;
    int outHeight;
    
    
    unsigned char *dSrc, *dDst; 
    NppiRect srcRectROI;
    NppiSize sizeSource;
    NppiRect dstRectROI;
    NppiSize sizeDest;
 
    double numOfFrames = 0;
    clock_t stop;
    clock_t start;
    float time ;
    
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    const char* filename = "forest.mp4";
    VideoCapture cap(filename);
    if(!cap.isOpened())return -1;
    
    namedWindow("Video",1);
    while(1)
    {
        start = clock();
        Mat frame;
        cap >> frame ;
        Mat out(frame.size().height*3/4,frame.size().width*3/4,CV_8UC1,Scalar(0));


        inWidth = frame.step[0];
        inHeight = frame.size().height;
        outWidth = out.step[0];
        outHeight = out.size().height;


    


        stepSrc = inWidth; // Step : number of bytes between succesive rows  
        stepDst = outWidth;

        bytesSrc = inWidth * inHeight;
        bytesDst = outWidth * outHeight ;
    


        // Source And Destination 
    
        cudaMalloc<unsigned char>(&dSrc,bytesSrc);
        cudaMalloc<unsigned char>(&dDst,bytesDst);



        //Copy Data From IplImage to Device Pointer
        cudaMemcpyAsync(dSrc,frame.data,bytesSrc,cudaMemcpyHostToDevice,stream1);
        

        //Setting up source-SIZE 
        
        sizeSource.width = inWidth;
        sizeSource.height = inHeight;
    
    

        // Setting up Region Of Interest 
    
        srcRectROI.x = inWidth/4; //2
        srcRectROI.y = inHeight/4 ; //2
        srcRectROI.width = inWidth*3/4;
        srcRectROI.height = inHeight*3/4;

        dstRectROI.x = 0;
        dstRectROI.y = 0;
        dstRectROI.width = outWidth;
        dstRectROI.height = outHeight;
    


        // Setting up destination-SIZE
    
       
        sizeDest.width = outWidth;
        sizeDest.height = outHeight;


        nppiResize_8u_C1R(dSrc,stepSrc,sizeSource,srcRectROI,
                          dDst,stepDst,sizeDest,dstRectROI,NPPI_INTER_LINEAR);

    
        stop = clock();
        time += (float)(stop-start)/CLOCKS_PER_SEC; 
        
        cudaMemcpyAsync(out.data,dDst,bytesDst,cudaMemcpyDeviceToHost,stream2);

        cudaFree(dSrc);
        cudaFree(dDst);

        imshow("output Video ",out);
        imshow("input Video ",frame);
        
        
        numOfFrames++;
        if(waitKey(30)=='c')break;
    }
    
    time = time/numOfFrames ;
    printf("Average response delay for image resizing : %f \n",time);

    return 0;
}
