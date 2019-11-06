# jetsonDemos
Some Demos for the embedded platform Jetson Nano 
Let me clear, that these codes in average are inspired from jetson inference tutorial and opencv threads in stack-overflow , opencv forum and github repositories.If your interests lie in results you can check the output of Demos.
Code Demos are basically implemented in : 
  1. Jetson inference and Jetson utils (Tracking objects)
  2. NPP CUDA API (Fast video Resizing)
  3. OpenCV::cuda class (Accelerated image stitching)
Requirements and Tips  : 
 ---------------------------------------------------------------------------------------------------------------------------------------
 Jetson Inference : https://github.com/dusty-nv/jetson-inference (Follow tutorial for installation and in-depth understanding)
 Tip1:  If you supply jetson via micro USB , make sure that five Watt Mode is enabled.
 ---------------------------------------------------------------------------------------------------------------------------------------
 NPP : pre-installed in jetson platforms, link libraries very carefully(either static or dynamic).
 ---------------------------------------------------------------------------------------------------------------------------------------
 OpenCV : I higly recommend OpenCV 4.0 because methods and functions are optimized for GPU support, instead of pre-installed OpenCV 3.0.
 I recommend the following building tutorial : https://pysource.com/2019/08/26/install-opencv-4-1-on-nvidia-jetson-nano/
 Tip1:  Be carefull with cmake -D flags  
    if you are C++ programmer disable python . If you want to produce .pc file for pkg-config linking and SURF class add the respective     flags. 
 Tip2:  Be carefull with make -j , if your jetson operates in five Watt Mode make -j2  is recommended, else make -j4 should work fine.
-----------------------------------------------------------------------------------------------------------------------------------------
