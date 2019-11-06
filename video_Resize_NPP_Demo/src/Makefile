CC = nvcc

#OPENCV = -I/usr/local/include/opencv4/opencv -I/usr/local/include/opencv4 -L/usr/local/lib -L/usr/local/lib/opencv4/3rdparty -L/home/robin/opencv/build/lib -L-L/usr/local/cuda -L-L/usr/lib -lopencv_gapi -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cudabgsegm -lopencv_cudafeatures2d -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_dnn_objdetect  -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor  -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_cudacodec -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_videostab -lopencv_cudaoptflow -lopencv_optflow -lopencv_cudalegacy -lopencv_video -lopencv_videoio -lopencv_cudawarping -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_imgproc -lopencv_cudaarithm -lopencv_core -lopencv_cudev  -lIlmImf   -lgtk-x11-2.0 -lgdk-x11-2.0 -lpangocairo-1.0 -latk-1.0 -lcairo -lgdk_pixbuf-2.0 -lgio-2.0 -lpangoft2-1.0 -lpango-1.0 -lgobject-2.0 -lglib-2.0 -lfontconfig -lgthread-2.0 -lpng -lz -lfreetype -lharfbuzz -lm -lpthread -lcudart_static -ldl -lrt -lnppc -lnppial -lnppicc -lnppicom -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps -lcublas -lcudnn -lcufft 

OPENCV = `pkg-config --cflags  --libs  opencv4`



INCLUDE = -I /usr/local/cuda/include
LIBRARY  = -L /usr/local/cuda/lib64
FLAGS = -lnppig -lnppidei

SRC := $(wildcard *.cu)
BIN := $(SRC:%.cu=%)


build_target:
	${CC} ${SRC} -o ${BIN} ${INCLUDE} ${LIBRARY} ${FLAGS}  ${OPENCV} -Xlinker --verbose

