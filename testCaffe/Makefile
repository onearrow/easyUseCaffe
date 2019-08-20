# this is a makefile for caffe  
TARGET := pose 
# caffe directory  
#CAFFE_DIR := /home/hup/project/openpose/caffe
CAFFE_DIR := /home/hup/download/caffe-master
# caffe include  
CAFFE_INCLUDE := $(CAFFE_DIR)/include 
# cuda include
CUDA_INCLUDE :=	/usr/local/cuda-8.0/include
# caffe build   
CAFFE_BUILD := $(CAFFE_DIR)/build 
# caffe build lib  
#CAFFE_BUILD_LIB := /home/hup/project/openpose/caffe/build/lib 
CAFFE_BUILD_LIB := /home/hup/download/caffe-master/build/lib 
# caffe build src
#CAFFE_BUILD_SRC := /home/hup/project/openpose/caffe/build
CAFFE_BUILD_SRC := /home/hup/download/caffe-master/build

# dependency libraries and caffe linking libs  
#LFLAGS := -lpthread\   
LFLAGS += -lcaffe -lglog -lgflags -lopencv_flann -lprotobuf\
		 -lboost_system -lboost_filesystem -lboost_thread -lboost_coroutine -lboost_context\
		 -lboost_regex -lm -lhdf5_hl -lhdf5 -lleveldb -lsnappy -llmdb -lopencv_core\
		 -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lstdc++ -lcblas -latlas
LFLAGS += -Wl,-rpath=$(CAFFE_BUILD_LIB) -L/usr/lib/x86_64-linux-gnu/hdf5/serial -L/usr/lib/x86_64-linux-gnu
LFLAGS += -L/usr/local/opencv330/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui 

# rules  
CFLAGS := -g -I . -I $(CAFFE_INCLUDE) -I $(CUDA_INCLUDE) -I $(CAFFE_DIR)/src -I $(CAFFE_BUILD_SRC)/src/caffe -I/usr/local/opencv330/include/opencv -I/usr/local/opencv330/include -I/usr/include/hdf5/serial -I ./ -Wall
CC := g++ 
SRC += pose.cpp 

$(TARGET):$(SRC)
	$(CC) -o $(TARGET) -O0 -std=c++11 $(SRC) $(CFLAGS) $(LFLAGS) -L$(CAFFE_BUILD_LIB) -Wno-sign-compare

clean : rm -rf $(TARGET)

#--------------------- 
#作者：hanss2 
#来源：CSDN 
#原文：https://blog.csdn.net/hanss2/article/details/78927225 
#版权声明：本文为博主原创文章，转载请附上博文链接！
