GXX=/usr/bin/g++
GCC=/usr/bin/gcc
CUDA_DIR=/usr/local/cuda
OPENBLAS_DIR=/opt/OpenBLAS
PROTOBUF_DIR=/data1/medivhgu/my_install/protobuf-2.6.1
CAFFE_DIR=/data1/medivhgu/deep_tools/caffe

CXXFLAG=-fopenmp -march=core2 -O2 -fomit-frame-pointer -pipe
CFLAG=$(CXXFLAG)
DIR_INC=./include
DIR_SRC=./src
DIR_OBJ=./obj
DIR_BIN=./bin
DIR_LIB=./lib
LDFLAG:=#-Wl
INCLUDE=-I$(DIR_INC) -I$(CUDA_DIR)/include -I$(OPENBLAS_DIR)/include -I$(PROTOBUF_DIR)/include -I$(CAFFE_DIR)/include
LIBS=-L$(DIR_LIB) -L$(OPENBLAS_DIR)/lib -L$(PROTOBUF_DIR)/lib -L$(CAFFE_DIR)/build/lib -Wl,--whole-archive -lcaffe -Wl,--no-whole-archive -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_objdetect -lglog -lprotobuf -lhdf5 -lhdf5_hl -llmdb -lleveldb -lopenblas -lboost_thread

SOCROP=lib/libface_gender_classifier.so # DO NOT CHANGE THR LIB NAME
COBJS=$(DIR_OBJ)/face_gender_classifier.o
DEF= -DLINUX #-DCPU_ONLY -DR_DEBUG
SRC=$(wildcard ${DIR_SRC}/*.cpp)
MAINOBJ=$(DIR_OBJ)/main.o
MAIN=$(DIR_BIN)/main.bin

all: $(MAIN)

$(MAIN):$(SOCROP) $(MAINOBJ)
	$(GXX) $(CXXFLAG) $(LDFLAG) -o $@ $(SOCROP) $(MAINOBJ) $(LIBS)

$(SOCROP):$(COBJS)
	$(GXX) -shared -fPIC $(CXXFLAG) $(LDFLAG) -o $@ $(COBJS) $(LIBS)

$(DIR_OBJ)/%.o:$(DIR_SRC)/%.cpp
	$(GXX) -fPIC -o $@ -c $< $(CXXFLAG) $(INCLUDE) $(DEF)

clean:
	rm -rf $(DIR_OBJ)/*.o  $(SOCROP) $(MAIN)

