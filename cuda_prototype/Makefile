CXX=g++
NVCC=nvcc
CXX_OPT=-fabi-version=6 -ffast-math -finline-limit=10000000 -mavx -ftree-vectorize -O2
CXX_FLAGS=-m64 -std=c++11
CXX_FLAGS+=${CXX_OPT}
NVCC_FLAGS=-O2 -arch=sm_30
CXX_INCLUDE=-I ./ -I ${VCROOT}/include
CXX_LIBS=-L ${VCROOT}/lib -lVc
NVCC_INCLUDE=-I ./ -I ${VCROOT}/include
NVCC_LIBS=-L ${VCROOT}/lib

all: LauncherCUDA

Launcher: Launcher.cpp Library.h
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) Launcher.cpp -o Launcher $(CXX_LIBS)

LauncherCUDA: Launcher.cu Library.h
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDE) Launcher.cu -o LauncherCUDA $(NVCC_LIBS)