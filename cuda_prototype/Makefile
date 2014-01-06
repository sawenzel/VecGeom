CXX=g++
CXX_OPT=-fabi-version=6 -ffast-math -finline-limit=10000000 -mavx -ftree-vectorize -O2
CXX_FLAGS=-m64 -std=c++11
CXX_FLAGS+=${CXX_OPT}
CXX_INCLUDE=-I ./ -I ${VCROOT}/include -I$ {TBBROOT}/include
CXX_LIBS=-L ${VCROOT}/lib -lVc -L ${TBBROOT}/lib -ltbb -lrt
CXX_SRC=WrappersVc.cpp
CXX_OBJS=$(addsuffix .cpp.o, $(basename $(CXX_SRC)))

NVCC=nvcc
NVCC_FLAGS=-O2 -arch=sm_30
NVCC_INCLUDE=-I ./ -I ${TBBROOT}/include
NVCC_SRC=WrappersCuda.cu
NVCC_OBJS=$(addsuffix .cu.o, $(basename $(NVCC_SRC)))
NVCC_LIBS=-L ${TBBROOT}/lib -ltbb

all: objs

%.cpp.o: %.cpp
	$(CXX) -c $(CXX_FLAGS) $(CXX_INCLUDE) $< -o obj/$@ 

%.cu.o: %.cu
	$(NVCC) -c $(NVCC_FLAGS) $(NVCC_INCLUDE) $< -o obj/$@ 

objs: $(CXX_OBJS) $(NVCC_OBJS)

MainVc: MainVc.cpp $(CXX_OBJS)
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) obj/*.cpp.o MainVc.cpp -o bin/MainVc $(CXX_LIBS)

MainCuda: MainCuda.cu $(NVCC_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDE) obj/*.cu.o MainCuda.cu -o bin/MainCuda $(NVCC_LIBS)

clean:
	@rm obj/*.o
	@rm bin/*