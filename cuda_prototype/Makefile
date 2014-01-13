CXX=g++
CXX_OPT=-fabi-version=6 -ffast-math -finline-limit=10000000 -mavx -ftree-vectorize -O2
CXX_FLAGS=-m64 -std=c++11 -Wall
CXX_FLAGS+=${CXX_OPT}
CXX_INCLUDE=-I ./ -I ${VCROOT}/include -I$ ${TBBROOT}/include
CXX_LIBS=-L ${TBBROOT}/lib -ltbb -lrt
CXX_SRC=WrappersScalar.cpp Shape.cpp
CXX_OBJS=$(addprefix obj/, $(addsuffix .o, $(CXX_SRC)))

VC_SRC=WrappersVc.cpp
VC_OBJS=$(addprefix obj/, $(addsuffix .vc.o, $(VC_SRC)))
VC_OPT=-L ${VCROOT}/lib -lVc

NVCC=nvcc
NVCC_FLAGS=-O2 -arch=sm_30
NVCC_INCLUDE=-I ./ -I ${TBBROOT}/include
NVCC_SRC=WrappersCuda.cu
NVCC_OBJS=$(addprefix obj/, $(addsuffix .o, $(NVCC_SRC)))
NVCC_LIBS=-L ${TBBROOT}/lib -ltbb

all: objs

%.cpp.vc.o: ../%.cpp
	$(CXX) -c $(CXX_FLAGS) $(VC_OPT) $(CXX_INCLUDE) $< -o $@

%.cpp.o: ../%.cpp
	$(CXX) -c $(CXX_FLAGS) $(CXX_INCLUDE) $< -o $@

%.cu.o: ../%.cu
	$(NVCC) -c $(NVCC_FLAGS) $(NVCC_INCLUDE) $< -o $@ 

objs: $(CXX_OBJS) $(VC_OBJS) $(NVCC_OBJS)

MainVc: MainVc.cpp $(CXX_OBJS) $(VC_OBJS)
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) $(CXX_OBJS) $(VC_OBJS) MainVc.cpp -o bin/MainVc $(VC_OPT) $(CXX_LIBS)

MainCuda: MainCuda.cu $(CXX_OBJS) $(NVCC_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDE) $(CXX_OBJS) $(NVCC_OBJS) MainCuda.cu -o bin/MainCuda $(NVCC_LIBS)

clean:
	@rm obj/*.o
	@rm bin/*