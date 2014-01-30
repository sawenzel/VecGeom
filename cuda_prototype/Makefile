CXX=g++
GNU_OPT=-fabi-version=6 -ffast-math -finline-limit=10000000 -mavx -ftree-vectorize -O2
CLANG_OPT=-ffast-math -mavx -ftree-vectorize -O2
CXX_OPT=$(GNU_OPT)
CXX_FLAGS=-m64 -std=c++11 -Wall -lprofiler
CXX_FLAGS+=${CXX_OPT}
CXX_INCLUDE=-I ./ -I ${VCROOT}/include -I$ ${TBBROOT}/include
GNU_LIBS=
CLANG_LIBS=-stdlib=libc++
CXX_LIBS=-L ${TBBROOT}/lib -ltbb -lrt $(GNU_LIBS) -lirc
CXX_HEADERS=LibraryGeneric.h Shape.h Box.h KernelBox.h Tube.h KernelTube.h
CXX_SRC=Box.cpp Tube.cpp Shape.cpp
CXX_OBJS=$(addprefix obj/, $(addsuffix .o, $(CXX_SRC)))

SCALAR_SRC=ImplementationsScalar.cpp
SCALAR_OBJS=$(addprefix obj/, $(addsuffix .scalar.o, $(SCALAR_SRC)))

VC_SRC=ImplementationsVc.cpp
VC_OBJS=$(addprefix obj/, $(addsuffix .vc.o, $(VC_SRC)))
VC_OPT=-L ${VCROOT}/lib -lVc

CILK=icc
CILK_SRC=ImplementationsCilk.cpp
CILK_OBJS=$(addprefix obj/, $(addsuffix .cilk.o, $(CILK_SRC)))
CILK_OPT=-m64 -std=c++11 -Wall -xHost -finline-limit=10000000 -ftree-vectorize -O2
CILK_INCLUDE=$(CXX_INCLUDE)

NVCC=nvcc
NVCC_FLAGS=-O2 -arch=sm_30 
NVCC_INCLUDE=-I ./ -I ${TBBROOT}/include
NVCC_SRC=ImplementationsCuda.cu
NVCC_OBJS=$(addprefix obj/, $(addsuffix .o, $(NVCC_SRC)))
NVCC_LIBS=-L ${TBBROOT}/lib -ltbb # -profiler

all: objs

%.cpp.vc.o: ../%.cpp LibraryVc.h
	$(CXX) -c $(CXX_FLAGS) $(VC_OPT) $(CXX_INCLUDE) $< -o $@

%.cpp.scalar.o: ../%.cpp LibraryGeneric.h
	$(CXX) -c $(CXX_FLAGS) $(CXX_INCLUDE) $< -o $@

%.cpp.cilk.o: ../%.cpp LibraryCilk.h
	$(CILK) -c $(CILK_OPT) $(CILK_INCLUDE) $< -o $@

%.cpp.o: ../%.cpp LibraryGeneric.h
	$(CXX) -c $(CXX_FLAGS) $(CXX_INCLUDE) $< -o $@

%.cu.o: ../%.cu LibraryCuda.cuh
	$(NVCC) -c $(NVCC_FLAGS) $(NVCC_INCLUDE) $< -o $@ 

objs: $(CXX_OBJS) $(VC_OBJS) $(NVCC_OBJS) $(CILK_OBJS)

MainScalar: LibraryGeneric.h $(CXX_HEADERS) MainScalar.cpp $(SCALAR_OBJS) $(CXX_OBJS)
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) $(CXX_OBJS) $(SCALAR_OBJS) MainScalar.cpp -o bin/MainScalar $(CXX_LIBS)

MainVc: LibraryVc.h $(CXX_HEADERS) MainVc.cpp $(CXX_OBJS) $(VC_OBJS)
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) $(CXX_OBJS) $(VC_OBJS) MainVc.cpp -o bin/MainVc $(VC_OPT) $(CXX_LIBS)

MainCuda: LibraryCuda.cuh $(CXX_HEADERS) MainCuda.cu $(CXX_OBJS) $(NVCC_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDE) $(CXX_OBJS) $(NVCC_OBJS) MainCuda.cu -o bin/MainCuda $(NVCC_LIBS)

MainCilk: LibraryCilk.h $(CXX_HEADERS) MainCilk.cpp $(CXX_OBJS) $(CILK_OBJS)
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) $(CXX_OBJS) $(CILK_OBJS) MainCilk.cpp -o bin/MainCilk $(CXX_LIBS)

TestCase: $(CXX_HEADERS) TestCase.cpp $(CXX_OBJS) $(VC_OBJS)
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) $(CXX_OBJS) $(VC_OBJS) TestCase.cpp -o bin/TestCase $(VC_OPT) $(CXX_LIBS)

CompareOutput: CompareOutput.cpp
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) CompareOutput.cpp -o bin/CompareOutput $(CXX_LIBS)

clean:
	@rm obj/*.o
	@rm bin/*