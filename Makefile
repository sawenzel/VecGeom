CXX=g++
CXX_OPT=-O2 -fpermissive -ffast-math  -finline-limit=10000000 -msse4.2 -ftree-vectorize
CXX_FLAGS=-fabi-version=6 -m64 -std=c++11 #`root-config --cflags`
CXX_FLAGS+=${CXX_OPT}
CXX_INCLUDE=-I./ -I./Tests/ -I ${VCROOT}/include
CXX_LIBS=-lGeom -L ${VCROOT}/lib -lVc -L ${TBBROOT}/lib -ltbb -lrt -L ${USOLIDSROOT}/build -lusolids -L ${ROOTSYS}/lib -lGeom 
CXX_SRC=GeoManager_MakeBox.cpp GeoManager_MakeCone.cpp GeoManager_MakeTube.cpp GeoManager_MakePolycone.cpp PhysicalBox.cpp PhysicalVolume.cpp TransformationMatrix.cpp SimpleVecNavigator.cpp
CXX_HDR=PhysicalBox.h PhysicalTube.h PhysicalCone.h
CXX_OBJS=$(addsuffix .cpp.o, $(basename $(CXX_SRC)))

all: CHEP13Benchmark CHEP13BenchmarkSpec CHEP13Benchmark_unspecplacement TestTubeContains

%.cpp.o: %.cpp 
	$(CXX) -c $(CXX_FLAGS) $(CXX_INCLUDE) $< -o $@ 

objs: $(CXX_OBJS) $(CXX_HDR)

CHEP13Benchmark: objs Tests/CHEP13Benchmark.cpp 
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.cpp.o Tests/CHEP13Benchmark.cpp $(CXX_LIBS) -o bin/CHEP13Benchmark

TestTubeContains: objs Tests/TestTubeContains.cpp 
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.cpp.o Tests/TestTubeContains.cpp $(CXX_LIBS) -o bin/TestTubeContains

CHEP13BenchmarkSpec: objs Tests/CHEP13BenchmarkSpecialMatrices.cpp 
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.cpp.o Tests/CHEP13BenchmarkSpecialMatrices.cpp $(CXX_LIBS) -o bin/CHEP13BenchmarkSpec
:
CHEP13Benchmark_unspecplacement: objs Tests/CHEP13Benchmark_unspecializedplacements.cpp 
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.cpp.o Tests/CHEP13Benchmark_unspecializedplacements.cpp $(CXX_LIBS) -o bin/CHEP13Benchmark_unspecplacement

clean:
	@rm *.o