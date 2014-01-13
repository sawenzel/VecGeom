CXX=g++
CXX_OPT=-O2 -fpermissive -ffast-math  -finline-limit=10000000 -mavx -ftree-vectorize
CXX_FLAGS=-fabi-version=6 -m64 -std=c++11 #`root-config --cflags`
CXX_FLAGS+=${CXX_OPT}
#CXX_FLAGS+=-DUSETEMPLATETUBEIF
CXX_FLAGS+=-DSHOWDIFFERENCES
CXX_INCLUDE=-I./ -I./Tests/ -I ${VCROOT}/include
CXX_LIBS=-lGeom -L ${VCROOT}/lib -lVc -L ${TBBROOT}/lib -ltbb -lrt -L ${USOLIDSROOT}/build -lusolids -L ${ROOTSYS}/lib -lGeom 
CXX_SRC=GeoManager_MakeBox.cpp GeoManager_MakeCone.cpp GeoManager_MakeTube.cpp PhysicalBox.cpp PhysicalVolume.cpp TransformationMatrix.cpp SimpleVecNavigator.cpp PhysicalTube.cpp
CXX_OBJS=$(addsuffix .cpp.o, $(basename $(CXX_SRC)))

#all: CHEP13Benchmark CHEP13BenchmarkSpec TestVectorizedPlacedTube
all: CHEP13Benchmark  TestVectorizedPlacedTube

%.cpp.o: %.cpp 
	$(CXX) -c $(CXX_FLAGS) $(CXX_INCLUDE) $< -o $@ 

objs: $(CXX_OBJS)

CHEP13Benchmark: objs Tests/CHEP13Benchmark.cpp
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.cpp.o Tests/CHEP13Benchmark.cpp $(CXX_LIBS) -o bin/CHEP13Benchmark


#CHEP13BenchmarkSpec: objs Tests/CHEP13BenchmarkSpecialMatrices.cpp
#	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.cpp.o Tests/CHEP13BenchmarkSpecialMatrices.cpp $(CXX_LIBS) -o bin/CHEP13BenchmarkSpec


TestVectorizedPlacedTube: objs Tests/TestVectorizedPlacedTube.cpp
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.cpp.o Tests/TestVectorizedPlacedTube.cpp $(CXX_LIBS) -o bin/TestVectorizedPlacedTube



clean:
	@rm *.o