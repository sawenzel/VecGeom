CXX=g++
CXX_OPT=-O2 -ffast-math  -finline-limit=10000000 -mavx
CXX_FLAGS=-fabi-version=6 -m64 -std=c++11 #`root-config --cflags`
CXX_FLAGS+=${CXX_OPT}
CXX_INCLUDE=-I./ -I./Tests/ -I ${VCROOT}/include
CXX_LIBS=-lGeom -L ${VCROOT}/lib -lVc -L ${TBBROOT}/lib -ltbb -lrt -L ${USOLIDSROOT}/build -lusolids -L ${ROOTSYS}/lib -lGeom 
CXX_SRC=GeoManager_MakeBox.cpp GeoManager_MakeCone.cpp GeoManager_MakeTube.cpp PhysicalBox.cpp PhysicalVolume.cpp TransformationMatrix.cpp SimpleVecNavigator.cpp
CXX_OBJS=$(addsuffix .cpp.o, $(basename $(CXX_SRC)))

all: CHEP13Benchmark

%.cpp.o: %.cpp
	$(CXX) -c $(CXX_FLAGS) $(CXX_INCLUDE) $< -o $@ 

objs: $(CXX_OBJS)

CHEP13Benchmark: objs Tests/CHEP13Benchmark.cpp
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.cpp.o Tests/CHEP13Benchmark.cpp $(CXX_LIBS) -o bin/CHEP13Benchmark

clean:
	@rm *.o