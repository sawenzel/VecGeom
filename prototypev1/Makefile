CXX=g++
CXX_OPT=-ggdb -fpermissive -ffast-math  -finline-limit=10000000 -msse4.2 -ftree-vectorize
CXX_FLAGS=-fabi-version=6 -m64 -std=c++11 #`root-config --cflags`
CXX_FLAGS+=${CXX_OPT}

CXX_INCLUDE=-I./ -I./Tests/ -I ${VCROOT}/include -I ${USOLIDSROOT}/include -I ${GEANTVROOTPATH}/include -I ${TBBROOT}/include
CXX_LIBS=-lGeom -L ${VCROOT}/lib -lVc -L ${TBBROOT}/lib/intel64/gcc4.4 -ltbb -lrt -L ${USOLIDSROOT}/.build -lusolids -L ${GEANTVROOTPATH}/lib -lGeom -lRIO -lThread -lCore -lCint -lMathCore -llzma

CXX_SRC=GeoManager_MakeBox.cpp GeoManager_MakeCone.cpp GeoManager_MakeTube.cpp GeoManager_MakePolycone.cpp GeoManager.cpp PhysicalBox.cpp PhysicalVolume.cpp TransformationMatrix.cpp SimpleVecNavigator.cpp

CXX_OBJS=$(addsuffix .o, $(basename $(CXX_SRC)))

all: CHEP13Benchmark CHEP13BenchmarkSpec CHEP13Benchmark_unspecplacement TestPolyconeContains NavigationInBoxDetector TestTransformations transTest

.cpp.o: %.cpp
	@echo Compiling $<
	@$(CXX) -c $(CXX_FLAGS) $(CXX_INCLUDE) $< -o $@ 
	@echo "    done:" $<

objs: $(CXX_OBJS)

CHEP13Benchmark: $(CXX_OBJS) Tests/CHEP13Benchmark.cpp 
	@echo Building $@
	@$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) $(CXX_OBJS) Tests/CHEP13Benchmark.cpp $(CXX_LIBS) -o bin/CHEP13Benchmark

TestPolyconeContains: $(CXX_OBJS) Tests/TestPolyconeContains.cpp 
	@echo Building $@
	@$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.o Tests/TestPolyconeContains.cpp $(CXX_LIBS) -o bin/TestPolyconeContains

TestTransformations: $(CXX_OBJS) Tests/TestTransformations.cpp 
	@echo Building $@
	@$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.o Tests/TestTransformations.cpp $(CXX_LIBS) -o bin/TestTransformations

CHEP13BenchmarkSpec: $(CXX_OBJS) Tests/CHEP13BenchmarkSpecialMatrices.cpp 
	@echo Building $@
	@$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.o Tests/CHEP13BenchmarkSpecialMatrices.cpp $(CXX_LIBS) -o bin/CHEP13BenchmarkSpec

CHEP13Benchmark_unspecplacement: $(CXX_OBJS) Tests/CHEP13Benchmark_unspecializedplacements.cpp 
	@echo Building $@
	@$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.o Tests/CHEP13Benchmark_unspecializedplacements.cpp $(CXX_LIBS) -o bin/CHEP13Benchmark_unspecplacement

NavigationInBoxDetector: $(CXX_OBJS) Tests/NavigationInBoxDetector.cpp 
	@echo Building $@
	@$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.o Tests/NavigationInBoxDetector.cpp $(CXX_LIBS) -o bin/NavigationInBoxDetector

transTest: $(CXX_OBJS) Tests/transTest.cpp
	@echo Building $@
	@$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.o Tests/transTest.cpp $(CXX_LIBS) -o bin/transTest

depend:
	@makedepend -- $(CXX_FLAGS) $(CXX_INCLUDE) -- $(CXX_SRC)

clean:
	@rm *.o
# DO NOT DELETE
