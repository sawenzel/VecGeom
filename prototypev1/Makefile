CXX=g++
CXX_OPT=-O3 -fpermissive -ffast-math  -finline-limit=10000000 -msse4.2 -ftree-vectorize
#CXX_OPT=-O3 -fpermissive -ffast-math  -finline-limit=10000000 -mavx -ftree-vectorize
CXX_FLAGS=-fabi-version=6 -m64 -std=c++11 #`root-config --cflags`
CXX_FLAGS+=${CXX_OPT}

CXX_INCLUDE=-I./ -I./Tests/ -I ${VCROOT}/include -I ${USOLIDSROOT}/include -I ${GEANTVROOTPATH}/include -I ${TBBROOT}/include
CXX_LIBS=-lGeom -L ${VCROOT}/lib -lVc -L ${TBBROOT}/lib/intel64/gcc4.4 -ltbb -lrt -L ${USOLIDSROOT}/.build -lusolids -L ${GEANTVROOTPATH}/lib -lGeom -lRIO -lThread -lCore -lCint -lMathCore -llzma

CXX_SRC=GeoManager_MakeBox.cpp GeoManager_MakeCone.cpp GeoManager_MakeTube.cpp GeoManager_MakePolycone.cpp GeoManager.cpp PhysicalBox.cpp PhysicalVolume.cpp TransformationMatrix.cpp SimpleVecNavigator.cpp Vector3DFast.cpp

#CXX_TESTS=$(wildcard Tests/*.cpp)
CXX_TESTS=Tests/CHEP13Benchmark.cpp Tests/CHEP13BenchmarkSpecialMatrices.cpp Tests/CHEP13Benchmark_unspecializedplacements.cpp Tests/transTest.cpp Tests/NavigationInBoxDetector_Benchmark.cpp Tests/NavigationInBoxDetector_Benchmark_FastTest.cpp Tests/BoxPlacementTest.cpp Tests/TransformOfVectorsTest.cpp Tests/TestPolyconeContains.cpp Tests/DistanceToOutTest.cpp
# Tests/BuildBoxDetector.cpp  Tests/TestBox.cpp

CXX_OBJS=$(addsuffix .o, $(basename $(CXX_SRC)))
CXX_BINS=$(patsubst Tests/%,bin/%,$(basename $(CXX_TESTS)))
CXX_BASE=$(basename $(CXX_TESTS))

#all: CHEP13Benchmark CHEP13BenchmarkSpec CHEP13Benchmark_unspecplacement TestPolyconeContains NavigationInBoxDetector TestTransformations transTest
all: $(CXX_OBJS) $(CXX_BINS)

check:
	@echo CXX_TESTS=$(CXX_TESTS)
	@echo CXX_OBJS=$(CXX_OBJS)
	@echo CXX_BINS=$(CXX_BINS)
	@echo CXX_BASE=$(CXX_BASE)

%.o : %.cpp
	@echo Compiling $<
	@$(CXX) -c $(CXX_FLAGS) $(CXX_INCLUDE) $< -o $@ 
	@echo "    done:" $<

bin/% : Tests/%.cpp $(CXX_OBJS)
	@echo Building $@
	@$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) $(CXX_OBJS) $< $(CXX_LIBS) -o $@

depend:
	@makedepend -- $(CXX_FLAGS) $(CXX_INCLUDE) -- $(CXX_SRC)

clean:
	@rm *.o bin/*
# DO NOT DELETE
