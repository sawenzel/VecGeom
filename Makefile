CXX=g++
CXX_FLAGS=-fabi-version=6 -m64 -std=c++11 `root-config --cflags`
CXX_INCLUDE=-I./ -I./Tests/
CXX_LIBS=-lGeom -lVc -ltbb -lrt -lusolids `root-config --libs`
CXX_SRC=GeoManager.cpp PhysicalBox.cpp PhysicalTube.cpp PhysicalVolume.cpp ShapeTester.cpp TransformationMatrix.cpp
CXX_OBJS=$(addsuffix .cpp.o, $(basename $(CXX_SRC)))

%.cpp.o: %.cpp
	$(CXX) -c $(CXX_FLAGS) $(CXX_INCLUDE) $< -o $@ $(CXX_LIBS)

objs: $(CXX_OBJS)

ShapeTesterTester: objs Tests/ShapeTesterTester.cpp
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDE) *.cpp.o Tests/ShapeTesterTester.cpp $(CXX_LIBS) -o bin/ShapeTesterTester

clean:
	@rm *.o