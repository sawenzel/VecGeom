CXX=clang++
VC=clang++
CILK=icc

CXX_OPT=-m64 -std=c++11 -O2
VC_OPT=$(CXX_OPT) -mavx
CILK_OPT=$(CXX_OPT) -xHost

CXX_SRC=VectorTester.cpp
CXX_OBJ=$(addprefix obj/, $(addsuffix .o, $(CXX_SRC)))

VC_SRC=Vc.cpp
VC_OBJ=$(addprefix obj/, $(addsuffix .vc.o, $(VC_SRC)))

CILK_SRC=Cilk.cpp
CILK_OBJ=$(addprefix obj/, $(addsuffix .cilk.o, $(CILK_SRC)))

LIBS=-L ${VCROOT}/lib -lVc -ltbb -lrt -lirc

all: objs

%.cpp.o: ../%.cpp
	$(CXX) -c $(CXX_OPT) $< -o $@

%.cpp.vc.o: ../%.cpp
	$(VC) -c $(VC_OPT) $< -o $@

%.cpp.cilk.o: ../%.cpp
	$(CILK) -c $(CILK_OPT) $< -o $@

objs: $(CXX_OBJ) $(VC_OBJ) $(CILK_OBJ)

Main: objs Main.cpp
	$(CXX) $(CXX_OPT) $(CXX_OBJ) $(VC_OBJ) $(CILK_OBJ) Main.cpp $(LIBS) -o bin/Main

clean:
	@rm obj/*.o