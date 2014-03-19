#ifndef CPUGPU_TESTCLASS_H_
#define CPUGPU_TESTCLASS_H_

#ifdef CPUGPU_CUDA
  #define CPUGPU_NAMESPACE gpu
#else
  #define CPUGPU_NAMESPACE cpu
#endif

namespace CPUGPU_NAMESPACE {

class TestClass {
  int vector[10];
public:
  TestClass() { for (int i = 0; i < 10; ++i) vector[i] = i; }
  void DoWork();
  void Print() const;
};

}

#endif // CPUGPU_TESTCLASS_H_