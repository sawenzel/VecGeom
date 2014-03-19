#ifndef CPUGPU_TESTCLASS_H_
#define CPUGPU_TESTCLASS_H_

#ifdef CPUGPU_CUDA
  #define CPUGPU_NAMESPACE gpu
#else
  #define CPUGPU_NAMESPACE cpu
#endif

namespace CPUGPU_NAMESPACE {

class TestClass {
  int bar;
public:
  #ifdef CPUGPU_CUDA
  TestClass() { bar = 5; }
  #else
  TestClass() { bar = 42; }
  #endif
  int foo() const;
};

}

#endif // CPUGPU_TESTCLASS_H_