#include <iostream>
#include "interface.h"
#include "testclass.h"

namespace CPUGPU_NAMESPACE {
  void entry() {
    TestClass a;
    std::cout << a.foo() << std::endl;
  }
}