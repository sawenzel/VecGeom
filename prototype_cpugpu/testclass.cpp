#include <iostream>
#include "testclass.h"

namespace CPUGPU_NAMESPACE {

void TestClass::Print() const {
  for (int i = 0; i < 10; ++i) std::cout << vector[i] << " ";
  std::cout << std::endl;
}

}