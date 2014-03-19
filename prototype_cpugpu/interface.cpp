#include <iostream>
#include "interface.h"
#include "testclass.h"

namespace CPUGPU_NAMESPACE {
  void entry() {
    TestClass a;
    std::cout << "Doing work...\n";
    a.DoWork();
    std::cout << "Printing result...\n";
    a.Print();
  }
}