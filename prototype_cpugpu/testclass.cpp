#include "testclass.h"

namespace CPUGPU_NAMESPACE {

int TestClass::foo() const {
  return 2 * this->bar;
}

}