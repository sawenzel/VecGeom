#include <Vc/Vc>
#include "testclass.h"

void cpu::TestClass::DoWork() {
  for (int i = 0; i < 10; i += Vc::int_v::Size) {
    Vc::int_v vec(&vector[i]);
    vec *= 10;
    vec.store(&vector[i]);
  }
}