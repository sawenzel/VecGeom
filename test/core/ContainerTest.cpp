#include "base/SOA3D.h"
#include "base/AOS3D.h"

#include <cassert>

using namespace vecgeom;

template <typename T, template <typename> class ContainerType>
void SizeTest() {
  ContainerType<T> container(0);
  assert(container.size() == 0);
  assert(container.capacity() == 0);
  container.reserve(8);
  assert(container.size() == 0);
  assert(container.capacity() == 8);
  container.resize(6);
  assert(container.size() == 6);
  assert(container.capacity() == 8);
  container.set(3, Vector3D<T>(1, 2, 3));
  container.set(2, 3, 2, 1);
  assert(container[3] == Vector3D<T>(1, 2, 3));
  assert(container[2] == Vector3D<T>(3, 2, 1));
  assert(container.z(3) == 3);
  assert(container.x(2) == 3);
}

int main() {
  SizeTest<Precision, AOS3D>();
  SizeTest<Precision, SOA3D>();
  return 0;
}