#include "base/Vector3D.h"
#include "volumes/Planes.h"

using namespace vecgeom;

int main() {
  Planes planes(6);
  planes.Set(0, Vector3D<Precision>(0, 0, 1), Vector3D<Precision>(0, 0, 1));
  planes.Set(1, Vector3D<Precision>(0, 0, -1), Vector3D<Precision>(0, 0, -1));
  planes.Set(2, Vector3D<Precision>(1, 0, 0), Vector3D<Precision>(1, 0, 0));
  planes.Set(3, Vector3D<Precision>(-1, 0, 0), Vector3D<Precision>(-1, 0, 0));
  planes.Set(4, Vector3D<Precision>(0, 1, 0), Vector3D<Precision>(0, 1, 0));
  planes.Set(5, Vector3D<Precision>(0, -1, 0), Vector3D<Precision>(0, -1, 0));
  std::cout << planes;
  assert(planes.Inside(Vector3D<Precision>(-0.1, 0., 0.6)) == EInside::kInside);
  assert(planes.Inside(Vector3D<Precision>(0.1, 0., -0.6)) == EInside::kInside);
  assert(planes.Inside(Vector3D<Precision>(-1, 0.1, 0.3)) == EInside::kSurface);
  assert(planes.Inside(Vector3D<Precision>(-1, 3, 0)) == EInside::kOutside);
  assert(planes.Inside(Vector3D<Precision>(-1.1, 0.3, 0)) == EInside::kOutside);
  return 0;
}