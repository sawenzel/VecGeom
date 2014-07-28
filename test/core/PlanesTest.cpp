#include "volumes/Planes.h"

using namespace vecgeom;

int main() {
  Planes planes(6);
  planes.set(0, Vector3D<Precision>(0, 0, 1), Vector3D<Precision>(0, 0, 1));
  planes.set(1, Vector3D<Precision>(0, 0, -1), Vector3D<Precision>(0, 0, -1));
  planes.set(2, Vector3D<Precision>(1, 0, 0), Vector3D<Precision>(1, 0, 0));
  planes.set(3, Vector3D<Precision>(-1, 0, 0), Vector3D<Precision>(-1, 0, 0));
  planes.set(4, Vector3D<Precision>(0, 1, 0), Vector3D<Precision>(0, -1, 0));
  planes.set(5, Vector3D<Precision>(0, -1, 0), Vector3D<Precision>(0, -1, 0));
  Inside_t *inside = new Inside_t[6];
  planes.Inside(Vector3D<Precision>(0, 0, 2), inside);
  for (int i = 0; i < 6; ++i) {
    std::cout << inside[i] << " ";
  }
  return 0;
}