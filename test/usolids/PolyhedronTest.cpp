#include "volumes/Polyhedron.h"
#include "volumes/kernel/GenericKernels.h"
#include "UPolyhedra.hh"

using namespace vecgeom;

void NormalizeAngle();

int main() {
  Precision zPlanes[] = {1, 3, 5, 8};
  Precision rInner[] = {1, 1, 1, 1};
  Precision rOuter[] = {3, 2, 3, 2};
  UnplacedPolyhedron unplaced(3, 0, kTwoPi, 4, zPlanes, rInner, rOuter);
  LogicalVolume logical(&unplaced);
  Transformation3D placement;
  VPlacedVolume *polyhedron = logical.Place(&placement);
  delete polyhedron;
  NormalizeAngle();
  return 0;
}

void NormalizeAngle() {
  Precision angles[] = {-0.1, 3*kPi, 0.1, -kTwoPi};
  Precision result[] = {kTwoPi-0.1, kPi, 0.1, kTwoPi};
  for (int i = 0; i < 4; ++i) {
    Precision normalized = GenericKernels<kScalar>::NormalizeAngle(angles[i]);
    if (normalized != result[i]) {
      printf("Mismatch between angles %.4f and %.4f.\n",
             normalized, result[i]);
      assert(0);
    }
  }
#ifdef VECGEOM_NVC
  for (int i = 0; i < 4; i += Vc::Precision::Size) {
    assert(GenericKernels<kVc>::NormalizeAngle(VcPrecision(&angles[i]))
           == VcPrecision(&result[i]));
  }
#endif
}