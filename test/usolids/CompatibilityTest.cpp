#include "volumes/Box.h"
#include "volumes/Parallelepiped.h"
#include "volumes/Tube.h"
#include "UBox.hh"
#include "UTubs.hh"

using namespace vecgeom;

template <class Box_t>
VUSolid* ConstructBox(double x, double y, double z) {
  return new Box_t("test", x, y, z);
}

template <class Tube_t>
VUSolid* ConstructTube(double rMin, double rMax, double dZ, double sPhi,
                     double dPhi) {
  return new Tube_t("test", rMin, rMax, dZ, sPhi, dPhi);
}

void CompareBox(VUSolid const *first, VUSolid const *second) {
  Vector3D<Precision> normal;
  bool convex;
  Vector3D<Precision> insidePoint(5, 5, 5);
  Vector3D<Precision> outsidePoint(-1, 7, 3);
  Vector3D<Precision> direction(0.01, -1.033, 0);
  assert(first->Inside(insidePoint) == second->Inside(insidePoint));
  assert(std::abs(first->DistanceToIn(outsidePoint, direction) -
         second->DistanceToIn(outsidePoint, direction)) < kTolerance);
  assert(std::abs(first->DistanceToOut(insidePoint, direction, normal, convex) -
         second->DistanceToOut(insidePoint, direction, normal, convex))
         < kTolerance);
}

void CompareTube(VUSolid const *first, VUSolid const *second) {
  Vector3D<Precision> normal;
  bool convex;
  Vector3D<Precision> insidePoint(1, -1, 1);
  Vector3D<Precision> outsidePoint(1, 1, -4);
  Vector3D<Precision> direction(0.01, -0.0033, 1);
  assert(first->Inside(insidePoint) == second->Inside(insidePoint));
  assert(std::abs(first->DistanceToIn(outsidePoint, direction) -
         second->DistanceToIn(outsidePoint, direction)) < kTolerance);
  assert(std::abs(first->DistanceToOut(insidePoint, direction, normal, convex) -
         second->DistanceToOut(insidePoint, direction, normal, convex))
         < kTolerance);
}

int main() {
  VUSolid *simpleBox = ConstructBox<SimpleBox>(5., 5., 5.);
  VUSolid *uBox = ConstructBox<UBox>(5., 5., 5.);
  VUSolid *simpleTube = ConstructTube<SimpleTube>(0., 5., 3., 0, kTwoPi);
  VUSolid *uTube = ConstructTube<UTubs>(0., 5., 3., 0, kTwoPi);
  CompareBox(simpleBox, uBox);
  CompareBox(uBox, simpleBox);
  CompareTube(simpleTube, uTube);
  CompareTube(uTube, simpleTube);
  return 0;
}