#include "volumes/Box.h"
//#include "volumes/Parallelepiped.h"
//#include "volumes/UTubs.h"
#include "volumes/Trapezoid.h"
#include "UBox.hh"
#include "UTrap.hh"

using namespace vecgeom;

template <class Box_t>
VUSolid* ConstructBox(double x, double y, double z) {
  return new Box_t("test", x, y, z);
}

template <class Tube_t>
VUSolid* ConstructTrap(double dz, double theta, double phi,
                       double dy1, double dx1, double dx2, double tanAlpha1,
                       double dy2, double dx3, double dx4, double tanAlpha2) {
  return new Tube_t("testTrap", dz, theta, phi, dy1, dx1, dx2, tanAlpha1, dy2, dx3, dx4, tanAlpha2);
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
  Assert(std::abs(first->DistanceToOut(insidePoint, direction, normal, convex) -
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
  Assert(std::abs(first->DistanceToOut(insidePoint, direction, normal, convex) -
         second->DistanceToOut(insidePoint, direction, normal, convex))
         < kTolerance);
}

void CompareTrap(VUSolid const *first, VUSolid const *second) {
  Vector3D<Precision> normal;
  bool convex;
  Vector3D<Precision> insidePoint(1, -1, 0);
  Vector3D<Precision> outsidePoint(10, -10, -5);
  Vector3D<Precision> direction(-0.6, 0.6, 0.5291502622129182);
  assert(first->Inside(insidePoint) == second->Inside(insidePoint));
  assert(std::abs(first->DistanceToIn(outsidePoint, direction) -
         second->DistanceToIn(outsidePoint, direction)) < 10.*kTolerance);
  Assert(std::abs(first->DistanceToOut(insidePoint, direction, normal, convex) -
         second->DistanceToOut(insidePoint, direction, normal, convex))
         < kTolerance);
}

int main() {
  VUSolid *simpleBox = ConstructBox<SimpleBox>(5., 5., 5.);
  VUSolid *uBox = ConstructBox<UBox>(5., 5., 5.);

  VUSolid *simpleTrap = ConstructTrap<SimpleTrapezoid>(5, 0.3, 0.4, 4, 2, 3, 0, 8, 4, 6, 0);
  VUSolid *uTrap = ConstructTrap<UTrap>(5, 0.3, 0.4, 4, 2, 3, 0, 8, 4, 6, 0);

  CompareBox(simpleBox, uBox);
  printf("CompareBox test 1 passed.\n");
  CompareBox(uBox, simpleBox);
  printf("CompareBox test 2 passed.\n");

  CompareTrap(uTrap, simpleTrap);
  printf("CompareTrap test 1 passed.\n");
  CompareTrap(simpleTrap, uTrap);
  printf("CompareTrap test 2 passed.\n");

  return 0;
}
