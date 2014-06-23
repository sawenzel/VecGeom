#include "volumes/Box.h"
#include "volumes/Parallelepiped.h"
#include "volumes/Tube.h"
#include "UBox.hh"
#include "UTubs.hh"

using namespace vecgeom;

template <class Box_t>
void ConstructBox(double x, double y, double z) {
  Box_t *test = new Box_t("test", x, y, z);
  delete test;
}

template <class Tube_t>
void ConstructTube(double rMin, double rMax, double dZ, double sPhi,
                   double dPhi) {
  Tube_t *test = new Tube_t("test", rMin, rMax, dZ, sPhi, dPhi);
  delete test;
}

int main() {
  ConstructBox<SimpleBox>(5., 5., 5.);
  ConstructBox<UBox>(5., 5., 5.);
  ConstructTube<SimpleTube>(3., 5., 3., 0, kTwoPi);
  ConstructTube<UTubs>(3., 5., 3., 0, kTwoPi);
  return 0;
}