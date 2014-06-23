#include "volumes/Polyhedron.h"
#include "UPolyhedra.hh"

using namespace vecgeom;

int main() {
  Precision zPlanes[] = {1, 3, 5, 8};
  Precision rInner[] = {1, 1, 1, 1};
  Precision rOuter[] = {3, 2, 3, 2};
  UnplacedPolyhedron polyhedron(3, 0, kTwoPi, 4, zPlanes, rInner, rOuter);
  return 0;
}