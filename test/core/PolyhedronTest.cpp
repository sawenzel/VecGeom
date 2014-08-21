#include "volumes/Polyhedron.h"
#include "volumes/kernel/GenericKernels.h"
#ifdef VECGEOM_USOLIDS
#include "UPolyhedra.hh"
#endif
#include "backend/Backend.h"

using namespace vecgeom;

int main() {
  Precision zPlanes[] = {1, 3, 5, 8};
  Precision rInner[] = {0, 0, 0, 0};
  Precision rOuter[] = {3, 3, 3, 3};
#ifdef VECGEOM_USOLIDS
  UPolyhedra usolids("", 0, 360, 4, 4, zPlanes, rInner, rOuter);
#endif
  UnplacedPolyhedron vecgeom(4, 4, zPlanes, rInner, rOuter);
  LogicalVolume logical(&vecgeom);
  Transformation3D placement;
  VPlacedVolume *polyhedron = logical.Place(&placement);
  Vector3D<Precision> point(0, 0, 2), direction(0, 0, -1);
  assert(polyhedron->DistanceToOut(point, direction) == 1);
  assert(polyhedron->Contains(point));
  assert(polyhedron->Contains(Vector3D<Precision>(2.9, 0, 7)));
  assert(!polyhedron->Contains(Vector3D<Precision>(-2.9, 0, 0)));
  assert(polyhedron->Inside(Vector3D<Precision>(3, 0, 2)) == EInside::kSurface);
  return 0;
}