#include "volumes/Polyhedron.h"
#include "volumes/kernel/GenericKernels.h"
#ifdef VECGEOM_USOLIDS
#include "UPolyhedra.hh"
#endif
#ifdef VECGEOM_ROOT
#include "TBrowser.h"
#include "TPad.h"
#include "TGeoPgon.h"
#endif
#include "backend/Backend.h"

using namespace vecgeom;

int main() {
  Precision zPlanes[] = {1, 3, 5, 8};
  Precision rInner[] = {1, 1, 1, 1};
  Precision rOuter[] = {3, 2, 3, 2};
#ifdef VECGEOM_USOLIDS
  UPolyhedra usolids("", 0, 360, 4, 4, zPlanes, rInner, rOuter);
#endif
  UnplacedPolyhedron vecgeom(4, 4, zPlanes, rInner, rOuter);
  LogicalVolume logical(&vecgeom);
  Transformation3D placement;
  VPlacedVolume *polyhedron = logical.Place(&placement);
  Vector3D<Precision> point(0, 1.5, 3), direction(0, -1, 0);
  assert(Abs(polyhedron->DistanceToOut(point, direction) - 0.5) < kTolerance);
  assert(polyhedron->Contains(point));
  assert(polyhedron->Contains(Vector3D<Precision>(2.9, 0, 5)));
  assert(!polyhedron->Contains(Vector3D<Precision>(-2.9, 0, 0)));
  assert(polyhedron->Inside(Vector3D<Precision>(2, 0, 3)) == EInside::kSurface);
#ifdef VECGEOM_ROOT
  TGeoShape const* root = polyhedron->ConvertToRoot();
  double point0[] = {0, 0, 0}, point1[] = {1.1, 1.1, 2}, point2[] = {0, 1.5, 5};
  assert(!root->Contains(point0));
  assert(root->Contains(point1));
  assert(root->Contains(point2));
#endif
  return 0;
}