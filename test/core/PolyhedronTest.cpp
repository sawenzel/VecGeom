#include "backend/Backend.h"
#include "volumes/Polyhedron.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/utilities/VolumeUtilities.h"
#ifdef VECGEOM_USOLIDS
#include "UPolyhedra.hh"
#endif
#ifdef VECGEOM_ROOT
#include "TApplication.h"
#include "TPad.h"
#include "TPolyMarker3D.h"
#endif

using namespace vecgeom;

int main() {
  TApplication app("", NULL, NULL);
  Precision zPlanes[] = {1, 3, 5, 8};
  Precision rInner[] = {1, 1, 1, 1};
  Precision rOuter[] = {3, 2, 3, 2};
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

  Precision array[] = {1, 3, 8, 10};
  std::cout << FindSegmentIndex<kScalar>(array, 4., 2.) << FindSegmentIndex<kScalar>(array, 4., 9.) << "\n";

#if defined(VECGEOM_ROOT) && defined(VECGEOM_USOLIDS)
  TGeoShape const* root = polyhedron->ConvertToRoot();
  // VUSolid const* usolid = polyhedron->ConvertToUSolids();
  const_cast<TGeoShape*>(root)->Draw();
  constexpr int nSamples = 256;
  TPolyMarker3D blue(nSamples);
  blue.SetMarkerStyle(5);
  blue.SetMarkerSize(1);
  blue.SetMarkerColor(kBlue);
  TPolyMarker3D red(blue), yellow(blue), green(blue);
  red.SetMarkerColor(kRed);
  yellow.SetMarkerColor(kYellow);
  green.SetMarkerColor(kGreen);
  Vector3D<Precision> bounds(3.25, 3.25, 4.25);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample = volumeUtilities::SamplePoint(bounds)
                               + Vector3D<Precision>(0, 0, 4.5);
    bool vecgeomContains = polyhedron->Contains(sample);
    bool rootContains = root->Contains(&sample[0]);
    if (vecgeomContains && rootContains) {
      green.SetNextPoint(sample[0], sample[1], sample[2]);
    } else if (!vecgeomContains && !rootContains) {
      red.SetNextPoint(sample[0], sample[1], sample[2]);
    } else if (vecgeomContains && !rootContains) {
      blue.SetNextPoint(sample[0], sample[1], sample[2]);
    } else {
      yellow.SetNextPoint(sample[0], sample[1], sample[2]);
    }
  }
  blue.Draw();
  red.Draw();
  green.Draw();
  yellow.Draw();
  app.Run();
  delete root;
  // delete usolid;
#endif
  return 0;
}