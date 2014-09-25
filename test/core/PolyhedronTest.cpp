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
#ifdef VECGEOM_GEANT4
#include "G4Polyhedra.hh"
#endif

using namespace vecgeom;

int main() {

  // double array[] = {0, 1, 4, 6};
  // assert(FindSegmentIndex<kScalar>(array, 4, -3) == -1);
  // assert(FindSegmentIndex<kScalar>(array, 4, 0.5) == 0);
  // assert(FindSegmentIndex<kScalar>(array, 4, 2) == 1);
  // assert(FindSegmentIndex<kScalar>(array, 4, 5) == 2);
  // assert(FindSegmentIndex<kScalar>(array, 4, 7) == 3);

  TApplication app("", NULL, NULL);
  constexpr int nPlanes = 2;
  Precision zPlanes[nPlanes] = {-1, 1};
  Precision rInner[nPlanes] = {0.5, 0.5};
  Precision rOuter[nPlanes] = {1, 1};
  // Precision zPlanes[] = {0, 2, 4, 6, 8, 10};
  // Precision rInner[] = {0, 0, 0, 0, 0, 0};
  // Precision rOuter[] = {1, 3, 1, 3, 1, 3};
  UnplacedPolyhedron vecgeom(4, nPlanes, zPlanes, rInner, rOuter);
  LogicalVolume logical(&vecgeom);
  Transformation3D placement;
  VPlacedVolume *polyhedron = logical.Place(&placement);

#ifdef VECGEOM_ROOT
  TGeoShape const* root = polyhedron->ConvertToRoot();
#ifdef VECGEOM_USOLIDS
  VUSolid const* usolid = polyhedron->ConvertToUSolids();
#endif
#ifdef VECGEOM_GEANT4
  G4VSolid const* geant4 = polyhedron->ConvertToGeant4();
#endif
  const_cast<TGeoShape*>(root)->Draw();
  constexpr int nSamples = 1024;
  TPolyMarker3D blue(nSamples);
  blue.SetMarkerStyle(5);
  blue.SetMarkerSize(1);
  blue.SetMarkerColor(kBlue);
  TPolyMarker3D red(blue), yellow(blue), green(blue),
                magenta(blue);
  red.SetMarkerColor(kRed);
  yellow.SetMarkerColor(kYellow);
  green.SetMarkerColor(kGreen);
  magenta.SetMarkerColor(kMagenta);
  magenta.SetMarkerStyle(21);

  Vector3D<Precision> bounds(1.25, 1.25, 1.25);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample = volumeUtilities::SamplePoint(bounds);
    bool vecgeomContains = polyhedron->Contains(sample);
    bool rootContains = root->Contains(&sample[0]);
#ifdef VECGEOM_USOLIDS
    bool usolidsContains = usolid->Inside(sample) != EInside::kOutside;
#endif
    if (vecgeomContains && rootContains) {
      green.SetNextPoint(sample[0], sample[1], sample[2]);
    } else if (!vecgeomContains && !rootContains) {
      red.SetNextPoint(sample[0], sample[1], sample[2]);
    } else if (vecgeomContains && !rootContains) {
      blue.SetNextPoint(sample[0], sample[1], sample[2]);
    } else if (!vecgeomContains && rootContains) {
      yellow.SetNextPoint(sample[0], sample[1], sample[2]);
    } else {
      assert(0); // All cases should be covered
    }
  }
  Precision cornerLength = 1. / vecgeom::cos(kPi/4);
  magenta.SetNextPoint(cornerLength, 0, 1);
  magenta.SetNextPoint(-cornerLength, 0, 1);
  magenta.SetNextPoint(0, cornerLength, 1);
  magenta.SetNextPoint(0, -cornerLength, 1);
  // for (int i = 0; i < 4; ++i) {
  //   Vector3D<Precision> s =
  //       Vector3D<Precision>::FromCylindrical(1, i*kPi/2 + kPi/4, 1);
  //   magenta.SetNextPoint(s[0], s[1], s[2]);
  // }
  magenta.Draw();
  blue.Draw();
  red.Draw();
  green.Draw();
  yellow.Draw();
  app.Run();
  delete root;
#ifdef VECGEOM_USOLIDS
  delete usolid;
#endif
#ifdef VECGEOM_GEANT4
  delete geant4;
#endif
#endif
  return 0;
}