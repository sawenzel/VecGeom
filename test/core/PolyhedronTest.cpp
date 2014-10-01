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

  constexpr int nPlanes = 4;
  Precision zPlanes[nPlanes] = {-2, -1, 1, 2};
  Precision rInner[nPlanes] = {1, 1, 1, 1};
  Precision rOuter[nPlanes] = {3, 3, 3, 3};
  // constexpr int nPlanes = 2;
  // Precision zPlanes[nPlanes] = {-1, 1};
  // Precision rInner[nPlanes] = {0.5, 1};
  // Precision rOuter[nPlanes] = {1, 2};
  // Precision zPlanes[] = {0, 2, 4, 6, 8, 10};
  // Precision rInner[] = {0, 0, 0, 0, 0, 0};
  // Precision rOuter[] = {1, 3, 1, 3, 1, 3};
  UnplacedPolyhedron vecgeom(4, nPlanes, zPlanes, rInner, rOuter);
  LogicalVolume logical(&vecgeom);
  Transformation3D placement;
  VPlacedVolume *polyhedron = logical.Place(&placement);
  
  SOA3D<Precision> p(kVectorSize), d(kVectorSize);
  Precision output[kVectorSize];
  Precision stepMax[kVectorSize];
  for (int i = 0; i < kVectorSize; ++i) {
    // p.set(i, 2.483732, -4.248398, 1.070332);
    // d.set(i, -0.356131, 0.489997, -0.795660)
    p.set(i, -4.691678, -3.838186, -2.737104);
    d.set(i, 0.687631, 0.203269, 0.697026);
    // p.set(i, 3.071539, -2.304781, -0.629474);
    // d.set(i, -0.862077, -0.346298, -0.370002);
    stepMax[i] = vecgeom::kInfinity;
  }
  Precision specializedRes = polyhedron->DistanceToIn(p[0], d[0], stepMax[0]);
  polyhedron->DistanceToIn(p, d, stepMax, output);
  std::cout << specializedRes << " / " << output[0] << "\n";

#ifdef VECGEOM_ROOT
  TApplication app("", NULL, NULL);
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

  Vector3D<Precision> bounds(4, 4, 3);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample = volumeUtilities::SamplePoint(bounds);
    bool vecgeomContains = polyhedron->Contains(sample);
    // bool rootContains = root->Contains(&sample[0]);
#ifdef VECGEOM_USOLIDS
    bool usolidsContains = usolid->Inside(sample) != EInside::kOutside;
#endif
#ifdef VECGEOM_GEANT4
    bool geant4Contains = geant4->Inside(G4ThreeVector(sample[0], sample[1], sample[2])) == ::EInside::kInside;
    if (vecgeomContains && geant4Contains) {
      green.SetNextPoint(sample[0], sample[1], sample[2]);
    } else if (!vecgeomContains && !geant4Contains) {
      red.SetNextPoint(sample[0], sample[1], sample[2]);
    } else if (vecgeomContains && !geant4Contains) {
      blue.SetNextPoint(sample[0], sample[1], sample[2]);
    } else if (!vecgeomContains && geant4Contains) {
      yellow.SetNextPoint(sample[0], sample[1], sample[2]);
    } else {
      assert(0); // All cases should be covered
    }
  }
#endif
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