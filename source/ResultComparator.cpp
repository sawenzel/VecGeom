#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/Global.h"

#include <iostream>
#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#include "TGeoManager.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4VSolid.hh"
#include "G4ThreeVector.hh"
#include "geomdefs.hh"
#endif

#include <sstream>
#include <map>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// reusable utility function to compare distance results against ROOT/Geant4/ etc.
namespace DistanceComparator {

namespace {
#ifdef VECGEOM_ROOT
std::shared_ptr<TGeoShape const> LookupROOT(VPlacedVolume const *vol)
{
  static std::map<VPlacedVolume const *, std::shared_ptr<TGeoShape const>> gROOTshapes;
  if (gROOTshapes.find(vol) == gROOTshapes.end()) {
    gROOTshapes.insert(std::pair<VPlacedVolume const *, std::shared_ptr<TGeoShape const>>(
        vol, std::shared_ptr<TGeoShape const>(vol->ConvertToRoot())));
  }
  return gROOTshapes[vol];
}
#endif

#ifdef VECGEOM_GEANT4
std::shared_ptr<G4VSolid const> LookupG4(VPlacedVolume const *vol)
{
  static std::map<VPlacedVolume const *, std::shared_ptr<G4VSolid const>> gG4shapes;
  if (gG4shapes.find(vol) == gG4shapes.end()) {
    gG4shapes.insert(std::pair<VPlacedVolume const *, std::shared_ptr<G4VSolid const>>(
        vol, std::shared_ptr<G4VSolid const>(vol->ConvertToGeant4())));
  }
  return gG4shapes[vol];
}
#endif
}

void CompareUnplacedContains(VPlacedVolume const *vol, bool vecgeomresult, Vector3D<Precision> const &point)
{

  // convenience counter to enumerate the problems
  // TODO: make this thread safe
  static uint callcounter = 0;
  callcounter++;

  bool mismatch = false;

#ifdef VECGEOM_ROOT
  auto rootshape  = LookupROOT(vol);
  bool rootresult = rootshape->Contains((double *)&point[0]);
  mismatch |= rootresult != vecgeomresult;
#endif

#ifdef VECGEOM_GEANT4
  auto g4shape  = LookupG4(vol);
  auto g4inside = g4shape->Inside(G4ThreeVector(point[0], point[1], point[2]));

  // due to possible ambigouity here use fix values from Geant4 ( 0 == kOutside ; 2 == kInside )
  bool bothoutside = (g4inside == 0) && (!vecgeomresult);
  bool bothinside  = (g4inside == 2) && (vecgeomresult);
  mismatch |= g4inside != 1 && !(bothoutside || bothinside);
#endif

  if (mismatch) {
    std::cerr << "## WARNING (  " << callcounter << " ) ## UnplacedContains VecGeom  " << vecgeomresult;
#ifdef VECGEOM_ROOT
    std::cerr << " ROOT: " << rootresult;
#endif
#ifdef VECGEOM_GEANT4
    std::cerr << " G4: " << g4inside;
#endif
    std::cerr << "\n";
  }
}

void PrintPointInformation(VPlacedVolume const *vol, Vector3D<Precision> const &point)
{
  std::cerr << " INFORMATION FOR POINT " << point << "\n";
  std::cerr << " RELATIVE TO VOLUME " << vol << "\n";
  std::cerr << " Volume Name " << vol->GetLabel() << "\n";
  std::cerr << " Volume Type ";
  vol->PrintType();
  std::cerr << "\n";
  std::cerr << " CONTAINS " << vol->Contains(point) << "\n";
  std::cerr << " INSIDE " << vol->Inside(point) << "\n";
  std::cerr << " SafetyToIn " << vol->SafetyToIn(point) << "\n";
  std::cerr << " SafetyToOut " << vol->SafetyToOut(point) << "\n";
}

void CompareDistanceToIn(VPlacedVolume const *vol, Precision vecgeomresult, Vector3D<Precision> const &point,
                         Vector3D<Precision> const &direction, Precision const stepMax = VECGEOM_NAMESPACE::kInfLength)
{
// this allows to compare distance calculations in each calculation (during a simulation)
// and to report errors early

// other packages usually require transformed points
#if defined(VECGEOM_ROOT) || defined(VECGEOM_GEANT4)
  Precision rootresult{-1};
  Vector3D<Precision> tpoint     = vol->GetTransformation()->Transform(point);
  Vector3D<Precision> tdirection = vol->GetTransformation()->TransformDirection(direction);
#endif

#ifdef VECGEOM_ROOT
  bool mismatch = false;
  auto rootshape = LookupROOT(vol);
  if (rootshape != nullptr) {
    rootresult = rootshape->DistFromOutside((double *)&tpoint[0], (double *)&tdirection[0], 3, stepMax);

    if (Abs(rootresult - vecgeomresult) > kTolerance * rootresult && Abs(rootresult - vecgeomresult) < 1e30) {
      mismatch = true;
      std::cerr << "## WARNING ## DI VecGeom  " << tpoint << " " << tdirection << " " << vecgeomresult;
      std::cerr << " ROOT: " << rootresult << "Delta(" << rootresult - vecgeomresult << ")\n";
    }
  }
#endif

#ifdef VECGEOM_GEANT4
  auto g4shape = LookupG4(vol);
  // g4shape->StreamInfo(std::cerr);
  if (g4shape != nullptr) {
    Precision g4result = g4shape->DistanceToIn(G4ThreeVector(tpoint[0], tpoint[1], tpoint[2]),
                                               G4ThreeVector(tdirection[0], tdirection[1], tdirection[2]));
    if (mismatch || (Abs(g4result - vecgeomresult) > kTolerance * g4result && Abs(rootresult - vecgeomresult) < 1e30)) {
      std::cerr << "## WARNING ## DI VecGeom  " << vecgeomresult;
      std::cerr << " G4: " << g4result << "Delta(" << g4result - vecgeomresult << ")\n";
    }
  }
#endif
}

void CompareDistanceToOut(VPlacedVolume const *vol, Precision vecgeomresult, Vector3D<Precision> const &point,
                          Vector3D<Precision> const &direction, Precision const stepMax = VECGEOM_NAMESPACE::kInfLength)
{
#ifdef VECGEOM_ROOT
  auto rootshape       = LookupROOT(vol);
  Precision rootresult = rootshape->DistFromInside((double *)&point[0], (double *)&direction[0], 3, stepMax);
  if (vecgeomresult < 0) std::cerr << "## WARNING ## DO VecGeom negative (ROOT = " << rootresult << ")\n";
  if (Abs(rootresult - vecgeomresult) > kTolerance * rootresult) {
    std::cerr << "## WARNING ## DO VecGeom  " << vecgeomresult;
    std::cerr << " ROOT: " << rootresult << "\n";
    PrintPointInformation(vol, point);
  }
#endif

#ifdef VECGEOM_GEANT4
  auto g4shape       = LookupG4(vol);
  Precision g4result = g4shape->DistanceToOut(G4ThreeVector(point[0], point[1], point[2]),
                                              G4ThreeVector(direction[0], direction[1], direction[2]), false);
  if (Abs(g4result - vecgeomresult) > kTolerance * g4result) {
    std::cerr << "## WARNING ## DO VecGeom  " << vecgeomresult;
    std::cerr << " G4: " << g4result << "\n";
  }
#endif
}
} // end inner namespace
}
} // end namespace
