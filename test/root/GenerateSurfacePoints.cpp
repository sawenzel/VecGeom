#include "VecGeomTest/RootGeoManager.h"

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "VecGeomTest/Visualizer.h"
#include <string>
#include <cmath>
#include <iostream>

#ifdef VECGEOM_GEANT4
#include "G4ThreeVector.hh"
#include "G4VSolid.hh"
#endif

constexpr size_t N = 1000;

using namespace vecgeom;

int main(int argc, char *argv[])
{

  if (argc < 3) {
    std::cerr << "need to give root geometry file + logical volume name\n";
    std::cerr << "example: " << argv[0] << " cms2015.root CALO\n";
    return 1;
  }

  TGeoManager::Import(argv[1]);
  std::string testvolume(argv[2]);
  int found               = 0;
  TGeoVolume *foundvolume = gGeoManager->FindVolumeFast(argv[2]);
  if (!foundvolume) {
    // now try to find shape with logical volume name given on the command line
    TObjArray *vlist = gGeoManager->GetListOfVolumes();
    for (auto i = 0; i < vlist->GetEntries(); ++i) {
      TGeoVolume *vol = reinterpret_cast<TGeoVolume *>(vlist->At(i));
      std::string fullname(vol->GetName());

      std::size_t founds = fullname.compare(testvolume);
      if (founds == 0) {
        found++;
        foundvolume = vol;
        std::cerr << "found matching volume " << fullname << " of type " << vol->GetShape()->ClassName() << "\n";
      }
    }
    std::cerr << "volume found " << found << " times \n";
    foundvolume->GetShape()->InspectShape();
    std::cerr << "volume capacity " << foundvolume->GetShape()->Capacity() << "\n";
  }

  // now get the VecGeom shape and benchmark it
  if (foundvolume) {
    LogicalVolume *converted     = RootGeoManager::Instance().Convert(foundvolume);
    VPlacedVolume *vecgeomplaced = converted->Place();

    // generate POINTS ON SURFACE FROM VECGEOM
    for (size_t i = 0; i < N; ++i) {
      auto sp = vecgeomplaced->GetUnplacedVolume()->SamplePointOnSurface();
      std::cerr << "SPVG " << sp.x() << " " << sp.y() << " " << sp.z() << "\n";
    }

// generate POINTS ON SURFACE FOR G4
#ifdef VECGEOM_GEANT4
    auto g4volume = vecgeomplaced->ConvertToGeant4();
    for (size_t i = 0; i < N; ++i) {
      auto sp = g4volume->GetPointOnSurface();
      std::cerr << "SPG4 " << sp.x() << " " << sp.y() << " " << sp.z() << "\n";
    }

#endif
    // generate POINTS ON SURFACE FOR ROOT
    double points[3 * N];
    bool success = foundvolume->GetShape()->GetPointsOnSegments(N, points);
    if (success) {
      for (size_t i = 0; i < N; ++i) {
        std::cerr << "SPR " << points[3 * i] << " " << points[3 * i + 1] << " " << points[3 * i + 2] << "\n";
      }
    } else {
      std::cerr << "ROOT cannot generate surface points \n";
    }
  } else {
    std::cerr << " NO SUCH VOLUME [" << testvolume << "] FOUND ... EXITING \n";
    return 1;
  }
}
