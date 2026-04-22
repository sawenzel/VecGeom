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

using namespace vecgeom;

std::string VecGeomInsideToString(vecgeom::EnumInside e)
{
  if (e == vecgeom::kInside)
    return "kInside";
  else if (e == vecgeom::kOutside)
    return "kOutside";
  return "kSurface";
}

#ifdef VECGEOM_GEANT4
std::string G4InsideToString(::EInside e)
{
  if (e == ::kInside)
    return "kInside";
  else if (e == ::kOutside)
    return "kOutside";
  return "kSurface";
}
#endif

int main(int argc, char *argv[])
{

  if (argc < 9) {
    std::cerr << "need to give root geometry file + logical volume name + local point + local dir\n";
    std::cerr << "example: " << argv[0] << " cms2015.root CALO 10.0 0.8 -3.5 1 0 0\n";
    return 1;
  }

  TGeoManager::Import(argv[1]);
  std::string testvolume(argv[2]);
  double px   = atof(argv[3]);
  double py   = atof(argv[4]);
  double pz   = atof(argv[5]);
  double dirx = atof(argv[6]);
  double diry = atof(argv[7]);
  double dirz = atof(argv[8]);

  int found               = 0;
  TGeoVolume *foundvolume = NULL;
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

  // now get the VecGeom shape and benchmark it
  if (foundvolume) {
    LogicalVolume *converted     = RootGeoManager::Instance().Convert(foundvolume);
    VPlacedVolume *vecgeomplaced = converted->Place();
    Vector3D<Precision> point(px, py, pz);
    Vector3D<Precision> dir(dirx, diry, dirz);

    SOA3D<Precision> pointcontainer(4);
    pointcontainer.resize(4);
    SOA3D<Precision> dircontainer(4);
    dircontainer.resize(4);
    Precision *steps = new Precision[4];

    if (argc > 9) {
      pointcontainer.set(0, point);
      dircontainer.set(0, dir.x(), dir.y(), dir.z());
      double px2   = atof(argv[9]);
      double py2   = atof(argv[10]);
      double pz2   = atof(argv[11]);
      double dirx2 = atof(argv[12]);
      double diry2 = atof(argv[13]);
      double dirz2 = atof(argv[14]);
      pointcontainer.set(1, px2, py2, pz2);
      dircontainer.set(1, dirx2, diry2, dirz2);
      pointcontainer.set(2, point);
      dircontainer.set(2, dir.x(), dir.y(), dir.z());
      pointcontainer.set(3, px2, py2, pz2);
      dircontainer.set(3, dirx2, diry2, dirz2);

      for (auto i = 0; i < 4; ++i) {
        steps[i] = vecgeom::kInfLength;
      }

    } else {
      for (auto i = 0; i < 4; ++i) {
        pointcontainer.set(i, point);
        dircontainer.set(i, dir.x(), dir.y(), dir.z());
        steps[i] = vecgeom::kInfLength;
      }
    }
    if (!dir.IsNormalized()) {
      std::cerr << "** Attention: Direction is not normalized **\n";
      std::cerr << "** Direction differs from 1 by "
                << std::sqrt(dir.x() * dir.x() + dir.y() * dir.y() + dir.z() * dir.z()) - 1. << "\n";
    }
    double dist;
    std::cout << "VecGeom Capacity " << vecgeomplaced->Capacity() << "\n";
    std::cout << "VecGeom CONTAINS " << vecgeomplaced->Contains(point) << "\n";
    std::cout << "VecGeom INSIDE " << VecGeomInsideToString(vecgeomplaced->Inside(point)) << "\n";
    dist = vecgeomplaced->DistanceToIn(point, dir);
    std::cout << "VecGeom DI " << dist << "\n";
    if (dist < vecgeom::kInfLength) {
      std::cout << "VecGeom INSIDE(p=p+dist*dir) ";
      auto inside = vecgeomplaced->Inside(point + dir * dist);
      std::cout << VecGeomInsideToString(inside) << "\n";
      if (inside == vecgeom::kOutside)
        std::cout << "VecGeom Distance seems to be to big  DI(p=p+dist*dir,-dir) "
                  << vecgeomplaced->DistanceToIn(point + dir * dist, -dir) << "\n";
      if (inside == vecgeom::kInside)
        std::cout << "VecGeom Distance seems to be to small DO(p=p+dist*dir,dir) "
                  << vecgeomplaced->DistanceToOut(point + dir * dist, dir) << "\n";
    }
    dist = vecgeomplaced->DistanceToOut(point, dir);
    std::cout << "VecGeom DO " << dist << "\n";
    std::cout << "VecGeom INSIDE(p=p+do*dir) " << VecGeomInsideToString(vecgeomplaced->Inside(point + dir * dist))
              << "\n";
    Vector3D<Precision> norm;
    auto valid = vecgeomplaced->Normal(point + dist * dir, norm);
    std::cout << "VecGeom Normal(p+do*dir)" << norm << " valid : " << valid << "\n";

    std::cout << "VecGeom SI " << vecgeomplaced->SafetyToIn(point) << "\n";
    std::cout << "VecGeom SO " << vecgeomplaced->SafetyToOut(point) << "\n";

    std::cout << "ROOT Capacity " << foundvolume->GetShape()->Capacity() << "\n";
    std::cout << "ROOT CONTAINS " << foundvolume->GetShape()->Contains(&Vector3D<double>(point)[0]) << "\n";
    std::cout << "ROOT DI "
              << foundvolume->GetShape()->DistFromOutside(&Vector3D<double>(point)[0], &Vector3D<double>(dir)[0])
              << "\n";
    std::cout << "ROOT DO "
              << foundvolume->GetShape()->DistFromInside(&Vector3D<double>(point)[0], &Vector3D<double>(dir)[0])
              << "\n";
    std::cout << "ROOT SI " << foundvolume->GetShape()->Safety(&Vector3D<double>(point)[0], false) << "\n";
    std::cout << "ROOT SO " << foundvolume->GetShape()->Safety(&Vector3D<double>(point)[0], true) << "\n";

    TGeoShape const *rootback = vecgeomplaced->ConvertToRoot();
    if (rootback) {
      std::cout << "ROOTBACKCONV CONTAINS " << rootback->Contains(&Vector3D<double>(point)[0]) << "\n";
      std::cout << "ROOTBACKCONV DI "
                << rootback->DistFromOutside(&Vector3D<double>(point)[0], &Vector3D<double>(dir)[0]) << "\n";
      std::cout << "ROOTBACKCONV DO "
                << rootback->DistFromInside(&Vector3D<double>(point)[0], &Vector3D<double>(dir)[0]) << "\n";
      std::cout << "ROOTBACKCONV SI " << rootback->Safety(&Vector3D<double>(point)[0], false) << "\n";
      std::cout << "ROOTBACKCONV SO " << rootback->Safety(&Vector3D<double>(point)[0], true) << "\n";
    } else {
      std::cerr << "ROOT backconversion failed\n";
    }
#ifdef VECGEOM_GEANT4
    G4ThreeVector g4p(point.x(), point.y(), point.z());
    G4ThreeVector g4d(dir.x(), dir.y(), dir.z());

    G4VSolid const *g4solid = vecgeomplaced->ConvertToGeant4();
    if (g4solid != NULL) {
      std::cout << "G4 CONTAINS " << G4InsideToString(g4solid->Inside(g4p)) << "\n";
      auto distg4 = g4solid->DistanceToIn(g4p, g4d);
      std::cout << "G4 DI " << distg4 << "\n";
      // check status of boundary point
      // G4 has this convention {0=kOutside,1=kSurface,2=kInside};
      std::cout << "G4 DI transported Inside " << G4InsideToString(g4solid->Inside(g4p + distg4 * g4d)) << "\n";
      distg4 = g4solid->DistanceToOut(g4p, g4d);
      std::cout << "G4 DO " << distg4 << "\n";
      std::cout << "G4 DO transported Inside " << G4InsideToString(g4solid->Inside(g4p + distg4 * g4d)) << "\n";
      std::cout << "G4 SI " << g4solid->DistanceToIn(g4p) << "\n";
      std::cout << "G4 SO " << g4solid->DistanceToOut(g4p) << "\n";
    } else {
      std::cerr << "G4 conversion failed\n";
    }
#endif

    double step = 0;
    if (foundvolume->GetShape()->Contains(&Vector3D<double>(point)[0])) {
      step = foundvolume->GetShape()->DistFromInside(&Vector3D<double>(point)[0], &Vector3D<double>(dir)[0]);
    } else {
      step = foundvolume->GetShape()->DistFromOutside(&Vector3D<double>(point)[0], &Vector3D<double>(dir)[0]);
    }
    Visualizer visualizer;
    visualizer.AddVolume(*vecgeomplaced);
    visualizer.AddPoint(point);
    visualizer.AddLine(point, point + step * dir);
    visualizer.Show();

  } else {
    std::cerr << " NO SUCH VOLUME [" << testvolume << "] FOUND ... EXITING \n";
    return 1;
  }
}
