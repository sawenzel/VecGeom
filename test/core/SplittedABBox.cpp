#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeomTest/Visualizer.h"
#include "VecGeom/volumes/Polycone.h"
#include "VecGeom/volumes/Cone.h"
#include "VecGeom/management/ABBoxManager.h"

using namespace vecgeom;

void Visualize(Visualizer *visualizer, VPlacedVolume const *pvol, std::vector<Vector3D<Precision>> lowerc_v,
               std::vector<Vector3D<Precision>> upperc_v);

int main()
{

  // Now trying to generate aligned bounding box for cone

  std::vector<Vector3D<Precision>> lowerc, upperc;

  int numOfSlices = 10;
  // UnplacedBox box(4,6,10);
  auto cone = GeoManager::MakeInstance<UnplacedCone>(0., 3., 0., 8., 30., 0., 2 * kPi);
  Transformation3D tr(0, 0, 0, 0, 30, 45);
  VPlacedVolume const *conePlaced = LogicalVolume("", cone).Place(&tr);
  ABBoxManager<Precision>::Instance().ComputeSplittedABBox(conePlaced, lowerc, upperc, numOfSlices);

  Visualizer visualizer;
  Visualize(&visualizer, conePlaced, lowerc, upperc);
  visualizer.Show();

  return 0;
}

void Visualize(Visualizer *visualizer, VPlacedVolume const *pvol, std::vector<Vector3D<Precision>> lowerc_v,
               std::vector<Vector3D<Precision>> upperc_v)
{

  Transformation3D const *tr = pvol->GetTransformation();

  // Adding placedVolume to visualizer
  visualizer->AddVolume(*pvol, *tr);

  Vector3D<Precision> lowerc(0., 0., 0.), upperc(0., 0., 0.);
  pvol->Extent(lowerc, upperc);
  Vector3D<Precision> delta = (upperc - lowerc) / 2;
  UnplacedBox box(delta);
  VPlacedVolume const *boxPlaced = LogicalVolume("", &box).Place(tr);

  // Adding placedBoundingBox to visualizer
  visualizer->AddVolume(*boxPlaced, *tr);

  // Adding Global Aligned Box of the placed Volume
  Vector3D<Precision> lowerc_ABBox(0., 0., 0.), upperc_ABBox(0., 0., 0.);
  ABBoxManager<Precision>::Instance().ComputeABBox(pvol, &lowerc_ABBox, &upperc_ABBox);
  delta                           = (upperc_ABBox - lowerc_ABBox) / 2;
  Vector3D<Precision> translation = (upperc_ABBox + lowerc_ABBox) / 2;
  Transformation3D trans(translation.x(), translation.y(), translation.z());
  UnplacedBox unplacedGlobalABBox(delta);
  VPlacedVolume const *placedGlobalABBox = LogicalVolume("", &unplacedGlobalABBox).Place(&trans);
  visualizer->AddVolume(*placedGlobalABBox, trans);

  // Adding SplittedABBoxes to visualizer
  for (auto itLower = lowerc_v.begin(), itUpper = upperc_v.begin(); itLower != lowerc_v.end(); itLower++, itUpper++) {
    delta       = (*itUpper - *itLower) / 2;
    translation = (*itUpper + *itLower) / 2;
    trans.SetTranslation(translation);
    trans.SetProperties();
    UnplacedBox unplacedSplittedABBox(delta);
    VPlacedVolume const *placedSplittedABBox = LogicalVolume("", &unplacedSplittedABBox).Place(&trans);
    visualizer->AddVolume(*placedSplittedABBox, trans);
  }
}
