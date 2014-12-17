#include "management/RootGeoManager.h"
#include "volumes/Polyhedron.h"
#include "volumes/utilities/VolumeUtilities.h"

#include "TGeoPgon.h"

#include <cassert>
#include <memory>

using namespace vecgeom;

int main() {

  // Create some vecgeom polyhedron
  constexpr int nPlanes = 4;
  Precision zPlanes[nPlanes] = {-2, -1, 1, 2};
  Precision rInner[nPlanes] = {1, 0.5, 0.5, 1};
  Precision rOuter[nPlanes] = {2, 1, 1, 2};
  SimplePolyhedron vecgeom("Vecgeom", 45, 180, 5, nPlanes, zPlanes, rInner,
                           rOuter);

  // Convert to ROOT using the PlacedVolume interface
  std::unique_ptr<const TGeoShape> rootPolyhedron(vecgeom.ConvertToRoot());

  // Convert it back through the RootGeoManager
  std::unique_ptr<VUnplacedVolume> reconvertedUnplaced(
      RootGeoManager::Instance().Convert(rootPolyhedron.get()));
  LogicalVolume reconvertedLogical(reconvertedUnplaced.get());
  std::unique_ptr<VPlacedVolume> reconverted(reconvertedLogical.Place());

  // Compare
  const Vector3D<Precision> bounds(4, 4, 4);
  for (int i = 0; i < 50000; ++i) {
    Vector3D<Precision> sample = volumeUtilities::SamplePoint(bounds);
    assert(vecgeom.Contains(sample) == reconverted->Contains(sample));
  }

  return 0;
}
