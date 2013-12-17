#include "GeoManager.h"
#include "TGeoManager.h"

int main(void) {
  TGeoManager *tgeom = new TGeoManager();
  PhysicalVolume const *vol = GeoManager::MakePlacedTube(new TubeParameters<>(.8, 1, 1), new TransformationMatrix(0, 0, 0, 0, 0, 0));
  std::cout << *vol->getMatrix()->GetAsTGeoMatrix()->GetScale() << std::endl;
  return 0;
}