#include "GeoManager.h"

int main(void) {
  PhysicalVolume const *vol = GeoManager::MakePlacedTube(new TubeParameters<>(.8, 1, 1), new TransformationMatrix(0, 0, 0, 0, 0, 0));
  return 0;
}