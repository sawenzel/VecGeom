#include <stdio.h>
#include "volumes/logical_volume.h"
#include "volumes/placed_volume.h"
#include "management/volume_factory.h"

namespace vecgeom {

LogicalVolume::~LogicalVolume() {
  if (external_daughters_) return;
  for (Iterator<VPlacedVolume const*> i = daughters().begin();
       i != daughters().end(); ++i) {
    delete *i;
  }
  delete static_cast<Vector<VPlacedVolume const*> *>(daughters_);
}

void LogicalVolume::PlaceDaughter(LogicalVolume const *const volume,
                                  TransformationMatrix const *const matrix) {
  VPlacedVolume *placed =
      VolumeFactory::Instance().CreateSpecializedVolume(volume, matrix);
  static_cast<Vector<VPlacedVolume const*> *>(
    daughters_
  )->push_back(placed);
}

VECGEOM_CUDA_HEADER_BOTH
void LogicalVolume::PrintContent(const int depth) const {
  char const *const tab = "  ";
  for (int i = 0; i < depth; ++i) printf("%s", tab);
  unplaced_volume()->Print();
  printf("\n");
  for (Iterator<VPlacedVolume const*> vol = daughters_->begin();
       vol != daughters_->end(); ++vol) {
    (*vol)->logical_volume()->PrintContent(depth + 1);
  }
}

std::ostream& operator<<(std::ostream& os, LogicalVolume const &vol) {
  os << *vol.unplaced_volume() << std::endl;
  for (Iterator<VPlacedVolume const*> i = vol.daughters().begin();
       i != vol.daughters().end(); ++i) {
    os << "  " << (**i) << std::endl;
  }
  return os;
}

} // End namespace vecgeom