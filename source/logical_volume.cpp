#include "volumes/logical_volume.h"
#include "volumes/placed_volume.h"
#include "management/volume_factory.h"

namespace vecgeom {

LogicalVolume::~LogicalVolume() {
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

void LogicalVolume::PrintContent(std::string prefix) const {
  std::cout << *unplaced_volume_ << std::endl;
  prefix += "  ";
  for (Iterator<VPlacedVolume const*> i = daughters().begin();
       i != daughters().end(); ++i) {
    std::cout << prefix << (*i)->matrix() << ": ";
    (*i)->logical_volume()->PrintContent(prefix);
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