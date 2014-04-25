/**
 * @file placed_volume.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include <stdio.h>

#include "management/geo_manager.h"
#include "volumes/placed_volume.h"

namespace VECGEOM_NAMESPACE {

int VPlacedVolume::g_id_count = 0;

#ifndef VECGEOM_NVCC
VPlacedVolume::VPlacedVolume(char const *const label,
                             LogicalVolume const *const logical_volume,
                             Transformation3D const *const transformation,
                             PlacedBox const *const bounding_box)
    : logical_volume_(logical_volume), transformation_(transformation),
      bounding_box_(bounding_box) {
  id_ = g_id_count++;
  GeoManager::Instance().RegisterPlacedVolume(this);
  label_ = new std::string(label);
}
#endif

VPlacedVolume::~VPlacedVolume() {
  delete label_;
}

VECGEOM_CUDA_HEADER_BOTH
void VPlacedVolume::Print(const int indent) const {
  for (int i = 0; i < indent; ++i) printf("  ");
  PrintType();
  printf(" [%i]", id_);
#ifndef VECGEOM_NVCC
  if (label_->size()) {
    printf(" \"%s\"", label_->c_str());
  }
#endif
  printf(": \n");
  for (int i = 0; i <= indent; ++i) printf("  ");
  transformation_->Print();
  printf("\n");
  logical_volume_->Print(indent+1);
}

VECGEOM_CUDA_HEADER_BOTH
void VPlacedVolume::PrintContent(const int indent) const {
  Print(indent);
  if( daughters().size() > 0){
    printf(":");
    for (Iterator<VPlacedVolume const*> vol = daughters().begin();
         vol != daughters().end(); ++vol) {
      printf("\n");
      (*vol)->PrintContent(indent+3);
    }
  }
}

VECGEOM_CUDA_HEADER_HOST
std::ostream& operator<<(std::ostream& os, VPlacedVolume const &vol) {
  os << "(" << (*vol.unplaced_volume()) << ", " << (*vol.transformation())
     << ")";
  return os;
}

} // End global namespace
