/// \file PlacedRootVolume.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/PlacedRootVolume.h"

#include "base/AOS3D.h"
#include "base/SOA3D.h"

namespace vecgeom {

PlacedRootVolume::PlacedRootVolume(char const *const label,
                                   TGeoShape const *const rootShape,
                                   LogicalVolume const *const logicalVolume,
                                   Transformation3D const *const transformation)
    : VPlacedVolume(label, logicalVolume, transformation, NULL),
      fRootShape(rootShape) {}

PlacedRootVolume::PlacedRootVolume(TGeoShape const *const rootShape,
                                   LogicalVolume const *const logicalVolume,
                                   Transformation3D const *const transformation)
    : PlacedRootVolume(rootShape->GetName(), rootShape, logicalVolume,
                       transformation) {}

void PlacedRootVolume::PrintType() const {
  printf("PlacedRootVolume");
}

void PlacedRootVolume::Contains(SOA3D<Precision> const &points,
                                bool *const output) const {
  for (int i = 0; i < points.size(); ++i) output[i] = Contains(points[i]);
}

void PlacedRootVolume::Contains(AOS3D<Precision> const &points,
                                bool *const output) const {
  for (int i = 0; i < points.size(); ++i) output[i] = Contains(points[i]);
}

void PlacedRootVolume::Inside(SOA3D<Precision> const &points,
                              Inside_t *const output) const {
  for (int i = 0; i < points.size(); ++i) output[i] = Inside(points[i]);
}

void PlacedRootVolume::Inside(AOS3D<Precision> const &points,
                              Inside_t *const output) const {
  for (int i = 0; i < points.size(); ++i) output[i] = Inside(points[i]);
}

void PlacedRootVolume::DistanceToIn(SOA3D<Precision> const &position,
                                    SOA3D<Precision> const &direction,
                                    Precision const *const step_max,
                                    Precision *const output) const {
  for (int i = 0, i_max = position.size(); i < i_max; ++i) {
    output[i] = DistanceToIn(position[i], direction[i], step_max[i]);
  }
}

void PlacedRootVolume::DistanceToIn(AOS3D<Precision> const &position,
                                    AOS3D<Precision> const &direction,
                                    Precision const *const step_max,
                                    Precision *const output) const {
  for (int i = 0, i_max = position.size(); i < i_max; ++i) {
    output[i] = DistanceToIn(position[i], direction[i], step_max[i]);
  }
}

void PlacedRootVolume::DistanceToOut(SOA3D<Precision> const &position,
                                     SOA3D<Precision> const &direction,
                                     Precision const *const step_max,
                                     Precision *const output) const {
  for (int i = 0, i_max = position.size(); i < i_max; ++i) {
    output[i] = DistanceToOut(position[i], direction[i], step_max[i]);
  }
}

void PlacedRootVolume::DistanceToOut(AOS3D<Precision> const &position,
                                     AOS3D<Precision> const &direction,
                                     Precision const *const step_max,
                                     Precision *const output) const {
  for (int i = 0, i_max = position.size(); i < i_max; ++i) {
    output[i] = DistanceToOut(position[i], direction[i], step_max[i]);
  }
}

void PlacedRootVolume::SafetyToOut(SOA3D<Precision> const &position,
                                   Precision *const safeties) const {
  for (int i = 0, i_max = position.size(); i < i_max; ++i) {
    safeties[i] = SafetyToOut(position[i]);
  }
}

void PlacedRootVolume::SafetyToOut(AOS3D<Precision> const &position,
                                   Precision *const safeties) const  {
  for (int i = 0, i_max = position.size(); i < i_max; ++i) {
    safeties[i] = SafetyToOut(position[i]);
  }
}

void PlacedRootVolume::SafetyToIn(SOA3D<Precision> const &position,
                                  Precision *const safeties) const  {
  for (int i = 0, i_max = position.size(); i < i_max; ++i) {
    safeties[i] = SafetyToIn(position[i]);
  }
}

void PlacedRootVolume::SafetyToIn(AOS3D<Precision> const &position,
                                  Precision *const safeties) const  {
  for (int i = 0, i_max = position.size(); i < i_max; ++i) {
    safeties[i] = SafetyToIn(position[i]);
  }
}

#ifdef VECGEOM_BENCHMARK
VPlacedVolume const* PlacedRootVolume::ConvertToUnspecialized() const {
  assert(0 && "Attempted to perform conversion on unsupported ROOT volume.");
}
#ifdef VECGEOM_ROOT
TGeoShape const* PlacedRootVolume::ConvertToRoot() const {
  assert(0 && "Attempted to perform conversion on unsupported ROOT volume.");
}
#endif
#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedRootVolume::ConvertToUSolids() const {
  assert(0 && "Attempted to perform conversion on unsupported ROOT volume.");
}
#endif
#endif // VECGEOM_BENCHMARK

#ifdef VECGEOM_CUDA_INTERFACE
VPlacedVolume* PlacedRootVolume::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  assert(0 && "Attempted to copy unsupported ROOT volume to GPU.");
}
VPlacedVolume* PlacedRootVolume::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  assert(0 && "Attempted to copy unsupported ROOT volume to GPU.");
}
#endif

} // End namespace vecgeom