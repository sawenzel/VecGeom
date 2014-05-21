//===-- volumes/PlacedParaboloid.h - Instruction class definition -------*- C++ -*-===//
//
//                     GeantV - VecGeom
//
// This file is distributed under the LGPL
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file volumes/PlacedParaboloid.h 
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
/// \brief This file contains the declaration of the PlacedParaboloid class
//===----------------------------------------------------------------------===//

#ifndef VECGEOM_VOLUMES_PLACEDPARABOLOID_H_
#define VECGEOM_VOLUMES_PLACEDPARABOLOID_H_

#include "base/global.h"
#include "volumes/placed_volume.h"
#include "volumes/UnplacedParaboloid.h"

namespace VECGEOM_NAMESPACE {

class PlacedParaboloid : public VPlacedVolume {

public:

  typedef UnplacedParaboloid UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedParaboloid(char const *const label,
                   LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedParaboloid(LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox)
      : PlacedParaboloid("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedParaboloid(LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual ~PlacedParaboloid() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedParaboloid const* GetUnplacedVolume() const {
    return static_cast<UnplacedParaboloid const *>(
        logical_volume()->unplaced_volume());
  }

#ifdef VECGEOM_BENCHMARK
  virtual VPlacedVolume const* ConvertToUnspecialized() const;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const;
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const;
#endif
#endif // VECGEOM_BENCHMARK

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation,
                                   VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation) const;
#endif

};

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDPARABOLOID_H_