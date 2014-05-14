/*
 * PlacedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */



#ifndef VECGEOM_VOLUMES_PLACEDCONE_H_
#define VECGEOM_VOLUMES_PLACEDCONE_H_

#include "base/global.h"
#include "volumes/placed_volume.h"
#include "volumes/UnplacedCone.h"

namespace VECGEOM_NAMESPACE {

class PlacedCone : public VPlacedVolume {

public:

  typedef UnplacedCone UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedCone(char const *const label,
                       LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedCone(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : PlacedCone("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedCone(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual ~PlacedCone() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedCone const* GetUnplacedVolume() const {
    return static_cast<UnplacedCone const *>(
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

  Precision GetRmin1() const {return GetUnplacedVolume()->GetRmin1();}
  Precision GetRmax1() const {return GetUnplacedVolume()->GetRmax1();}
  Precision GetRmin2() const {return GetUnplacedVolume()->GetRmin2();}
  Precision GetRmax2() const {return GetUnplacedVolume()->GetRmax2();}
  Precision GetDz() const {return GetUnplacedVolume()->GetDz();}
  Precision GetSPhi() const {return GetUnplacedVolume()->GetSPhi();}
  Precision GetDPhi() const {return GetUnplacedVolume()->GetDPhi();}
  Precision GetInnerSlope() const {return GetUnplacedVolume()->GetInnerSlope();}
  Precision GetOuterSlope() const {return GetUnplacedVolume()->GetOuterSlope();}
  Precision GetInnerOffset() const {return GetUnplacedVolume()->GetInnerOffset();}
  Precision GetOuterOffset() const {return GetUnplacedVolume()->GetOuterOffset();}

  Precision Capacity() const { return GetUnplacedVolume()->Capacity();}
 }; // end class

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDCONE_H_
