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

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
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
    
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedParaboloid * GetUnplacedVolumeNonConst() const {
        return static_cast<UnplacedParaboloid *>(const_cast<VUnplacedVolume *>(
            logical_volume()->unplaced_volume()));
    }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRlo() const { return GetUnplacedVolume()->GetRlo(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRhi() const { return GetUnplacedVolume()->GetRhi(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetDz() const { return GetUnplacedVolume()->GetDz(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetA() const { return GetUnplacedVolume()->GetA(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetB() const { return GetUnplacedVolume()->GetB(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetAinv() const { return GetUnplacedVolume()->GetAinv(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetBinv() const { return GetUnplacedVolume()->GetBinv(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetA2() const { return GetUnplacedVolume()->GetA2(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetB2() const { return GetUnplacedVolume()->GetB2(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRlo2() const { return GetUnplacedVolume()->GetRlo2(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRhi2() const { return GetUnplacedVolume()->GetRhi2(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTolIz() const { return GetUnplacedVolume()->GetTolIz(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTolOz() const { return GetUnplacedVolume()->GetTolOz(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetfTolIrlo2() const { return GetUnplacedVolume()->GetTolIrlo2(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTolOrlo2() const { return GetUnplacedVolume()->GetTolOrlo2(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTolIrhi2() const { return GetUnplacedVolume()->GetTolIrhi2(); }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTolOrhi2() const { return GetUnplacedVolume()->GetTolOrhi2(); }
    
    
    VECGEOM_CUDA_HEADER_BOTH
    void Normal(const Precision *point, const Precision *dir, Precision *norm) const { GetUnplacedVolume()->Normal(point, dir, norm) ;}
    
    VECGEOM_CUDA_HEADER_BOTH
    void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const { GetUnplacedVolume()->Extent(aMin, aMax) ;}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision Capacity() const { return GetUnplacedVolume()->Capacity(); }

    
    VECGEOM_CUDA_HEADER_BOTH
    Precision SurfaceArea() const { return GetUnplacedVolume()->SurfaceArea();}

    
    VECGEOM_CUDA_HEADER_BOTH
    Vector3D<Precision>  GetPointOnSurface() const { return GetUnplacedVolume()->GetPointOnSurface() ;}

    
    VECGEOM_CUDA_HEADER_BOTH
    void ComputeBoundingBox() {  GetUnplacedVolumeNonConst()->ComputeBoundingBox() ;}

   
    VECGEOM_CUDA_HEADER_BOTH
    const char* GetEntityType() const { return GetUnplacedVolume()->GetEntityType() ;}

   
    VECGEOM_CUDA_HEADER_BOTH
    void GetParameterList() const { return GetUnplacedVolume()->GetParameterList() ;}


    VECGEOM_CUDA_HEADER_BOTH
    UnplacedParaboloid* Clone() const{ return GetUnplacedVolume()->Clone() ;}

    
    VECGEOM_CUDA_HEADER_BOTH
    void StreamInfo(std::ostream &os) const { return GetUnplacedVolume()->StreamInfo( os) ;}

    
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
