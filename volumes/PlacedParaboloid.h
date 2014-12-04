//===-- volumes/PlacedParaboloid.h - Instruction class definition -------*- C++ -*-===//
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
#ifdef USOLIDS
class VUSOLID;
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedParaboloid; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedParaboloid );

inline namespace VECGEOM_IMPL_NAMESPACE {

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
    VECGEOM_CUDA_HEADER_BOTH
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
    
    virtual
    bool Normal(Vector3D<Precision> const &, Vector3D<double> &normal) const {
      Assert(0, "Normal with point only not implemented for Paraboloid.\n");
      return false;
    }
    
    void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const { GetUnplacedVolume()->Extent(aMin, aMax) ;}
    
    Precision Capacity() const { return GetUnplacedVolume()->Capacity(); }

    Precision SurfaceArea() const { return GetUnplacedVolume()->SurfaceArea();}

#if !defined(VECGEOM_NVCC) && defined(VECGEOM_USOLIDS)
    Vector3D<Precision> GetPointOnSurface() const {
      return GetUnplacedVolume()->GetPointOnSurface();
    }
#endif

    void ComputeBoundingBox() {  GetUnplacedVolumeNonConst()->ComputeBoundingBox() ;}

    virtual
    std::string GetEntityType() const { return GetUnplacedVolume()->GetEntityType() ;}

    void GetParameterList() const { return GetUnplacedVolume()->GetParameterList() ;}


#ifdef USOLIDS
    VECGEOM_CUDA_HEADER_BOTH
    virtual
    VUSolid* Clone() const{ 
      return NULL;
      //return GetUnplacedVolume()->Clone() ;
}
#endif
    
  //    VECGEOM_CUDA_HEADER_BOTH
  //    virtual
  //    void StreamInfo(std::ostream &os) const { return GetUnplacedVolume()->StreamInfo( os) ;}


#ifndef VECGEOM_NVCC
  virtual VPlacedVolume const* ConvertToUnspecialized() const;
#ifdef VECGEOM_ROOT
    virtual TGeoShape const* ConvertToRoot() const;
#endif
#ifdef VECGEOM_USOLIDS
    virtual ::VUSolid const* ConvertToUSolids() const;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const* ConvertToGeant4() const;
#endif
#endif // VECGEOM_NVCC


};

} } // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDPARABOLOID_H_
