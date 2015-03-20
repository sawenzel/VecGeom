/// \file PlacedBox.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDBOX_H_
#define VECGEOM_VOLUMES_PLACEDBOX_H_

#include "base/Global.h"
#include "backend/Backend.h"
 
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/BoxImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedBox; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedBox )

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedBox : public VPlacedVolume {

public:

#ifndef VECGEOM_NVCC

  PlacedBox(char const *const label,
            LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logicalVolume, transformation, boundingBox) {}

  PlacedBox(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : PlacedBox("", logicalVolume, transformation, boundingBox) {}

#else

  __device__
  PlacedBox(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox,
            const int id)
      : VPlacedVolume(logicalVolume, transformation, boundingBox, id) {}

#endif
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedBox() {}

  // Accessors

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedBox const* GetUnplacedVolume() const {
    return static_cast<UnplacedBox const *>(
        GetLogicalVolume()->unplaced_volume());
  }


  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> const& dimensions() const {
    return GetUnplacedVolume()->dimensions();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision x() const { return GetUnplacedVolume()->x(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision y() const { return GetUnplacedVolume()->y(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision z() const { return GetUnplacedVolume()->z(); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision Capacity() {
      return GetUnplacedVolume()->volume();
  }

  virtual
  void Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  virtual
  bool Normal(Vector3D<Precision> const & point, Vector3D<Precision> & normal ) const
  {
      bool valid;
      BoxImplementation<translation::kIdentity, rotation::kIdentity>::NormalKernel<kScalar>(
              *GetUnplacedVolume(),
              point,
              normal, valid);
      return valid;
  }

#if !defined(VECGEOM_NVCC)
  virtual
  Vector3D<Precision> GetPointOnSurface() const {
    return GetUnplacedVolume()->GetPointOnSurface();
  }
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual double SurfaceArea() {
     return GetUnplacedVolume()->SurfaceArea();
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual std::string GetEntityType() const { return GetUnplacedVolume()->GetEntityType() ;}

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;

  // CUDA specific

  virtual int memory_size() const { return sizeof(*this); }

  // Comparison specific

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

#endif // VECGEOM_VOLUMES_PLACEDBOX_H_
