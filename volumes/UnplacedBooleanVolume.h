#ifndef UNPLACEDBOOLEANVOLUME_H_
#define UNPLACEDBOOLEANVOLUME_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/PlacedVolume.h"

enum BooleanOperation {
    kUnion,
    kIntersection,
    kSubtraction
};

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedBooleanVolume; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedBooleanVolume );

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A class representing a simple UNPLACED boolean volume A-B
 * It takes two template arguments:
 * 1.: the mother (or left) volume A in unplaced form
 * 2.: the subtraction (or right) volume B in placed form;
 * the placement is with respect to the left volume
 *
 *
 *
 * will be a boolean solid where two boxes are subtracted
 * and B is only translated (not rotated) with respect to A
 *
 */
class UnplacedBooleanVolume : public VUnplacedVolume, public AlignedBase {

public:
    VPlacedVolume const* fLeftVolume;
    VPlacedVolume const* fRightVolume;
    BooleanOperation const fOp;

public:
  // need a constructor
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedBooleanVolume(
          BooleanOperation op,
          VPlacedVolume const* left,
          VPlacedVolume const* right ) :
            fLeftVolume(left),
            fRightVolume(right), fOp(op) {}

  virtual int memory_size() const { return sizeof(*this); }

  #ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedBooleanVolume>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
  #endif

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  BooleanOperation GetOp() const {return fOp;}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Capacity() const {
    // TBDONE -- need some sampling
    return 0.;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision SurfaceArea() const {
    // TBDONE -- need some sampling
    return 0.;
  }


  VECGEOM_CUDA_HEADER_BOTH
  void Extent( Vector3D<Precision> &, Vector3D<Precision> &) const {
     // TBDONE
  };


  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetPointOnSurface() const {
    // TBDONE
      return Vector3D<Precision>() ;
  }


  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const {} ;

  virtual void Print(std::ostream &os) const {};

  template <TranslationCode transCodeT, RotationCode rotCodeT>
   VECGEOM_CUDA_HEADER_DEVICE
   static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                                Transformation3D const *const transformation,
 #ifdef VECGEOM_NVCC
                                const int id,
 #endif
                                VPlacedVolume *const placement = NULL);


 private:

   VECGEOM_CUDA_HEADER_DEVICE
   virtual VPlacedVolume* SpecializedVolume(
       LogicalVolume const *const volume,
       Transformation3D const *const transformation,
       const TranslationCode trans_code, const RotationCode rot_code,
 #ifdef VECGEOM_NVCC
       const int id,
 #endif
       VPlacedVolume *const placement = NULL) const;

}; // End class

} // End impl namespace

} // End global namespace



#endif /* UNPLACEDBOOLEANVOLUME_H_ */
