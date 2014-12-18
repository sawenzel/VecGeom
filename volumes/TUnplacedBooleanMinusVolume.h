/*
 * TUnplacedBooleanMinusVolume.h
 *
 *  Created on: Aug 13, 2014
 *      Author: swenzel
 */

#ifndef TUNPLACEDBOOLEANMINUSVOLUME_H_
#define TUNPLACEDBOOLEANMINUSVOLUME_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
//#include "volumes/TSpecializedBooleanMinusVolume.h"

namespace VECGEOM_NAMESPACE {

/**
 * A class representing a simple UNPLACED substraction boolean volume A-B
 * It takes two template arguments:
 * 1.: the mother (or left) volume A in unplaced form
 * 2.: the subtraction (or right) volume B in placed form; the placement is with respect to the left volume
 *
 *
 * Example:
 *
 * typedef TUnplacedBooleanMinusVolume<UnplacedBox,
 *           SpecializedBox<translation::kGeneric,rotation::kIdentity> > BoxMinusTranslatedBox_t
 *
 *
 * will be a boolean solid where two boxes are subtracted
 * and B is only translated (not rotated) with respect to A
 *
 */
//template< typename LeftUnplacedVolume_t, typename RightPlacedVolume_t >
typedef VPlacedVolume LeftUnplacedVolume_t;
typedef VPlacedVolume RightPlacedVolume_t;
class TUnplacedBooleanMinusVolume : public VUnplacedVolume, public AlignedBase {

public:
    VPlacedVolume const* fLeftVolume;
    VPlacedVolume const* fRightVolume;
    //LeftUnplacedVolume_t const* fLeftVolume;
    //RightPlacedVolume_t  const* fRightVolume;

public:
  // need a constructor
  TUnplacedBooleanMinusVolume( LeftUnplacedVolume_t const* left,
                               RightPlacedVolume_t const* right ) :
                                   fLeftVolume(left),
                                   fRightVolume(right) {}

  typedef LeftUnplacedVolume_t LeftType;
  typedef RightPlacedVolume_t  RightType;

  virtual int memory_size() const { return sizeof(*this); }

  #ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const;
  #endif


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

#ifndef VECGEOM_NVCC
  template <typename LeftUnplacedVolume_t, typename RightPlacedVolume_t, TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL) {
//      if (placement) {
//          new(placement) TSpecializedBooleanMinusVolume<LeftUnplacedVolume_t, RightPlacedVolume_t,
//                  trans_code, rot_code>(logical_volume,
//                                                              transformation);
//          return placement;
//        }
  //      return new TSpecializedBooleanMinusVolume<LeftUnplacedVolume_t, RightPlacedVolume_t, trans_code, rot_code>(logical_volume,
   //                                                     transformation);
 }

  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) {

     // return VolumeFactory::CreateByTransformation<TUnplacedBooleanMinusVolume<LeftUnplacedVolume_t, RightPlacedVolume_t> >(
     //           volume, transformation, trans_code, rot_code, placement
     //         );
  }

#else // for CUDA
  template <TranslationCode trans_code, RotationCode rot_code>
  __device__
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
                               const int id,
                               VPlacedVolume *const placement = NULL);

  __device__
  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL);
#endif

private:

#ifndef VECGEOM_NVCC
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
                                   placement);
  }
#else
  __device__
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
                                   id, placement);
  }
#endif

}; // End class

} // End global namespace



#endif /* TUNPLACEDBOOLEANMINUSVOLUME_H_ */
