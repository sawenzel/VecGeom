/**
 * @file placed_box.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_VOLUMES_PLACEDBOX_H_
#define VECGEOM_VOLUMES_PLACEDBOX_H_

#include "base/global.h"
#include "backend/backend.h"
 
#include "volumes/placed_volume.h"
#include "volumes/unplaced_box.h"
#include "volumes/kernel/box_kernel.h"

namespace VECGEOM_NAMESPACE {

class PlacedBox : public VPlacedVolume {

public:

  PlacedBox(char const *const label,
            LogicalVolume const *const logical_volume,
            TransformationMatrix const *const matrix)
      : VPlacedVolume(label, logical_volume, matrix, this) {}

#ifdef VECGEOM_STD_CXX11
  PlacedBox(LogicalVolume const *const logical_volume,
            TransformationMatrix const *const matrix)
      : PlacedBox("", logical_volume, matrix) {}
#endif

#ifdef VECGEOM_NVCC
  VECGEOM_CUDA_HEADER_DEVICE
  PlacedBox(LogicalVolume const *const logical_volume,
            TransformationMatrix const *const matrix,
            const int id)
      : VPlacedVolume(logical_volume, matrix, this, id) {}
#endif

  virtual ~PlacedBox() {}

  // Accessors

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> const& dimensions() const {
    return unplaced_box()->dimensions();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision x() const { return unplaced_box()->x(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision y() const { return unplaced_box()->y(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision z() const { return unplaced_box()->z(); }

protected:

  /**
   * Retrieves the unplaced volume pointer from the logical volume and casts it
   * to an unplaced box.
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedBox const* unplaced_box() const;


public:

  // Navigation methods

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool Inside(Vector3D<Precision> const &point) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool Inside(Vector3D<Precision> const &point,
        Vector3D<Precision> & localpoint) const;

  virtual void Inside(SOA3D<Precision> const &points,
                      bool *const output) const;

  virtual void Inside(AOS3D<Precision> const &points,
                      bool *const output) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool UnplacedInside(Vector3D<Precision> const &) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Precision DistanceToIn(Vector3D<Precision> const &position,
                                 Vector3D<Precision> const &direction,
                                 const Precision step_max) const;

  virtual void DistanceToIn(SOA3D<Precision> const &position,
                            SOA3D<Precision> const &direction,
                            Precision const *const step_max,
                            Precision *const output) const;

  virtual void DistanceToIn(AOS3D<Precision> const &position,
                            AOS3D<Precision> const &direction,
                            Precision const *const step_max,
                            Precision *const output) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Precision DistanceToOut(Vector3D<Precision> const &position,
                                  Vector3D<Precision> const &direction,
                                  Precision const step_max) const;


  virtual void DistanceToOut(SOA3D<Precision> const &position,
                             SOA3D<Precision> const &direction,
                             Precision const *const step_max,
                             Precision *const output) const;


  virtual void DistanceToOut(AOS3D<Precision> const &position,
                             AOS3D<Precision> const &direction,
                             Precision const *const step_max,
                             Precision *const output) const;

  // for safeties
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Precision SafetyToOut( Vector3D<Precision> const &position ) const;

  virtual void SafetyToOut( SOA3D<Precision> const &position,
        Precision *const safeties ) const;
  virtual void SafetyToOut( AOS3D<Precision> const &position,
        Precision *const safeties ) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Precision SafetyToIn( Vector3D<Precision> const &position ) const;
  virtual void SafetyToIn( SOA3D<Precision> const &position,
        Precision *const safeties ) const;
  virtual void SafetyToIn( AOS3D<Precision> const &position,
          Precision *const safeties ) const;


  // CUDA specific

  virtual int memory_size() const { return sizeof(*this); }

  #ifdef VECGEOM_CUDA_INTERFACE
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      TransformationMatrix const *const matrix,
      VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      TransformationMatrix const *const matrix) const;
  #endif

  // Comparison specific

  virtual VPlacedVolume const* ConvertToUnspecialized() const;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const;
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const;
#endif


  // Templates to interact with common C-like kernels
  template <TranslationCode trans_code, RotationCode rot_code,
              typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::bool_v InsideDispatch(
      Vector3D<typename Backend::precision_v> const &point) const;

  template <TranslationCode trans_code, RotationCode rot_code,
                 typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v DistanceToInDispatch(
      Vector3D<typename Backend::precision_v> const &position,
      Vector3D<typename Backend::precision_v> const &direction,
      const typename Backend::precision_v step_max) const;


  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v DistanceToOutDispatch(
      Vector3D<typename Backend::precision_v> const &position,
        Vector3D<typename Backend::precision_v> const &direction,
        const typename Backend::precision_v step_max) const;


  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v SafetyToOutDispatch(
             Vector3D<typename Backend::precision_v> const &position
          ) const;

  template <TranslationCode trans_code, RotationCode rot_code,
               typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v SafetyToInDispatch(
             Vector3D<typename Backend::precision_v> const &position
         ) const;

};


template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::bool_v PlacedBox::InsideDispatch(
    Vector3D<typename Backend::precision_v> const &point) const {

  typename Backend::bool_v output;
  BoxInside<trans_code, rot_code, Backend>(
    unplaced_box()->dimensions(),
    *this->matrix(),
    point,
    &output
  );
  return output;
}

template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v PlacedBox::DistanceToInDispatch(
    Vector3D<typename Backend::precision_v> const &position,
    Vector3D<typename Backend::precision_v> const &direction,
    const typename Backend::precision_v step_max) const {

  typename Backend::precision_v output;
  BoxDistanceToIn<trans_code, rot_code, Backend>(
    unplaced_box()->dimensions(),
    *this->matrix(),
    position,
    direction,
    step_max,
    &output
  );
  return output;
}

template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v PlacedBox::DistanceToOutDispatch(
    Vector3D<typename Backend::precision_v> const &position,
    Vector3D<typename Backend::precision_v> const &direction,
    const typename Backend::precision_v step_max) const
{
   typename Backend::precision_v output;
   BoxDistanceToOut<Backend>(
         unplaced_box()->dimensions(),
         position,
         direction,
         step_max,
         output
   );
   return output;
}

template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v PlacedBox::SafetyToOutDispatch(
          Vector3D<typename Backend::precision_v> const &position
         ) const {
   typename Backend::precision_v safety;
   BoxSafetyToOut<Backend>(unplaced_box()->dimensions(), position, safety);
   return safety;
}

template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v PlacedBox::SafetyToInDispatch(
      Vector3D<typename Backend::precision_v> const &position
) const {
   typename Backend::precision_v safety;
   BoxSafetyToIn<trans_code, rot_code, Backend>(unplaced_box()->dimensions(), *matrix(), position, safety);
   return safety;
}


VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
UnplacedBox const* PlacedBox::unplaced_box() const {
  return static_cast<UnplacedBox const*>(this->unplaced_volume());
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool PlacedBox::Inside(Vector3D<Precision> const &point) const {
  return PlacedBox::InsideDispatch<1, 0, kScalar>(point);
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool PlacedBox::UnplacedInside(Vector3D<Precision> const &localpoint) const {
  // no translation and no rotation should look like this:
   return PlacedBox::InsideDispatch<0, rotation::kIdentity, kScalar>(localpoint);
}


VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool PlacedBox::Inside(Vector3D<Precision> const &point, Vector3D<Precision> & localpoint) const
{
   typename kScalar::bool_v output;
   BoxInside<1, 0, kScalar>(
         unplaced_box()->dimensions(),
         *this->matrix_,
         point,
         localpoint,
         &output );
   return output;
}


VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision PlacedBox::DistanceToIn(Vector3D<Precision> const &position,
                                  Vector3D<Precision> const &direction,
                                  const Precision step_max) const {
  return PlacedBox::DistanceToInDispatch<1, 0, kScalar>(position, direction,
                                                        step_max);
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision PlacedBox::DistanceToOut(Vector3D<Precision> const &position,
                                   Vector3D<Precision> const &direction,
                                   Precision const step_max) const {

   // for the moment we dispatch to the common kernel
   return PlacedBox::DistanceToOutDispatch<kScalar>(position,direction,step_max);

   // this kernel is an optimized "kInternalVectorization" kernel
   // we have how we can dispatch to this kernel later

   /*
   // const Vector3D<Precision> local = matrix()->Transform(position);

  Vector3D<Precision> const &dim = unplaced_box()->dimensions();

  const Vector3D<Precision> safety_plus  = dim + position;
  const Vector3D<Precision> safety_minus = dim - position;

  // TODO: compare safety to step_max

  Vector3D<Precision> distance = safety_minus;
  const Vector3D<bool> direction_plus = direction < 0.0;
  distance.MaskedAssign(direction_plus, safety_plus);

  distance /= direction.Abs();

  const Precision min = distance.Min();
  return (min < 0) ? 0 : min;
    */
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision PlacedBox::SafetyToOut( Vector3D<Precision> const &position ) const {
   return SafetyToOutDispatch<kScalar>( position );
}


VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision PlacedBox::SafetyToIn( Vector3D<Precision> const &position ) const {
   return SafetyToInDispatch<1,0,kScalar>( position );
}




} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDBOX_H_
