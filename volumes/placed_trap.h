/**
 * @file placed_trap.h
 * @author Guilherme Lima (Guilherme.Lima@cern.ch)
 *
 * 140407 G.Lima - based on equivalent box code
 */

#ifndef VECGEOM_VOLUMES_PLACEDTRAP_H_
#define VECGEOM_VOLUMES_PLACEDTRAP_H_

#include "base/global.h"
#include "backend/backend.h"

#include "volumes/placed_volume.h"
#include "volumes/unplaced_trap.h"
#include "volumes/kernel/trap_kernel.h"
#include "volumes/unplaced_box.h"
#include "volumes/placed_box.h"

namespace VECGEOM_NAMESPACE {

class PlacedTrap : public VPlacedVolume {

public:

    PlacedTrap(char const *const label,
               LogicalVolume const *const logical_volume,
               TransformationMatrix const *const matrix)
        : VPlacedVolume(label, logical_volume, matrix, NULL)
  {
    UnplacedTrap const* unplacedMe = static_cast<UnplacedTrap const*>( logical_volume->unplaced_volume() );

    // use the trap parameters to find the bounding box position and dimensions
    TrapCorners_t corners;
    unplacedMe -> fromParametersToCorners( corners );

    Precision bboxDz = corners[6].z();

    Precision bboxYp = Max( corners[2].y(), corners[6].y() );
    Precision bboxYm = Min( corners[0].y(), corners[4].y() );
    Precision bboxDy = 0.5 * (bboxYp - bboxYm);

    Precision bboxXpZp = Max( corners[5].x(), corners[7].x() );
    Precision bboxXpZm = Max( corners[1].x(), corners[3].x() );
    Precision bboxXmZp = Min( corners[4].x(), corners[6].x() );
    Precision bboxXmZm = Min( corners[0].x(), corners[2].x() );
    Precision bboxXp = Max( bboxXpZp, bboxXpZm );
    Precision bboxXm = Min( bboxXmZp, bboxXmZm );
    Precision bboxDx = 0.5 * (bboxXp - bboxXm);

    // create the required bounding box
    UnplacedBox const* pUnplacedBBox = new UnplacedBox( bboxDx, bboxDy, bboxDz );

    LogicalVolume* logVol = new LogicalVolume( pUnplacedBBox );
    logVol->Place( new TransformationMatrix{ 0.5*(bboxXp+bboxXm), 0.5*(bboxYp+bboxYm), 0.} );

    VPlacedVolume const* pBoundingBox = pUnplacedBBox->PlaceVolume(logVol,matrix);
    VPlacedVolume::set_bounding_box( pBoundingBox );
  }

#ifdef VECGEOM_STD_CXX11
  PlacedTrap(LogicalVolume const *const logical_volume,
            TransformationMatrix const *const matrix)
      : PlacedTrap("", logical_volume, matrix) {}
#endif

#ifdef VECGEOM_NVCC
  VECGEOM_CUDA_HEADER_DEVICE
  PlacedTrap(LogicalVolume const *const logical_volume,
             TransformationMatrix const *const matrix,
             const int id)
      : VPlacedVolume(logical_volume, matrix, NULL, id) {}
#endif

  virtual ~PlacedTrap() {
    delete this->logical_volume_;
    delete this->matrix_;
    delete this->bounding_box_;
  }

  // Accessors

  VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    TrapParameters const& parameters() const { return unplaced_trap()->parameters(); }

  VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    Precision getDz()    const { return unplaced_trap()->getDz(); }

    VECGEOM_CUDA_HEADER_BOTH
      VECGEOM_INLINE
      Precision getDy1() const { return unplaced_trap()->getDy1(); }

    VECGEOM_CUDA_HEADER_BOTH
      VECGEOM_INLINE
      Precision getDx1() const { return unplaced_trap()->getDx1(); }

    VECGEOM_CUDA_HEADER_BOTH
      VECGEOM_INLINE
      Precision getDx2() const { return unplaced_trap()->getDx2(); }

    VECGEOM_CUDA_HEADER_BOTH
      VECGEOM_INLINE
      Precision getTalpha1()    const { return unplaced_trap()->getTalpha1(); }

    VECGEOM_CUDA_HEADER_BOTH
      VECGEOM_INLINE
      Precision getDy2() const { return unplaced_trap()->getDy2(); }

    VECGEOM_CUDA_HEADER_BOTH
      VECGEOM_INLINE
      Precision getDx3() const { return unplaced_trap()->getDx3(); }

    VECGEOM_CUDA_HEADER_BOTH
      VECGEOM_INLINE
      Precision getDx4() const { return unplaced_trap()->getDx4(); }

    VECGEOM_CUDA_HEADER_BOTH
      VECGEOM_INLINE
      Precision getTalpha2()    const { return unplaced_trap()->getTalpha2(); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;

protected:

    /**
     * Retrieves the unplaced volume pointer from the logical volume and casts it
     * to an unplaced trap.
     */
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    UnplacedTrap const* unplaced_trap() const {
        return static_cast<UnplacedTrap const*>(this->unplaced_volume());
    }

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
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   TransformationMatrix const *const matrix,
                                   VPlacedVolume *const gpu_ptr) const;

  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
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
  template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::bool_v InsideDispatch( Vector3D<typename Backend::precision_v> const &point ) const;

  template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v
  DistanceToInDispatch( Vector3D<typename Backend::precision_v> const &position,
                        Vector3D<typename Backend::precision_v> const &direction,
                        const typename Backend::precision_v step_max ) const;

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v
  DistanceToOutDispatch( Vector3D<typename Backend::precision_v> const &position,
                         Vector3D<typename Backend::precision_v> const &direction,
                         const typename Backend::precision_v step_max ) const;

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v
  SafetyToOutDispatch( Vector3D<typename Backend::precision_v> const &position ) const;

  template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v
  SafetyToInDispatch( Vector3D<typename Backend::precision_v> const &position ) const;

};


template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::bool_v
PlacedTrap::InsideDispatch( Vector3D<typename Backend::precision_v> const &point) const {

  typename Backend::bool_v output;
  TrapInside<trans_code, rot_code, Backend>( unplaced_trap()->parameters(),
                                             *this->matrix(), point, &output );
  return output;
}

template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v
PlacedTrap::DistanceToInDispatch( Vector3D<typename Backend::precision_v> const &position,
                                  Vector3D<typename Backend::precision_v> const &direction,
                                  const typename Backend::precision_v step_max ) const {

  typename Backend::precision_v output;
  TrapDistanceToIn<trans_code, rot_code, Backend>( unplaced_trap()->parameters(),
                           *this->matrix(), position, direction, step_max, &output );
  return output;
}

template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v
PlacedTrap::DistanceToOutDispatch( Vector3D<typename Backend::precision_v> const &position,
                                   Vector3D<typename Backend::precision_v> const &direction,
                                   const typename Backend::precision_v step_max ) const {

      typename Backend::precision_v output;
      TrapDistanceToOut<Backend>( unplaced_trap()->parameters(),
                  position,    direction, step_max, output );
      return output;
}


template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v
PlacedTrap::SafetyToOutDispatch( Vector3D<typename Backend::precision_v> const &position ) const {
      typename Backend::precision_v safety;
      TrapSafetyToOut<Backend>(unplaced_trap()->parameters(), position, safety);
      return safety;
}


template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v
PlacedTrap::SafetyToInDispatch( Vector3D<typename Backend::precision_v> const &position ) const {
    typename Backend::precision_v safety;
    TrapSafetyToIn<trans_code, rot_code, Backend>(unplaced_trap()->parameters(), *matrix(), position, safety);
    return safety;
}


VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool PlacedTrap::Inside(Vector3D<Precision> const &point) const {
      return PlacedTrap::InsideDispatch<1, 0, kScalar>(point);
}

VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool PlacedTrap::UnplacedInside(Vector3D<Precision> const &localpoint) const {
      // no translation and no rotation should look like this:
      return PlacedTrap::InsideDispatch<0, rotation::kIdentity, kScalar>( localpoint );
}

VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool PlacedTrap::Inside(Vector3D<Precision> const &point, Vector3D<Precision>& localpoint) const {
      typename kScalar::bool_v output;
      TrapInside<1, 0, kScalar>( unplaced_trap()->parameters(),
                 *this->matrix(), point, localpoint, &output );
      return output;
}

VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision PlacedTrap::DistanceToIn(Vector3D<Precision> const &position,
                     Vector3D<Precision> const &direction,
                     const Precision step_max) const {

      return PlacedTrap::DistanceToInDispatch<1, 0, kScalar>(position, direction, step_max);
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision PlacedTrap::DistanceToOut(Vector3D<Precision> const &position,
                                    Vector3D<Precision> const &direction,
                                    Precision const step_max) const {

    // for the moment we dispatch to the common kernel
    return PlacedTrap::DistanceToOutDispatch<kScalar>(position,direction,step_max);

    // this kernel is an optimized "kInternalVectorization" kernel
    // we have how we can dispatch to this kernel later

    /*
    // const Vector3D<Precision> local = matrix()->Transform<1, 0>(position);

    Vector3D<Precision> const &dim = unplaced_trap()->parameters();

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
Precision PlacedTrap::SafetyToOut( Vector3D<Precision> const &position ) const {
   return SafetyToOutDispatch<kScalar>( position );
}


VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision PlacedTrap::SafetyToIn( Vector3D<Precision> const &position ) const {
   return SafetyToInDispatch<1,0,kScalar>( position );
}

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDTRAP_H_
