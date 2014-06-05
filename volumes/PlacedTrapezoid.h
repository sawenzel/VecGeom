/*
 * @file   volumes/PlacedTrapezoid.h
 * @author Guilherme Lima (lima 'at' fnal 'dot' gov)
 *
 * 2014-05-01 - Created, based on the Parallelepiped draft
 */

#ifndef VECGEOM_VOLUMES_PLACEDTRAPEZOID_H_
#define VECGEOM_VOLUMES_PLACEDTRAPEZOID_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
// #include "volumes/UnplacedBox.h"
// #include "volumes/PlacedBox.h"
// #include "volumes/UnplacedTrapezoid.h"
#include "volumes/kernel/TrapezoidImplementation.h"

namespace VECGEOM_NAMESPACE {

class PlacedTrapezoid : public VPlacedVolume {

public:

  typedef UnplacedTrapezoid UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedTrapezoid(char const *const label,
                  LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation,
                  PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedTrapezoid(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation,
                  PlacedBox const *const boundingBox)
    : PlacedTrapezoid("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedTrapezoid(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation,
                  PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual ~PlacedTrapezoid();

  /// Accessors
  /// @{

  /* Retrieves the unplaced volume pointer from the logical volume and casts it
   * to an unplaced trapezoid.
   */
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid const* GetUnplacedVolume() const {
    return static_cast<UnplacedTrapezoid const *>(
        logical_volume()->unplaced_volume());
  }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDz() const { return GetUnplacedVolume()->GetDz(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTheta() const { return GetUnplacedVolume()->GetTheta(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhi() const { return GetUnplacedVolume()->GetPhi(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDy1() const { return GetUnplacedVolume()->GetDy1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx1() const { return GetUnplacedVolume()->GetDx1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx2() const { return GetUnplacedVolume()->GetDx2(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha1() const { return GetUnplacedVolume()->GetTanAlpha1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDy2() const { return GetUnplacedVolume()->GetDy2(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx3() const { return GetUnplacedVolume()->GetDx3(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx4() const { return GetUnplacedVolume()->GetDx4(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha2() const { return GetUnplacedVolume()->GetTanAlpha2(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha1() const { return GetUnplacedVolume()->GetAlpha1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha2() const { return GetUnplacedVolume()->GetAlpha2(); }

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

/*
protected:

  static PlacedBox* make_bounding_box(LogicalVolume const *const logical_volume,
                                      Transformation3D const *const transformation) {

    UnplacedTrapezoid const *const utrap = static_cast<UnplacedTrapezoid const *const>(logical_volume->unplaced_volume());
    UnplacedBox const *const unplaced_bbox = new UnplacedBox(
      std::max(std::max(utrap->GetDx1(),utrap->GetDx2()),std::max(utrap->GetDx3(),utrap->GetDx4())),
      std::max(utrap->GetDy1(),utrap->GetDy2()), utrap->GetDz());
    LogicalVolume const *const box_volume =  new LogicalVolume(unplaced_bbox);
    return new PlacedBox(box_volume, transformation);
  }


public:

  // CUDA specific

  // Templates to interact with common C-like kernels
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::bool_v InsideDispatch(
    Vector3D<typename Backend::precision_v> const &point) const;

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToInDispatch(
    Vector3D<typename Backend::precision_v> const &position,
    Vector3D<typename Backend::precision_v> const &direction,
    const typename Backend::precision_v step_max) const;

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToOutDispatch(
    Vector3D<typename Backend::precision_v> const &position,
    Vector3D<typename Backend::precision_v> const &direction,
    const typename Backend::precision_v step_max) const;

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v SafetyToOutDispatch(
    Vector3D<typename Backend::precision_v> const &position ) const;

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v SafetyToInDispatch(
    Vector3D<typename Backend::precision_v> const &position ) const;
*/
}; // end of class PlacedTrapezoid

//===== function definitions in header file  ====

/*
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::bool_v PlacedTrapezoid::InsideDispatch(
  Vector3D<typename Backend::precision_v> const &point) const {

  printf("(%s L%d) PlacedTrap::InsideDisp(): sizeof(point): %ld\n",__FILE__,__LINE__,sizeof(point));

  typename Backend::bool_v output;

  // GL: I give up... don't know how to call what I want...
  printf("\n(%s L%d) *** I can't figure out how to call Inside() method from here...\n\n");
  // Inside<translation::kGeneric, rotation::kGeneric, Backend>(
  //     *GetUnplacedVolume(), *this->transformation(), point, &output );

  return output;
}

template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v PlacedTrapezoid::DistanceToInDispatch(
    Vector3D<typename Backend::precision_v> const &position,
    Vector3D<typename Backend::precision_v> const &direction,
    const typename Backend::precision_v step_max) const {

  typename Backend::precision_v output;
  TrapezoidDistanceToIn<Backend>(
    *GetUnplacedVolume(),
    *this->transformation(),
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
typename Backend::precision_v PlacedTrapezoid::DistanceToOutDispatch(
    Vector3D<typename Backend::precision_v> const &position,
    Vector3D<typename Backend::precision_v> const &direction,
    const typename Backend::precision_v step_max) const {
  typename Backend::precision_v output;
  TrapezoidDistanceToOut<Backend>(
      *GetUnplacedVolume(),
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
typename Backend::precision_v PlacedTrapezoid::SafetyToOutDispatch(
        Vector3D<typename Backend::precision_v> const &position
         ) const {
  typename Backend::precision_v safety;
  TrapezoidSafetyToOut<Backend>(*GetUnplacedVolume(), position, safety);
  return safety;
}

template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v PlacedTrapezoid::SafetyToInDispatch(
    Vector3D<typename Backend::precision_v> const &position
) const {
  typename Backend::precision_v safety;
  TrapezoidSafetyToIn<Backend>(*GetUnplacedVolume(), *transformation(), position, safety);
  return safety;
}
*/
} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDTRAPEZOID_H_
