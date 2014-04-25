/**
 * @file specialized_box.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_VOLUMES_SPECIALIZEDBOX_H_
#define VECGEOM_VOLUMES_SPECIALIZEDBOX_H_

#include "base/global.h"
#include "backend/backend.h"
#include "backend/implementation.h"
#include "base/transformation3d.h"
#include "volumes/placed_box.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode trans_code, RotationCode rot_code>
class SpecializedBox : public PlacedBox {

public:

  SpecializedBox(char const *const label,
                 LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation)
      : PlacedBox(label, logical_volume, transformation) {}

#ifdef VECGEOM_STD_CXX11
  SpecializedBox(LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation)
      : SpecializedBox<trans_code, rot_code>("", logical_volume,
                                             transformation) {}
#endif

#ifdef VECGEOM_NVCC
  VECGEOM_CUDA_HEADER_DEVICE
  SpecializedBox(LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation,
                 const int id)
      : PlacedBox(logical_volume, transformation, id) {}
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;

    virtual ~SpecializedBox(){}

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Inside(Vector3D<Precision> const &point) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Inside(Vector3D<Precision> const &point,
                      Vector3D<Precision> &local) const;

  virtual void Inside(SOA3D<Precision> const &points,
                      bool *const output) const;

  virtual void Inside(AOS3D<Precision> const &points,
                      bool *const output) const;

  VECGEOM_CUDA_HEADER_BOTH
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

  // we need to specialize SafetyToIn
  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToIn( Vector3D<Precision> const &position ) const;
  virtual void SafetyToIn( SOA3D<Precision> const &position, Precision *const safeties ) const;
  virtual void SafetyToIn( AOS3D<Precision> const &position, Precision *const safeties ) const;

  // for specialized box we need special dispatchers for
  // SafetyToIn, Inside, DistanceToIn
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  // TODO: why is this here? Dispatches should only be in Placed volume code
  typename Backend::precision_v DistanceToInDispatch(
      Vector3D<typename Backend::precision_v> const &position,
      Vector3D<typename Backend::precision_v> const &direction,
      const typename Backend::precision_v step_max) const {

      typename Backend::precision_v output;
      BoxDistanceToIn<trans_code, rot_code, Backend>(
          unplaced_box()->dimensions(), *this->transformation(),
          position, direction,
          step_max, &output );
      return output;
  }

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v SafetyToInDispatch(
        Vector3D<typename Backend::precision_v> const &position
  ) const {
     typename Backend::precision_v safety;
     BoxSafetyToIn<trans_code, rot_code, Backend>(unplaced_box()->dimensions(), *transformation(), position, safety);
     return safety;
  }

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::bool_v InsideDispatch(
      Vector3D<typename Backend::precision_v> const &point) const {

    typename Backend::bool_v output;
    BoxInside<trans_code, rot_code, Backend>(
      unplaced_box()->dimensions(),
      *this->transformation(),
      point,
      &output
    );
    return output;
  }

  virtual int memory_size() const { return sizeof(*this); }

  #ifdef VECGEOM_CUDA_INTERFACE
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation,
                                   VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation) const;
  #endif

};

template <TranslationCode trans_code, RotationCode rot_code>
VECGEOM_CUDA_HEADER_BOTH
void SpecializedBox<trans_code, rot_code>::PrintType() const {
  printf("SpecializedBox<%i, %i>", trans_code, rot_code);
}

template <TranslationCode trans_code, RotationCode rot_code>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedBox<trans_code, rot_code>::Inside(
    Vector3D<Precision> const &point) const {
  return InsideDispatch<kScalar>(point);
}

template <TranslationCode trans_code, RotationCode rot_code>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedBox<trans_code, rot_code>::Inside(
    Vector3D<Precision> const &point,
    Vector3D<Precision> &local) const {
  bool output;
  BoxInside<trans_code,rot_code,kScalar>(
    unplaced_box()->dimensions(),
    *this->transformation_,
    point,
    local,
    &output
  );
  return output;
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedBox<trans_code, rot_code>::Inside(
    SOA3D<Precision> const &points,
    bool *const output) const {
  Inside_Looper(*this, points, output);
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedBox<trans_code, rot_code>::Inside(
    AOS3D<Precision> const &points,
    bool *const output) const {
  Inside_Looper(*this, points, output);
}

template <TranslationCode trans_code, RotationCode rot_code>
VECGEOM_CUDA_HEADER_BOTH
Precision SpecializedBox<trans_code, rot_code>::DistanceToIn(
    Vector3D<Precision> const &position,
    Vector3D<Precision> const &direction,
    const Precision step_max) const {

  return DistanceToInDispatch<kScalar>(
           position, direction, step_max
         );
                                                  
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedBox<trans_code, rot_code>::DistanceToIn(
    SOA3D<Precision> const &positions,
    SOA3D<Precision> const &directions,
    Precision const *const step_max,
    Precision *const output) const {
  DistanceToIn_Looper(*this, positions, directions,
                                            step_max, output);
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedBox<trans_code, rot_code>::DistanceToIn(
    AOS3D<Precision> const &positions,
    AOS3D<Precision> const &directions,
    Precision const *const step_max,
    Precision *const output) const {
  DistanceToIn_Looper(*this, positions, directions,
                                            step_max, output);
}

template <TranslationCode trans_code, RotationCode rot_code>
Precision SpecializedBox<trans_code, rot_code>::SafetyToIn(
      Vector3D<Precision> const &position ) const {
   return SafetyToInDispatch<kScalar>( position );
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedBox<trans_code, rot_code>::SafetyToIn(
      SOA3D<Precision> const &position, Precision *const safeties ) const {
   SafetyToIn_Looper(*this, position, safeties);
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedBox<trans_code, rot_code>::SafetyToIn(
      AOS3D<Precision> const &position, Precision *const safeties ) const {
   SafetyToIn_Looper(*this, position, safeties);
}

} // End global namespace

namespace vecgeom {

class LogicalVolume;
class Transformation3D;

void SpecializedBox_CopyToGpu(int trans_code, int rot_code,
                              LogicalVolume const *const logical_volume,
                              Transformation3D const *const transformation,
                              const int id, VPlacedVolume *const gpu_ptr);

#ifdef VECGEOM_CUDA_INTERFACE

template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume* SpecializedBox<trans_code, rot_code>::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {

  SpecializedBox_CopyToGpu(trans_code, rot_code, logical_volume, transformation,
                           this->id(), gpu_ptr);
  vecgeom::CudaAssertError();
  return gpu_ptr;

}

template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume* SpecializedBox<trans_code, rot_code>::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {

  VPlacedVolume *const gpu_ptr =
      vecgeom::AllocateOnGpu<SpecializedBox<trans_code, rot_code> >();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);  

}

#endif // VECGEOM_CUDA_INTERFACE

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDBOX_H_
