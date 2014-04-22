/**
 * @file specialized_trap.h
 * @author Guilherme Lima (Guilherme.Lima@cern.ch)
 *
 * 140407 G.Lima - based on equivalent box code
 */

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTRAP_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTRAP_H_

#include "base/global.h"
#include "backend/backend.h"
#include "backend/implementation.h"
#include "base/transformation_matrix.h"
#include "volumes/placed_trap.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode trans_code, RotationCode rot_code>
class SpecializedTrap : public PlacedTrap {

public:

  SpecializedTrap(char const *const label,
				  LogicalVolume const *const logical_volume,
				  TransformationMatrix const *const matrix)
      : PlacedTrap(label, logical_volume, matrix) {}

#ifdef VECGEOM_STD_CXX11
	SpecializedTrap(LogicalVolume const *const logical_volume,
					TransformationMatrix const *const matrix)
		: SpecializedTrap<trans_code, rot_code>("", logical_volume, matrix) {}
#endif

#ifdef VECGEOM_NVCC
  VECGEOM_CUDA_HEADER_DEVICE
  SpecializedTrap(LogicalVolume const *const logical_volume,
				  TransformationMatrix const *const matrix, const int id)
	  : PlacedTrap(logical_volume, matrix, id) {}
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;

  virtual ~SpecializedTrap(){}

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

  virtual int memory_size() const { return sizeof(*this); }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   TransformationMatrix const *const matrix,
                                   VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      TransformationMatrix const *const matrix) const;
#endif

};

template <TranslationCode trans_code, RotationCode rot_code>
VECGEOM_CUDA_HEADER_BOTH
void SpecializedTrap<trans_code, rot_code>::PrintType() const {
  printf("SpecializedTrap<%i, %i>", trans_code, rot_code);
}

template <TranslationCode trans_code, RotationCode rot_code>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedTrap<trans_code, rot_code>::Inside(
    Vector3D<Precision> const &point) const {
  return InsideDispatch<trans_code, rot_code, kScalar>(point);
}

template <TranslationCode trans_code, RotationCode rot_code>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedTrap<trans_code, rot_code>::Inside(
    Vector3D<Precision> const &point,
    Vector3D<Precision> &local) const {
  bool output;
  TrapInside<trans_code, rot_code, kScalar>( unplaced_trap()->parameters(),
                                             *this->matrix_, point, local,
											 &output );
  return output;
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedTrap<trans_code, rot_code>::Inside(
    SOA3D<Precision> const &points,
    bool *const output) const {
  Inside_Looper<trans_code, rot_code>(*this, points, output);
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedTrap<trans_code, rot_code>::Inside(
    AOS3D<Precision> const &points,
    bool *const output) const {
  Inside_Looper<trans_code, rot_code>(*this, points, output);
}

template <TranslationCode trans_code, RotationCode rot_code>
VECGEOM_CUDA_HEADER_BOTH
Precision SpecializedTrap<trans_code, rot_code>::DistanceToIn(
    Vector3D<Precision> const &position,
    Vector3D<Precision> const &direction,
    const Precision step_max) const {

  return DistanceToInDispatch<trans_code, rot_code, kScalar>(
           position, direction, step_max
         );
                                                  
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedTrap<trans_code, rot_code>::DistanceToIn(
    SOA3D<Precision> const &positions,
    SOA3D<Precision> const &directions,
    Precision const *const step_max,
    Precision *const output) const {
  DistanceToIn_Looper<trans_code, rot_code>(*this, positions, directions,
                                            step_max, output);
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedTrap<trans_code, rot_code>::DistanceToIn(
    AOS3D<Precision> const &positions,
    AOS3D<Precision> const &directions,
    Precision const *const step_max,
    Precision *const output) const {
  DistanceToIn_Looper<trans_code, rot_code>(*this, positions, directions,
                                            step_max, output);
}

template <TranslationCode trans_code, RotationCode rot_code>
Precision SpecializedTrap<trans_code, rot_code>::SafetyToIn(
		Vector3D<Precision> const &position ) const {
	return SafetyToInDispatch<trans_code,rot_code,kScalar>( position );
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedTrap<trans_code, rot_code>::SafetyToIn(
		SOA3D<Precision> const &position, Precision *const safeties ) const {
	SafetyToIn_Looper<trans_code, rot_code>(*this, position, safeties);
}

template <TranslationCode trans_code, RotationCode rot_code>
void SpecializedTrap<trans_code, rot_code>::SafetyToIn(
		AOS3D<Precision> const &position, Precision *const safeties ) const {
	SafetyToIn_Looper<trans_code, rot_code>(*this, position, safeties);
}

} // End global namespace

namespace vecgeom {

class LogicalVolume;
class TransformationMatrix;

void SpecializedTrap_CopyToGpu(int trans_code, int rot_code,
							   LogicalVolume const *const logical_volume,
							   TransformationMatrix const *const matrix,
							   const int id, VPlacedVolume *const gpu_ptr);

#ifdef VECGEOM_CUDA_INTERFACE

template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume* SpecializedTrap<trans_code, rot_code>::CopyToGpu(
    LogicalVolume const *const logical_volume,
    TransformationMatrix const *const matrix,
    VPlacedVolume *const gpu_ptr) const {

  SpecializedTrap_CopyToGpu(trans_code, rot_code, logical_volume, matrix,
                           this->id(), gpu_ptr);
  vecgeom::CudaAssertError();
  return gpu_ptr;

}

template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume* SpecializedTrap<trans_code, rot_code>::CopyToGpu(
    LogicalVolume const *const logical_volume,
    TransformationMatrix const *const matrix) const {

  VPlacedVolume *const gpu_ptr =
      vecgeom::AllocateOnGpu<SpecializedTrap<trans_code, rot_code> >();
  return this->CopyToGpu(logical_volume, matrix, gpu_ptr);  

}

#endif // VECGEOM_CUDA_INTERFACE

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDTRAP_H_
