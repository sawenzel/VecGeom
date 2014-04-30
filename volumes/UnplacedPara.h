/// @file unplaced_para.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDPARA_H_
#define VECGEOM_VOLUMES_UNPLACEDPARA_H_

#include "base/global.h"

#include "volumes/unplaced_volume.h"

namespace VECGEOM_NAMESPACE {

class UnplacedPara : public UnplacedVolume {

private:

  Precision fX, fY, fZ;
  Precision fTanAlpha, fTanThetaCosPhi, fTanThetaSinPhi;

public:

  UnplacedPara(const Precision x, const Precision y, const Precision z,
               const Precision alpha, const Precision theta,
               const Precision phi);

  // Accessors

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetX() const { return fX; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetY() const { return fY; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetZ() const { return fZ; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha() const { return fTanAlpha; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaCosPhi() const { return fTanThetaCosPhi; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaSinPhi() const { return fTanThetaSinPhi; }

  // Mutators

  VECGEOM_CUDA_HEADER_BOTH
  void SetX(const Precision x) { fX = x; }

  VECGEOM_CUDA_HEADER_BOTH
  void SetY(const Precision y) { fY = y; }

  VECGEOM_CUDA_HEADER_BOTH
  void SetZ(const Precision z) { fZ = z; }

  VECGEOM_CUDA_HEADER_BOTH
  void SetAlpha(const Precision alpha);

  VECGEOM_CUDA_HEADER_BOTH
  void SetThetaAndPhi(const Precision theta, const Precision phi);

  // Virtual function implementations

  int memory_size() const { return sizeof(*this); }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const =0;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const =0;
#endif

  virtual void Print(std::ostream &os) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

private:

#ifndef VECGEOM_NVCC

  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) const;

  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL);

#else // Compiling for CUDA

  __device__
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL) const;

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

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPARA_H_