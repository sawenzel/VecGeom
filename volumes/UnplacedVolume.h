/**
 * @file unplaced_volume.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_VOLUMES_UNPLACEDVOLUME_H_
#define VECGEOM_VOLUMES_UNPLACEDVOLUME_H_

#include "base/Global.h"

#include "base/Transformation3D.h"

#include <string>
#include <ostream>

namespace VECGEOM_NAMESPACE {

class LogicalVolume;
class VPlacedVolume;

class VUnplacedVolume {

private:

  friend class CudaManager;

public:

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~VUnplacedVolume() {}

  /**
   * Uses the virtual print method.
   * \sa print(std::ostream &ps)
   */
  friend std::ostream& operator<<(std::ostream& os, VUnplacedVolume const &vol);

  /**
   * Should return the size of bytes of the deriving class. Necessary for
   * copying to the GPU.
   */
  virtual int memory_size() const =0;

  /**
   * Constructs the deriving class on the GPU and returns a pointer to GPU
   * memory where the object has been instantiated.
   */
#ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const =0;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const =0;
#endif

  /**
   * Virtual print to accommodate outstreams.
   */
  virtual void Print(std::ostream &os) const =0;

  /**
   * C-style printing for CUDA purposes.
   */
  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const =0;

  // Is not static because a virtual function must be called to initialize
  // specialized volume as the shape of the deriving class.
  VPlacedVolume* PlaceVolume(
      char const *const label,
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      VPlacedVolume *const placement = NULL) const;

  VPlacedVolume* PlaceVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      VPlacedVolume *const placement = NULL) const;

private:

#ifndef VECGEOM_NVCC

  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) const =0;

#else

  __device__
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL) const =0;

#endif

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDVOLUME_H_
