#ifndef VECGEOM_VOLUMES_PLACEDVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDVOLUME_H_

#include "base/global.h"
#include "base/transformation_matrix.h"
#include "management/geo_manager.h"
#include "volumes/logical_volume.h"

namespace vecgeom {

class VPlacedVolume {

private:

  // int id;

  friend class CudaManager;

protected:

  LogicalVolume const *logical_volume_;
  TransformationMatrix const *matrix_;

public:

  VECGEOM_CUDA_HEADER_BOTH
  VPlacedVolume(LogicalVolume const *const logical_volume__,
                TransformationMatrix const *const matrix__)
      : logical_volume_(logical_volume__), matrix_(matrix__) {}

  virtual ~VPlacedVolume() {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  LogicalVolume const* logical_volume() const {
    return logical_volume_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume const* unplaced_volume() const {
    return logical_volume()->unplaced_volume();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  TransformationMatrix const* matrix() const {
    return matrix_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void set_logical_volume(LogicalVolume const *const logical_volume__) {
    logical_volume_ = logical_volume__;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void set_matrix(TransformationMatrix const *const matrix__) {
    matrix_ = matrix__;
  }

  VECGEOM_CUDA_HEADER_HOST
  friend std::ostream& operator<<(std::ostream& os, VPlacedVolume const &vol);

  virtual int memory_size() const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Inside(Vector3D<Precision> const &point) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &position,
                                 Vector3D<Precision> const &direction,
                                 const Precision step_max) const =0;

  #ifdef VECGEOM_CUDA
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   TransformationMatrix const *const matrix,
                                   VPlacedVolume *const gpu_ptr) const =0;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      TransformationMatrix const *const matrix) const =0;
  #endif

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDVOLUME_H_]