/**
 * @file placed_volume.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_VOLUMES_PLACEDVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDVOLUME_H_

#include "base/global.h"

#include "base/transformation3d.h"
#include "volumes/logical_volume.h"

#include <string>
#include <list>

namespace VECGEOM_NAMESPACE {

// Forward declaration for bounding box
class PlacedBox;

class VPlacedVolume {

private:

  int id_;
  // Use a pointer so the string won't be constructed on the GPU
  std::string *label_;
  static int g_id_count;

protected:

  LogicalVolume const *logical_volume_;
  Transformation3D const *transformation_;
  PlacedBox const *bounding_box_;

#ifndef VECGEOM_NVCC

  VPlacedVolume(char const *const label,
                LogicalVolume const *const logical_volume,
                Transformation3D const *const transformation,
                PlacedBox const *const bounding_box);

  VPlacedVolume(LogicalVolume const *const logical_volume,
                Transformation3D const *const transformation,
                PlacedBox const *const bounding_box)
      :  VPlacedVolume("", logical_volume, transformation, bounding_box) {}

#else

  __device__
  VPlacedVolume(LogicalVolume const *const logical_volume,
                Transformation3D const *const transformation,
                PlacedBox const *const bounding_box,
                const int id)
      : logical_volume_(logical_volume), transformation_(transformation),
        bounding_box_(bounding_box), id_(id), label_(NULL) {}

#endif

public:

  virtual ~VPlacedVolume();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int id() const { return id_; }

  std::string label() const { return *label_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  PlacedBox const* bounding_box() const { return bounding_box_; }


  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  LogicalVolume const* logical_volume() const {
    return logical_volume_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector<Daughter> const& daughters() const {
    return logical_volume_->daughters();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume const* unplaced_volume() const {
    return logical_volume_->unplaced_volume();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Transformation3D const* transformation() const {
    return transformation_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void set_logical_volume(LogicalVolume const *const logical_volume) {
    logical_volume_ = logical_volume;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void set_transformation(Transformation3D const *const transformation) {
    transformation_ = transformation;
  }

  void set_label(char const *const label) { *label_ = label; }

  friend std::ostream& operator<<(std::ostream& os, VPlacedVolume const &vol);

  virtual int memory_size() const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print(const int indent = 0) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const =0;

  /**
   * Recursively prints contained volumes.
   */
  VECGEOM_CUDA_HEADER_BOTH
  void PrintContent(const int depth = 0) const;

  // Geometry functionality

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool Inside(Vector3D<Precision> const &point) const;

  virtual void Inside(SOA3D<Precision> const &point,
                      bool *const output) const =0;

  virtual void Inside(AOS3D<Precision> const &point,
                      bool *const output) const =0;

  /** an inside function that gives back the localpoint in the reference frame of the callee
   * this is useful for the locate function
   **/
  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Inside(Vector3D<Precision> const &point,
                       Vector3D<Precision> & localpoint) const =0;

  /** An inside function where we know that localpoint is already in the
   *  reference frame of the callee
   */
  VECGEOM_CUDA_HEADER_BOTH
  virtual bool UnplacedInside(Vector3D<Precision> const &localpoint) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &position,
                                 Vector3D<Precision> const &direction,
                                 const Precision step_max = kInfinity) const =0;

  virtual void DistanceToIn(SOA3D<Precision> const &position,
                            SOA3D<Precision> const &direction,
                            Precision const *const step_max,
                            Precision *const output) const =0;

  virtual void DistanceToIn(AOS3D<Precision> const &position,
                            AOS3D<Precision> const &direction,
                            Precision const *const step_max,
                            Precision *const output) const =0;


  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(
                               Vector3D<Precision> const &position,
                               Vector3D<Precision> const &direction,
                               Precision const step_max = kInfinity) const =0;

  virtual void DistanceToOut(SOA3D<Precision> const &position,
                              SOA3D<Precision> const &direction,
                              Precision const *const step_max,
                              Precision *const output) const =0;

  virtual void DistanceToOut(AOS3D<Precision> const &position,
                              AOS3D<Precision> const &direction,
                              Precision const *const step_max,
                              Precision *const output) const =0;

  // interfaces for safety
  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToOut( Vector3D<Precision> const &position ) const =0;
  virtual void SafetyToOut( SOA3D<Precision> const &position, Precision *const safeties ) const =0;
  virtual void SafetyToOut( AOS3D<Precision> const &position, Precision *const safeties ) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToIn( Vector3D<Precision> const &position ) const =0;
  virtual void SafetyToIn( SOA3D<Precision> const &position, Precision *const safeties ) const =0;
  virtual void SafetyToIn( AOS3D<Precision> const &position, Precision *const safeties ) const =0;

protected:

  // Implemented by the vector backend

  template <typename VolumeType, typename ContainerType>
  VECGEOM_INLINE
  static void Inside_Looper(VolumeType const &volume,
                            ContainerType const &points,
                            bool *const output);

  template <typename VolumeType, typename ContainerType>
  VECGEOM_INLINE
  static void DistanceToIn_Looper(VolumeType const &volume,
                                  ContainerType const &positions,
                                  ContainerType const &directions,
                                  Precision const *const step_max,
                                  Precision *const output);


  template <typename VolumeType, typename ContainerType>
  VECGEOM_INLINE
  static void DistanceToOut_Looper(VolumeType const &volume,
                                   ContainerType const &positions,
                                   ContainerType const &directions,
                                   Precision const *const step_max,
                                   Precision *const output);

  template <typename VolumeType, typename ContainerType>
  VECGEOM_INLINE
  static void SafetyToIn_Looper(VolumeType const &volume,
        ContainerType const &positions,
        Precision *const output);


  template <typename VolumeType, typename ContainerType>
  VECGEOM_INLINE
  static void SafetyToOut_Looper(VolumeType const &volume,
        ContainerType const &positions,
        Precision *const output);

public:

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation,
                                   VPlacedVolume *const gpu_ptr) const =0;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation) const =0;
#endif

#ifdef VECGEOM_BENCHMARK
  virtual VPlacedVolume const* ConvertToUnspecialized() const =0;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const =0;
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const =0;
#endif
#endif // VECGEOM_BENCHMARK

};

VECGEOM_CUDA_HEADER_BOTH
bool VPlacedVolume::Inside(Vector3D<Precision> const &point) const {
  Vector3D<Precision> localPoint;
  return Inside(point, localPoint);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDVOLUME_H_
