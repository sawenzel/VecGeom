/// \file placed_volume.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDVOLUME_H_

#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/LogicalVolume.h"
#include "volumes/USolidsInterfaceHelper.h"

#include <list>
#include <string>

namespace VECGEOM_NAMESPACE {

// Forward declaration for bounding box
class PlacedBox;

class VPlacedVolume : public USolidsInterfaceHelper {

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
                LogicalVolume const *const logical_vol,
                Transformation3D const *const transform,
                PlacedBox const *const boundingbox);

  VPlacedVolume(LogicalVolume const *const logical_vol,
                Transformation3D const *const transform,
                PlacedBox const *const boundingbox)
      :  VPlacedVolume("", logical_vol, transform, boundingbox) {}

#else

  __device__
  VPlacedVolume(LogicalVolume const *const logical_vol,
                Transformation3D const *const transformation,
                PlacedBox const *const boundingbox,
                const int id)
      : logical_volume_(logical_vol), transformation_(transform),
        bounding_box_(boundingbox), id_(id), label_(NULL) {}

#endif

  VPlacedVolume(VPlacedVolume const &other);
  VPlacedVolume& operator=(VPlacedVolume const &other);

public:

  virtual ~VPlacedVolume();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int id() const { return id_; }

  std::string GetLabel() const { return *label_; }

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
  void set_logical_volume(LogicalVolume const *const logical_vol) {
    logical_volume_ = logical_vol;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void set_transformation(Transformation3D const *const transform) {
    transformation_ = transform;
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
  virtual bool Contains(Vector3D<Precision> const &point) const =0;

  virtual void Contains(SOA3D<Precision> const &point,
                        bool *const output) const =0;

  virtual void Contains(AOS3D<Precision> const &point,
                        bool *const output) const =0;

  /// \return The input point transformed to the local reference frame.
  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Contains(Vector3D<Precision> const &point,
                        Vector3D<Precision> &localPoint) const =0;

  /// \param localPoint Point in the local reference frame of the volume.
  VECGEOM_CUDA_HEADER_BOTH
  virtual bool UnplacedContains(Vector3D<Precision> const &localPoint) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Inside_t Inside(Vector3D<Precision> const &point) const =0;

  virtual void Inside(SOA3D<Precision> const &point,
                      Inside_t *const output) const =0;

  virtual void Inside(AOS3D<Precision> const &point,
                      Inside_t *const output) const =0;

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
                                   Transformation3D const *const transform,
                                   VPlacedVolume *const gpu_ptr) const =0;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transform) const =0;
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

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDVOLUME_H_
