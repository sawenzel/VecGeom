/// @file placed_root_volume.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDROOTVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDROOTVOLUME_H_

#include "base/global.h"

#include "volumes/placed_volume.h"

#include "TGeoShape.h"

class TGeoShape;

namespace vecgeom {

class PlacedRootVolume : public VPlacedVolume {

private:

  TGeoShape const *fRootShape;

public:

  PlacedRootVolume(char const *const label,
                   TGeoShape const *const rootShape,
                   LogicalVolume const *const logicalVolume,
                   Transformation3D const *const transformation);

  PlacedRootVolume(TGeoShape const *const rootShape,
                   LogicalVolume const *const logicalVolume,
                   Transformation3D const *const transformation);

  virtual ~PlacedRootVolume() {}

  TGeoShape const* GetRootShape() const { return fRootShape; }

  virtual int memory_size() const { return sizeof(*this); }

  virtual void PrintType() const;

  VECGEOM_INLINE
  virtual bool Inside(Vector3D<Precision> const &point) const;

  VECGEOM_INLINE
  virtual bool Inside(Vector3D<Precision> const &point,
                      Vector3D<Precision> &localpoint) const;

  virtual void Inside(SOA3D<Precision> const &points,
                      bool *const output) const;

  virtual void Inside(AOS3D<Precision> const &points,
                      bool *const output) const;

  VECGEOM_INLINE
  virtual bool UnplacedInside(Vector3D<Precision> const &point) const;

  VECGEOM_INLINE
  virtual Precision DistanceToIn(Vector3D<Precision> const &position,
                                 Vector3D<Precision> const &direction,
                                 const Precision step_max) const;

  virtual void DistanceToIn(SOA3D<Precision> const &position,
                            SOA3D<Precision> const &direction,
                            Precision const *const stepMax,
                            Precision *const output) const;

  virtual void DistanceToIn(AOS3D<Precision> const &position,
                            AOS3D<Precision> const &direction,
                            Precision const *const stepMax,
                            Precision *const output) const;

  VECGEOM_INLINE
  virtual Precision DistanceToOut(Vector3D<Precision> const &position,
                                  Vector3D<Precision> const &direction,
                                  Precision const stepMax) const;

  virtual void DistanceToOut(SOA3D<Precision> const &position,
                             SOA3D<Precision> const &direction,
                             Precision const *const step_max,
                             Precision *const output) const;

  virtual void DistanceToOut(AOS3D<Precision> const &position,
                             AOS3D<Precision> const &direction,
                             Precision const *const stepMax,
                             Precision *const output) const;

  VECGEOM_INLINE
  virtual Precision SafetyToOut(Vector3D<Precision> const &position) const;

  virtual void SafetyToOut(SOA3D<Precision> const &position,
                           Precision *const safeties) const;

  virtual void SafetyToOut(AOS3D<Precision> const &position,
                           Precision *const safeties) const;

  VECGEOM_INLINE
  virtual Precision SafetyToIn(Vector3D<Precision> const &position) const;

  virtual void SafetyToIn(SOA3D<Precision> const &position,
                          Precision *const safeties) const;

  virtual void SafetyToIn(AOS3D<Precision> const &position,
                          Precision *const safeties) const;

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

};

bool PlacedRootVolume::Inside(Vector3D<Precision> const &point) const {
  const Vector3D<Precision> local = this->transformation_->Transform(point);
  return UnplacedInside(local);
}

bool PlacedRootVolume::Inside(Vector3D<Precision> const &point,
                              Vector3D<Precision> &localPoint) const {
  localPoint = this->transformation_->Transform(point);
  return UnplacedInside(localPoint);
}

bool PlacedRootVolume::UnplacedInside(Vector3D<Precision> const &point) const {
  Vector3D<Precision> pointCopy = point; // ROOT expects non const input
  return fRootShape->Contains(&pointCopy[0]);
}

Precision PlacedRootVolume::DistanceToIn(Vector3D<Precision> const &position,
                                         Vector3D<Precision> const &direction,
                                         const Precision stepMax) const {
  Vector3D<Precision> positionLocal =
      this->transformation_->Transform(position);
  Vector3D<Precision> directionLocal =
      this->transformation_->TransformDirection(direction);
  return fRootShape->DistFromOutside(
           &positionLocal[0],
           &directionLocal[0],
           1,
           (stepMax == kInfinity) ? TGeoShape::Big() : stepMax
         ); 
}

VECGEOM_INLINE
Precision PlacedRootVolume::DistanceToOut(Vector3D<Precision> const &position,
                                          Vector3D<Precision> const &direction,
                                          const Precision stepMax) const {
  Vector3D<Precision> positionLocal =
      this->transformation_->Transform(position);
  Vector3D<Precision> directionLocal =
      this->transformation_->TransformDirection(direction);
  return fRootShape->DistFromInside(
           &positionLocal[0],
           &directionLocal[0],
           1,
           (stepMax == kInfinity) ? TGeoShape::Big() : stepMax
         ); 
}

VECGEOM_INLINE
Precision PlacedRootVolume::SafetyToOut(
    Vector3D<Precision> const &position) const {
  Vector3D<Precision> position_local =
      this->transformation_->Transform(position);
  return fRootShape->Safety(&position_local[0], true);
}

VECGEOM_INLINE
Precision PlacedRootVolume::SafetyToIn(
    Vector3D<Precision> const &position) const {
  Vector3D<Precision> position_local =
      this->transformation_->Transform(position);
  return fRootShape->Safety(&position_local[0], false);
}

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDROOTVOLUME_H_