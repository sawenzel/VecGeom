/**
 * @file logical_volume.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_VOLUMES_LOGICALVOLUME_H_
#define VECGEOM_VOLUMES_LOGICALVOLUME_H_

#include "base/Global.h"
#include "base/Vector.h"
#include "volumes/UnplacedVolume.h"

#include <string>
#include <list>

class TGeoShape;
class VUSolid;

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class LogicalVolume; )
VECGEOM_DEVICE_FORWARD_DECLARE( class VPlacedVolume; )

inline namespace VECGEOM_IMPL_NAMESPACE {

typedef VPlacedVolume const* Daughter;

/**
 * @brief Class responsible for storing the unplaced volume, material and
 *        daughter volumes of a mother volume.
 */
class LogicalVolume {

private:
  // pointer to concrete unplaced volume/shape
  VUnplacedVolume const *fUnplacedVolume;

  int fId; // global id of logical volume object

  std::string *fLabel; // name of logical volume

  static int gIdCount; // static class counter

  // a pointer member to register arbitrary objects with logical volume;
  // included for the moment to model UserExtension like in TGeoVolume
  void *fUserExtensionPtr;

  // the container of daughter (placed) volumes which are placed inside this logical
  // Volume
  Vector<Daughter> *fDaughters;

  using CudaDaughter_t = cuda::VPlacedVolume const *;
  friend class CudaManager;

public:

#ifndef VECGEOM_NVCC

  // Standard constructor when constructing geometries. Will initiate an empty
  // daughter list which can be populated by placing daughters.
  // \sa PlaceDaughter()
  LogicalVolume(char const *const label, VUnplacedVolume const *const unplaced_vol);

  LogicalVolume(VUnplacedVolume const *const unplaced_vol) : LogicalVolume("", unplaced_vol) {}

  //
  // copy operator since we have pointer data members
  //
  LogicalVolume(LogicalVolume const &other);
  LogicalVolume *operator=(LogicalVolume const &other);

#else
  __device__
  LogicalVolume(VUnplacedVolume const *const unplaced_vol, Vector<Daughter> *GetDaughter)
      // Id for logical volumes is not needed on the device for CUDA
      : fUnplacedVolume(unplaced_vol),
        fId(-1),
        fLabel(NULL),
        fUserExtensionPtr(NULL),
        fDaughters(GetDaughter) {}
#endif

  ~LogicalVolume();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume const *GetUnplacedVolume() const { return fUnplacedVolume; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector<Daughter> const &GetDaughters() const { return *fDaughters; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector<Daughter> const *GetDaughtersp() const { return fDaughters; }

  VECGEOM_INLINE
  void *GetUserExtensionPtr() const { return fUserExtensionPtr; }

  int id() const { return fId; }

  std::string GetLabel() const { return *fLabel; }

  void SetLabel(char const *const label) {
    if(fLabel) delete fLabel;
    fLabel = new std::string(label);
  }

  VECGEOM_INLINE
  void SetUserExtensionPtr(void *userpointer) { fUserExtensionPtr = userpointer; }

  VECGEOM_CUDA_HEADER_BOTH
  void Print(const int indent = 0) const;

  VECGEOM_CUDA_HEADER_BOTH
  void PrintContent(const int depth = 0) const;

  VPlacedVolume *Place(char const *const label, Transformation3D const *const transformation) const;

  VPlacedVolume *Place(Transformation3D const *const transformation) const;

  VPlacedVolume *Place(char const *const label) const;

  VPlacedVolume *Place() const;

  VPlacedVolume const *PlaceDaughter(char const *const label, LogicalVolume const *const volume,
                                     Transformation3D const *const transformation);

  VPlacedVolume const *PlaceDaughter(LogicalVolume const *const volume,
                                     Transformation3D const *const transformation);

  void PlaceDaughter(VPlacedVolume const *const placed);

  friend std::ostream &operator<<(std::ostream &os, LogicalVolume const &vol);

#ifdef VECGEOM_CUDA_INTERFACE
  DevicePtr<cuda::LogicalVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const unplaced_vol,
                                           DevicePtr<cuda::Vector<CudaDaughter_t>> GetDaughter) const;
  DevicePtr<cuda::LogicalVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const unplaced_vol,
                                           DevicePtr<cuda::Vector<CudaDaughter_t>> GetDaughter,
                                           DevicePtr<cuda::LogicalVolume> const gpu_ptr) const;
#endif

}; // End class
} // End inline namespace
} // End global namespace

#endif // VECGEOM_VOLUMES_LOGICALVOLUME_H_
