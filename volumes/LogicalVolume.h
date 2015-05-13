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

  VUnplacedVolume const *unplaced_volume_;

  using CudaDaughter_t = cuda::VPlacedVolume const*;

  int id_;
  std::string *label_;

  static int g_id_count;

  /** a pointer member to register arbitrary objects with logical volume;
        included for the moment to model UserExtension like in TGeoVolume
  */
  void * user_extension_;

  Vector<Daughter> *daughters_;


  friend class CudaManager;

public:

#ifndef VECGEOM_NVCC

  /**
   * Standard constructor when constructing geometries. Will initiate an empty
   * daughter list which can be populated by placing daughters.
   * \sa PlaceDaughter()
   */
  LogicalVolume(char const *const label,
                VUnplacedVolume const *const unplaced_vol);

  LogicalVolume(VUnplacedVolume const *const unplaced_vol)
      : LogicalVolume("", unplaced_vol) {}

  /** 
   * copy operator since we have pointer data members
   */
  LogicalVolume( LogicalVolume const & other );    
  LogicalVolume * operator=( LogicalVolume const & other );

#else

  __device__
  LogicalVolume(VUnplacedVolume const *const unplaced_vol,
                Vector<Daughter> *daughters)
      // Id for logical volumes is not needed on the device for CUDA
      : unplaced_volume_(unplaced_vol), id_(-1), label_(NULL),
        user_extension_(NULL), daughters_(daughters) {}

#endif

  ~LogicalVolume();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume const* unplaced_volume() const { return unplaced_volume_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector<Daughter> const& daughters() const { return *daughters_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector<Daughter> const * daughtersp() const { return daughters_; }

  VECGEOM_INLINE
  void * getUserExtensionPtr( ) const {  return user_extension_;  }

  int id() const { return id_; }

  std::string GetLabel() const { return *label_; }

  void set_label(char const *const label) {
    if( label_ )
      *label_ = label;
    else
      if(label_) delete label_;
      label_=new std::string(label);
  }

  VECGEOM_INLINE
  void setUserExtensionPtr( void * userpointer ) { user_extension_ = userpointer; }

  VECGEOM_CUDA_HEADER_BOTH
  void Print(const int indent = 0) const;

  VECGEOM_CUDA_HEADER_BOTH
  void PrintContent(const int depth = 0) const;

  VPlacedVolume* Place(char const *const label,
                       Transformation3D const *const transformation) const;

  VPlacedVolume* Place(Transformation3D const *const transformation) const;

  VPlacedVolume* Place(char const *const label) const;

  VPlacedVolume* Place() const;

  VPlacedVolume const* PlaceDaughter(char const *const label,
                     LogicalVolume const *const volume,
                     Transformation3D const *const transformation);

  VPlacedVolume const* PlaceDaughter(LogicalVolume const *const volume,
                     Transformation3D const *const transformation);

  void PlaceDaughter(VPlacedVolume const *const placed);

  VECGEOM_CUDA_HEADER_BOTH
  int CountVolumes() const;

  friend std::ostream& operator<<(std::ostream& os, LogicalVolume const &vol);

#ifdef VECGEOM_CUDA_INTERFACE
  DevicePtr<cuda::LogicalVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const unplaced_vol,
                                           DevicePtr<cuda::Vector<CudaDaughter_t>> daughters) const;
  DevicePtr<cuda::LogicalVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const unplaced_vol,
                                           DevicePtr<cuda::Vector<CudaDaughter_t>> daughters,
                                           DevicePtr<cuda::LogicalVolume> const gpu_ptr) const;
#endif

};

} } // End global namespace

#endif // VECGEOM_VOLUMES_LOGICALVOLUME_H_
