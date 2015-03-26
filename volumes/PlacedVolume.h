/// \file placed_volume.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDVOLUME_H_

#include "base/Global.h"
#include "volumes/LogicalVolume.h"
#include "volumes/USolidsInterfaceHelper.h"
#include <string>

class G4VSolid;

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class VPlacedVolume; )
VECGEOM_DEVICE_DECLARE_CONV( VPlacedVolume )
#ifndef VECGEOM_NVCC
template <> struct kCudaType<const cxx::VPlacedVolume*> { using type_t = const cuda::VPlacedVolume*; };
#endif

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedBox;
template <typename T> class SOA3D;

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
      : logical_volume_(logical_vol), transformation_(transformation),
        bounding_box_(boundingbox), id_(id), label_(NULL) {}

#endif
  VPlacedVolume(VPlacedVolume const &);
  VPlacedVolume *operator=(VPlacedVolume const &);

public:
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~VPlacedVolume();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int id() const { return id_; }

  std::string const& GetLabel() const { return *label_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  PlacedBox const* bounding_box() const { return bounding_box_; }


  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  LogicalVolume const* GetLogicalVolume() const {
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
  Transformation3D const* GetTransformation() const {
    return transformation_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void SetLogicalVolume(LogicalVolume const *const logical_vol) {
    logical_volume_ = logical_vol;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void SetTransformation(Transformation3D const *const transform) {
    transformation_ = transform;
  }

  void set_label(char const * label) {
    //if(label != NULL){
        //std::cerr << label << std::endl;
        //std::cerr << *label_ << std::endl;
        //label_->assign(label);}
    //else{
    if(label_) delete label_;
    label_ = new std::string(label);
    //}
  }

  friend std::ostream& operator<<(std::ostream& os, VPlacedVolume const &vol);

  virtual int memory_size() const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print(const int indent = 0) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const =0;

  /// Recursively prints contained volumes to standard output.
  VECGEOM_CUDA_HEADER_BOTH
  void PrintContent(const int depth = 0) const;

  // Geometry functionality

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Contains(Vector3D<Precision> const &point) const =0;

  virtual void Contains(SOA3D<Precision> const &point,
                        bool *const output) const =0;

  // virtual void Contains(AOS3D<Precision> const &point,
  //                       bool *const output) const =0;

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

  // virtual void Inside(AOS3D<Precision> const &point,
  //                     Inside_t *const output) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &position,
                                 Vector3D<Precision> const &direction,
                                 const Precision step_max = kInfinity) const =0;

  virtual void DistanceToIn(SOA3D<Precision> const &position,
                            SOA3D<Precision> const &direction,
                            Precision const *const step_max,
                            Precision *const output) const =0;

  virtual void DistanceToInMinimize(SOA3D<Precision> const &position,
                                    SOA3D<Precision> const &direction,
                                    int daughterindex,
                                    Precision *const output,
                                    int *const nextnodeids
                                    ) const =0;

  // virtual void DistanceToIn(AOS3D<Precision> const &position,
  //                           AOS3D<Precision> const &direction,
  //                           Precision const *const step_max,
  //                           Precision *const output) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(
      Vector3D<Precision> const &position,
      Vector3D<Precision> const &direction,
      Precision const step_max = kInfinity) const =0;


  // a "placed" version of the distancetoout function; here
  // the point and direction are first of all transformed into the reference frame of the
  // callee. The normal DistanceToOut method does not do this
  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision PlacedDistanceToOut(
      Vector3D<Precision> const &position,
      Vector3D<Precision> const &direction,
      Precision const step_max = kInfinity) const = 0;

  virtual void DistanceToOut(SOA3D<Precision> const &position,
                             SOA3D<Precision> const &direction,
                             Precision const *const step_max,
                             Precision *const output) const =0;

  virtual void DistanceToOut(SOA3D<Precision> const &position,
                             SOA3D<Precision> const &direction,
                             Precision const *const step_max,
                             Precision *const output,
                             int *const nextnodeindex) const =0;

  // virtual void DistanceToOut(AOS3D<Precision> const &position,
  //                            AOS3D<Precision> const &direction,
  //                            Precision const *const step_max,
  //                            Precision *const output) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToIn(Vector3D<Precision> const &position) const =0;

  virtual void SafetyToIn(SOA3D<Precision> const &position,
                          Precision *const safeties) const =0;

  // virtual void SafetyToIn(AOS3D<Precision> const &position,
  //                         Precision *const safeties) const =0;

  virtual void SafetyToInMinimize(SOA3D<Precision> const &points,
                                  Precision *const safeties) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToOut(Vector3D<Precision> const &position) const =0;

  virtual void SafetyToOut(SOA3D<Precision> const &position,
                           Precision *const safeties) const =0;

  // virtual void SafetyToOut(AOS3D<Precision> const &position,
  //                          Precision *const safeties) const =0;

  virtual void SafetyToOutMinimize(SOA3D<Precision> const &points,
                                   Precision *const safeties) const =0;

  // returning the cubic volume of the shape satisfying the USolids interface
  // it is currently not a const function since some shapes might cache this value
  // if it is expensive to calculate
  virtual Precision Capacity() {
      assert(0 && "Capacity not implemented");
      return 0;
  }

  virtual void Extent(Vector3D<Precision> &min,
                      Vector3D<Precision> &max) const {
    assert(0 && "Extent not implemented for this shape type.");
  }

  virtual Precision SurfaceArea() {
    assert(0 && "SurfaceArea not implemented for this shape type.");
    return 0.0;
  }


public:

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const = 0;
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                                   DevicePtr<cuda::Transformation3D> const transform,
                                                   DevicePtr<cuda::VPlacedVolume> const gpu_ptr) const =0;
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(
      DevicePtr<cuda::LogicalVolume> const logical_volume,
      DevicePtr<cuda::Transformation3D> const transform) const =0;

  template <typename Derived>
  DevicePtr<cuda::VPlacedVolume> CopyToGpuImpl(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                               DevicePtr<cuda::Transformation3D> const transform,
                                               DevicePtr<cuda::VPlacedVolume> const in_gpu_ptr) const
  {
     DevicePtr<CudaType_t<Derived> > gpu_ptr(in_gpu_ptr);
     gpu_ptr.Construct(logical_volume, transform, nullptr, this->id());
     CudaAssertError();
     // Need to go via the void* because the regular c++ compilation
     // does not actually see the declaration for the cuda version
     // (and thus can not determine the inheritance).
     return DevicePtr<cuda::VPlacedVolume>((void*)gpu_ptr);
  }
  template <typename Derived>
  DevicePtr<cuda::VPlacedVolume> CopyToGpuImpl(
      DevicePtr<cuda::LogicalVolume> const logical_volume,
      DevicePtr<cuda::Transformation3D> const transform) const
  {
     DevicePtr<CudaType_t<Derived> > gpu_ptr;
     gpu_ptr.Allocate();
     return this->CopyToGpuImpl<Derived>(logical_volume,transform,
                                         DevicePtr<cuda::VPlacedVolume>((void*)gpu_ptr));
  }

#endif

#ifndef VECGEOM_NVCC
  virtual VPlacedVolume const* ConvertToUnspecialized() const =0;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const =0;
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const =0;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const* ConvertToGeant4() const =0;
#endif
#endif // VECGEOM_NVCC

};

} } // End global namespace

#ifdef VECGEOM_NVCC

#define VECGEOM_DEVICE_INST_PLACED_VOLUME( PlacedVol ) \
   namespace cxx { \
      template size_t DevicePtr<cuda::PlacedVol>::SizeOf(); \
      template void DevicePtr<cuda::PlacedVol>::Construct( \
         DevicePtr<cuda::LogicalVolume> const logical_volume, \
         DevicePtr<cuda::Transformation3D> const transform, \
         DevicePtr<cuda::PlacedBox> const boundingBox, \
         const int id) const; \
    }

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol, Extra )    \
   namespace cxx { \
      template size_t DevicePtr<cuda::PlacedVol, Extra>::SizeOf(); \
      template void DevicePtr<cuda::PlacedVol, Extra>::Construct( \
         DevicePtr<cuda::LogicalVolume> const logical_volume, \
         DevicePtr<cuda::Transformation3D> const transform, \
         DevicePtr<cuda::PlacedBox> const boundingBox, \
         const int id) const; \
    }

#ifdef VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT( PlacedVol, trans )   \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, rotation::kGeneric> )

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC( PlacedVol ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT(PlacedVol, translation::kGeneric)

#else // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT( PlacedVol, trans )   \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, rotation::kGeneric> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, rotation::kDiagonal> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, rotation::kIdentity> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x046> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x054> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x062> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x076> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x0a1> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x0ad> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x0dc> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x0e3> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x10a> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x11b> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x155> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x16a> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x18e> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<trans, 0x1b1> )

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC( PlacedVol ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT(PlacedVol, translation::kGeneric) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT(PlacedVol, translation::kIdentity)

#endif // VECGEOM_NO_SPECIALIZATION

#ifdef VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT( PlacedVol, radii )   \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<radii, Polyhedron::EPhiCutout::kGeneric> )

#define VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALLSPEC( PlacedVol ) \
   VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT(PlacedVol, Polyhedron::EInnerRadii::kGeneric)

#else // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT( PlacedVol, radii )   \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<radii, Polyhedron::EPhiCutout::kGeneric> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<radii, Polyhedron::EPhiCutout::kFalse> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<radii, Polyhedron::EPhiCutout::kTrue> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL( PlacedVol<radii, Polyhedron::EPhiCutout::kLarge> ) \

#define VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALLSPEC( PlacedVol ) \
   VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT(PlacedVol, Polyhedron::EInnerRadii::kGeneric) \
   VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT(PlacedVol, Polyhedron::EInnerRadii::kFalse) \
   VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT(PlacedVol, Polyhedron::EInnerRadii::kTrue)

#endif // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol, Extra, Type ) \
   namespace cxx { \
      template size_t DevicePtr<cuda::PlacedVol, Extra, cuda::Type>::SizeOf(); \
      template void DevicePtr<cuda::PlacedVol, Extra, cuda::Type>::Construct( \
         DevicePtr<cuda::LogicalVolume> const logical_volume, \
         DevicePtr<cuda::Transformation3D> const transform, \
         DevicePtr<cuda::PlacedBox> const boundingBox, \
         const int id) const; \
    }

#ifdef VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_3( PlacedVol, trans, Type ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, rotation::kGeneric, Type> )
#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( PlacedVol, Type )       \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_3(PlacedVol, translation::kGeneric, Type)

#else // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_3( PlacedVol, trans, Type ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, rotation::kGeneric, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, rotation::kDiagonal, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, rotation::kIdentity, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x046, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x054, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x062, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x076, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x0a1, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x0ad, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x0dc, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x0e3, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x10a, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x11b, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x155, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x16a, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x18e, Type> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3( PlacedVol<trans, 0x1b1, Type> )

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3( PlacedVol, Type )       \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_3(PlacedVol, translation::kGeneric, Type) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_3(PlacedVol, translation::kIdentity, Type)

#endif // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol, trans, rot ) \
   namespace cxx { \
      template size_t DevicePtr<cuda::PlacedVol, trans, rot>::SizeOf(); \
      template void DevicePtr<cuda::PlacedVol, trans, rot>::Construct( \
         DevicePtr<cuda::LogicalVolume> const logical_volume, \
         DevicePtr<cuda::Transformation3D> const transform, \
         DevicePtr<cuda::PlacedBox> const boundingBox, \
         const int id) const; \
    }

#ifdef VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_BOOLEAN( PlacedVol, Op, trans) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, rotation::kGeneric> )

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN( PlacedVol, Op )       \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_BOOLEAN(PlacedVol, Op, translation::kGeneric)

#else // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_BOOLEAN( PlacedVol, Op, trans) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, rotation::kGeneric> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, rotation::kDiagonal> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, rotation::kIdentity> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x046> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x054> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x062> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x076> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x0a1> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x0ad> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x0dc> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x0e3> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x10a> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x11b> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x155> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x16a> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x18e> ) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN( PlacedVol<Op, trans, 0x1b1> )

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN( PlacedVol, Op )       \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_BOOLEAN(PlacedVol, Op, translation::kGeneric) \
   VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_BOOLEAN(PlacedVol, Op, translation::kIdentity)

#endif // VECGEOM_NO_SPECIALIZATION


#endif

#endif // VECGEOM_VOLUMES_PLACEDVOLUME_H_
