/*
 * SpecializedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */


#ifndef VECGEOM_VOLUMES_SPECIALIZEDCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDCONE_H_

#include "base/Global.h"

#include "volumes/kernel/ConeImplementation.h"
#include "volumes/PlacedCone.h"
#include "volumes/ScalarShapeImplementationHelper.h"
#include "base/SOA3D.h"

#include <stdio.h>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(SpecializedCone, TranslationCode,transCodeT, RotationCode,rotCode,typename,ConeType)

inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT,
          typename ConeType>
class SpecializedCone : public ScalarShapeImplementationHelper<ConeImplementation<transCodeT, rotCodeT, ConeType> >
{

  typedef ScalarShapeImplementationHelper<ConeImplementation<
                                        transCodeT, rotCodeT, ConeType> > Helper;
  typedef ConeImplementation<transCodeT, rotCodeT, ConeType> Specialization;

public:

#ifndef VECGEOM_NVCC

  SpecializedCone(char const *const label,
                  LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, (PlacedBox const *const)nullptr) {}

  SpecializedCone(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation)
      : SpecializedCone("", logical_volume, transformation) {}

  SpecializedCone(char const *const label,
                   const Precision rmin1,
                   const Precision rmax1,
                   const Precision rmin2,
                   const Precision rmax2,
                   const Precision dz,
                   const Precision sphi=0,
                   const Precision dphi=kTwoPi)
        : SpecializedCone(label,
                new LogicalVolume(new UnplacedCone(rmin1, rmax1, rmin2, rmax2, dz, sphi, dphi )),
                &Transformation3D::kIdentity) {}

#else

  __device__
  SpecializedCone(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation,
                  PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  using Helper = SpecializedCone<transCodeT, rotCodeT, ConeType>;

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~SpecializedCone() {}

#ifdef VECGEOM_CUDA_INTERFACE

  virtual size_t DeviceSizeOf() const { return DevicePtr<CudaType_t<Helper> >::SizeOf(); }

  DevicePtr<cuda::VPlacedVolume> CopyToGpu(
    DevicePtr<cuda::LogicalVolume> const logical_volume,
    DevicePtr<cuda::Transformation3D> const transform,
    DevicePtr<cuda::VPlacedVolume> const in_gpu_ptr) const
 {
     DevicePtr<CudaType_t<Helper> > gpu_ptr(in_gpu_ptr);
     gpu_ptr.Construct(logical_volume, transform, nullptr, this->id());
     CudaAssertError();
     // Need to go via the void* because the regular c++ compilation
     // does not actually see the declaration for the cuda version
     // (and thus can not determine the inheritance).
     return DevicePtr<cuda::VPlacedVolume>((void*)gpu_ptr);
 }

 DevicePtr<cuda::VPlacedVolume> CopyToGpu(
    DevicePtr<cuda::LogicalVolume> const logical_volume,
    DevicePtr<cuda::Transformation3D> const transform) const
 {
     DevicePtr<CudaType_t<Helper> > gpu_ptr;
     gpu_ptr.Allocate();
     return CopyToGpu(logical_volume,transform,
                      DevicePtr<cuda::VPlacedVolume>((void*)gpu_ptr));
 }

#endif // VECGEOM_CUDA_INTERFACE

};

typedef SpecializedCone<translation::kGeneric, rotation::kGeneric, ConeTypes::UniversalCone>
    SimpleCone;
typedef SpecializedCone<translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>
    SimpleUnplacedCone;


template <TranslationCode transCodeT, RotationCode rotCodeT, typename ConeType>
void SpecializedCone<transCodeT, rotCodeT, ConeType>::PrintType() const {
  printf("SpecializedCone<%i, %i>", transCodeT, rotCodeT);
}

} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_

