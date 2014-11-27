/// \file SpecializedParallelepiped.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_

#include "base/Global.h"

#include "volumes/kernel/ParallelepipedImplementation.h"
#include "volumes/PlacedParallelepiped.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedParallelepiped
    : public ShapeImplementationHelper<PlacedParallelepiped,
                                       ParallelepipedImplementation<
                                           transCodeT, rotCodeT> > {

  typedef ShapeImplementationHelper<PlacedParallelepiped,
                                    ParallelepipedImplementation<
                                        transCodeT, rotCodeT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedParallelepiped(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedParallelepiped(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : SpecializedParallelepiped("", logical_volume, transformation) {}

  SpecializedParallelepiped(char const *const label,
                            Vector3D<Precision> const &dimensions,
                            const Precision alpha, const Precision theta,
                            const Precision phi)
      : SpecializedParallelepiped(
          label,
          new LogicalVolume(
            new UnplacedParallelepiped(dimensions, alpha, theta, phi)
          ),
          &Transformation3D::kIdentity
        ) {} 

  SpecializedParallelepiped(char const *const label,
                            const Precision x, const Precision y,
                            const Precision z, const Precision alpha,
                            const Precision theta, const Precision phi)
      : SpecializedParallelepiped(
          label,
          new LogicalVolume(
            new UnplacedParallelepiped(x, y, z, alpha, theta, phi)
          ),
          &Transformation3D::kIdentity
        ) {}

#else

  __device__
  SpecializedParallelepiped(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;
  

};

typedef SpecializedParallelepiped<translation::kGeneric, rotation::kGeneric>
    SimpleParallelepiped;

template <TranslationCode transCodeT, RotationCode rotCodeT>
void SpecializedParallelepiped<transCodeT, rotCodeT>::PrintType() const {
  printf("SpecializedParallelepiped<%i, %i>", transCodeT, rotCodeT);
}

} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
