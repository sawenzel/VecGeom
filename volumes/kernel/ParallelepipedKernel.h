/// @file ParallelePipedKernel.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDKERNEL_H_
#define VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDKERNEL_H_

#include "base/global.h"

#include "base/transformation3d.h"
#include "volumes/kernel/KernelUtilities.h"
#include "volumes/UnplacedParallelepiped.h"

namespace VECGEOM_NAMESPACE {

template <class ParallelepipedSpecialization, TranslationCode transCode,
          RotationCode rotCode>
struct ParallelepipedImplementation {

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void Inside(UnplacedParallelepiped const &unplaced,
              Transformation3D const &transformation,
              Vector3D<typename Backend::precision_v> const &point,
              typename Backend::int_v &inside) {

    typedef typename Backend::precision_v Float_t;

    Vector3D<Float_t> pointLocal =
        transformation.template Transform<transCode, rotCode>(point);

    if (ParallelepipedSpecialization::HasTanThetaSinPhi) {
      Float_t shift = unplaced.GetTanThetaSinPhi()*pointLocal.z();
      pointLocal.x() -= shift;
      pointLocal.y() -= shift;
    }

    if (ParallelepipedSpecialization::HasAlpha) {
      pointLocal.x() -= unplaced.GetTanAlpha()*pointLocal.y();
    }

    // Run regular kernel for point inside a box
    LocalPointInsideBox(unplaced.GetDimensions(), pointLocal, inside);

  }

};

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_PARALLELEPIPEDKERNEL_H_