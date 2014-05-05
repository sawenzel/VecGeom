/// @file KernelUtilities.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_KERNELUTILITIES_H_
#define VECGEOM_VOLUMES_KERNEL_KERNELUTILITIES_H_

#include "base/global.h"
#include "base/transformation3d.h"
#include "base/vector3d.h"

namespace VECGEOM_NAMESPACE {

template <class Backend>
struct KernelUtilities {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::int_v Int_t;
  typedef typename Backend::bool_v Bool_t;

  static void LocalPointInsideBox(Vector3D<Precision> const &boxDimensions,
                                  Vector3D<Float_t> const &point,
                                  Int_t &inside) {

    Bool_t done(false);
    inside = EInside::kOutside;
    Vector3D<Float_t> pointAbs = point.Abs();

    // Check if points are outside all surfaces
    Vector3D<Bool_t> insideOuterTolerance = pointAbs <= boxDimensions
                                                        + kHalfTolerance;
    Bool_t isOutside = !insideOuterTolerance[0] &&
                       !insideOuterTolerance[1] &&
                       !insideOuterTolerance[2];
    done |= isOutside;
    if (done == Backend::kTrue) return;

    // Check if points are inside all surfaces
    Vector3D<Bool_t> insideInnerTolerance = pointAbs <= boxDimensions
                                                        - kHalfTolerance;
    Bool_t isInside = insideInnerTolerance[0] &&
                      insideInnerTolerance[1] &&
                      insideInnerTolerance[2];
    MaskedAssign(isInside, EInside::kInside, &inside);
    done |= isInside;
    if (done == Backend::kTrue) return;

    // Check for remaining surface cases
    Vector3D<Bool_t> onSurface = !isInside && !isOutside;
    Bool_t isSurface = (onSurface[0] && !insideOuterTolerance[1]
                        && !insideOuterTolerance[2]) ||
                       (onSurface[1] && !insideOuterTolerance[0]
                        && !insideOuterTolerance[2]) ||
                       (onSurface[2] && !insideOuterTolerance[0]
                        && !insideOuterTolerance[1]);
    MaskedAssign(isSurface, EInside::kSurface, &inside);

  }

}; // End struct KernelUtilities

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_KERNELUTILITIES_H_