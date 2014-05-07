/// @file BoxImplementation.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_

#include "backend/backend.h"
#include "base/vector3d.h"
#include "volumes/unplaced_box.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct BoxImplementation {

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedInsideKernel(
      Vector3D<Precision> const &boxDimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::int_v &inside) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

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
    Vector3D<Bool_t> onSurface = insideOuterTolerance && !insideInnerTolerance;
    Bool_t isSurface = (onSurface[0] && insideOuterTolerance[1]
                        && insideOuterTolerance[2]) ||
                       (onSurface[1] && insideOuterTolerance[0]
                        && insideOuterTolerance[2]) ||
                       (onSurface[2] && insideOuterTolerance[0]
                        && insideOuterTolerance[1]);
    MaskedAssign(isSurface, EInside::kSurface, &inside);

  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedDistanceToInKernel(
      Vector3D<Precision> const &dimensions,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

    Vector3D<Float_t> safety;
    Bool_t done = Backend::kFalse;

    safety[0] = Abs(point[0]) - dimensions[0];
    safety[1] = Abs(point[1]) - dimensions[1];
    safety[2] = Abs(point[2]) - dimensions[2];

    done |= (safety[0] >= stepMax ||
             safety[1] >= stepMax ||
             safety[2] >= stepMax);
    if (done == true) return;

    Float_t next, coord1, coord2;
    Bool_t hit;

    // x
    next = safety[0] / Abs(direction[0] + kTiny);
    coord1 = point[1] + next * direction[1];
    coord2 = point[2] + next * direction[2];
    hit = safety[0] >= 0 &&
          point[0] * direction[0] < 0 &&
          Abs(coord1) <= dimensions[1] &&
          Abs(coord2) <= dimensions[2];
    MaskedAssign(!done && hit, next, &distance);
    done |= hit;
    if (done == true) return;

    // y
    next = safety[1] / Abs(direction[1] + kTiny);
    coord1 = point[0] + next * direction[0];
    coord2 = point[2] + next * direction[2];
    hit = safety[1] >= 0 &&
          point[1] * direction[1] < 0 &&
          Abs(coord1) <= dimensions[0] &&
          Abs(coord2) <= dimensions[2];
    MaskedAssign(!done && hit, next, &distance);
    done |= hit;
    if (done == true) return;

    // z
    next = safety[2] / Abs(direction[2] + kTiny);
    coord1 = point[0] + next * direction[0];
    coord2 = point[1] + next * direction[1];
    hit = safety[2] >= 0 &&
          point[2] * direction[2] < 0 &&
          Abs(coord1) <= dimensions[0] &&
          Abs(coord2) <= dimensions[1];
    MaskedAssign(!done && hit, next, &distance);

  }

}; // End struct BoxImplementation

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_