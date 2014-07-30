/// \file Rectangles.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_RECTANGLES_H_
#define VECGEOM_VOLUMES_RECTANGLES_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/SOA3D.h"
#include "base/Vector3D.h"

namespace VECGEOM_NAMESPACE {

namespace {
template <bool treatSurfaceT> struct SurfaceTraits;
template <> struct SurfaceTraits<true>  { typedef Inside_t Surface_t; };
template <> struct SurfaceTraits<false> { typedef bool Surface_t; };
}

/// Stores a number of rectangles in SOA form to allow vectorized operations
class Rectangles : public AlignedBase {

private:

  int fSize;
  SOA3D<Precision> fNormal;
  SOA3D<Precision> fCenter;
  Precision *fP, *fXDim, *fYDim;

public:

  inline Rectangles(int size);

  inline ~Rectangles();

  inline void Set(
      int index,
      Vector3D<Precision> const &normal,
      Vector3D<Precision> const &center,
      Vector2D<Precision> const &dimensions);

  template <bool treatSurfaceT>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename SurfaceTraits<treatSurfaceT>::Surface_t Inside(
      Vector3D<Precision> const &point,
      int begin,
      int end) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Inside_t Inside(
      Vector3D<Precision> const &point,
      int begin,
      int end) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Inside_t Inside(Vector3D<Precision> const &point) const;

};

Rectangles::Rectangles(int size)
  : fSize(size), fNormal(size), fCenter(size) {
  fXDim = AlignedAllocate<Precision>(size);
  fYDim = AlignedAllocate<Precision>(size);
}

Rectangles::~Rectangles() {
  AlignedFree(fXDim);
  AlignedFree(fYDim);
}

void Rectangles::Set(
      int index,
      Vector3D<Precision> const &normal,
      Vector3D<Precision> const &center,
      Vector2D<Precision> const &dimensions) {
  assert(index < fSize);

  // Normalize the normal and distance
  Precision inverseLength = 1. / normal.Length();
  Vector3D<Precision> unitNormal = inverseLength * normal;
  Precision d = -normal.Dot(center);
  fP[index] = inverseLength * d;

  // Store in SOA format
  fNormal.set(index, unitNormal);
  fCenter.set(index, center);
  fXDim[index] = dimensions[0];
  fYDim[index] = dimensions[1];
}

} // End global namespace

#endif // VECGEOM_VOLUMES_RECTANGLES_H_