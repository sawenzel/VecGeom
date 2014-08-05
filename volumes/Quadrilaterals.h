/// \file Quadrilaterals.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_QUADRILATERALS_H_
#define VECGEOM_VOLUMES_QUADRILATERALS_H_

#include "base/Global.h"

#include "base/SOA3D.h"
#include "base/Vector3D.h"

namespace VECGEOM_NAMESPACE {

/// \brief Convex, planar quadrilaterals.
class Quadrilaterals {

private:

  SOA3D<Precision> fNormal;
  Precision *fDistance;
  SOA3D<Precision> fSides[4];
  SOA3D<Precision> fCorners[4];

public:

#ifdef VECGEOM_STD_CXX11
  Quadrilaterals(int size);
#endif

  ~Quadrilaterals();

  VECGEOM_CUDA_HEADER_BOTH
  int size() const { return fNormal.size(); }

  /// \param corner0 First corner in counterclockwise order.
  /// \param corner1 Second corner in counterclockwise order.
  /// \param corner2 Third corner in counterclockwise order.
  /// \param corner3 Fourth corner in counterclockwise order.
  void Set(
      int index,
      Vector3D<Precision> const &corner0,
      Vector3D<Precision> const &corner1,
      Vector3D<Precision> const &corner2,
      Vector3D<Precision> const &corner3);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToIn(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction) const;

};

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToIn(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  int n = size();
  Float_t bestDistance = kInfinity;
  Float_t distance = AlignedAllocate<Float_t>(n);
  Bool_t inBounds = AlignedAllocate<Float_t>(n);
  // Compute intersection and pray for the loop being unrolled
  for (int i = 0; i < n; ++i) {
    distance[i] = -(fNormal[0][i]*point[0] + fNormal[1][i]*point[1] +
                    fNormal[2][i]*point[2] + fDistance[i])
                 / (fNormal[0][i]*direction[0] + fNormal[1][i]*direction[1] +
                    fNormal[2][i]*direction[2]);
    Vector3D<Float_t> intersection = point + direction*distance[i];
    Bool_t inside[4];
    for (int j = 0; j < 4; ++j) {
      Float_t dot = fSides[j][0][i] * (intersection[0] - fCorners[j][0][i]) +
                    fSides[j][1][i] * (intersection[1] - fCorners[j][1][i]) +
                    fSides[j][2][i] * (intersection[2] - fCorners[j][2][i]);
      inside[j] = dot >= 0;
    }
    inBounds[i] = inside[0] && inside[1] && inside[2] && inside[3];
  }
  // Aggregate result
  for (int i = 0; i < n; ++i) {
    if (inBounds[i] && distance[i] >= 0 && distance[i] < bestDistance) {
      bestDistance = distance;
    }
  }
  AlignedFree(distance);
  AlignedFree(inBounds);
  return bestDistance;
}

} // End global namespace

#endif // VECGEOM_VOLUMES_QUADRILATERALS_H_