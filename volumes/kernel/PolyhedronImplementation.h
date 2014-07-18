/// \file PolyhedronImplementation.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_

#include "backend/Backend.h"
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedPolyhedron.h"

namespace VECGEOM_NAMESPACE {

template <class PolyhedronType>
struct PolyhedronImplementation {

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::int_v FindPhiSegment(
      typename Backend::precision_v phi0,
      UnplacedPolyhedron const &polyhedron);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static Vector3D<typename Backend::precision_v> VectorFromSOA(
      SOA3D<Precision> const &soa,
      size_t index);

  template<typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void InsideSegment(
      UnplacedPolyhedron const &unplaced,
      UnplacedPolyhedron::PolyhedronSegment const &segment,
      Vector3D<Precision> const &point,
      Inside_t &inside,
      Precision &distance);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToSide(
    UnplacedPolyhedron::PolyhedronSegment const &segment,
    int sideIndex,
    Vector3D<Precision> const &point,
    typename Backend::precision_v &distance,
    typename Backend::precision_v &distanceNormal);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedPolyhedron const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedPolyhedron const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedPolyhedron const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ContainsKernel(
      Vector3D<Precision> const &polyhedronDimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void InsideKernel(
      Vector3D<Precision> const &polyhedronDimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      Vector3D<Precision> const &dimensions,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(
      Vector3D<Precision> const &dimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v & safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(
      Vector3D<Precision> const &dimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

}; // End struct PolyhedronImplementation

template <class PolyhedronType>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::int_v
PolyhedronImplementation<PolyhedronType>::FindPhiSegment(
    typename Backend::precision_v phi0,
    UnplacedPolyhedron const &polyhedron) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::int_v Int_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t phi = GenericKernels<Backend>::NormalizeAngle(
      phi0 - polyhedron.GetPhiStart());
  Int_t side = Int_t(phi / polyhedron.GetPhiDelta());
  if (PolyhedronType::phiTreatment) {
    Bool_t inPhi = side > polyhedron.GetSideCount();
    if (inPhi == Backend::kZero) return side;
    phi = GenericKernels<Backend>::NormalizeAngle(phi);
    Float_t start = polyhedron.GetPhiStart() - phi;
    Float_t end = phi - polyhedron.GetPhiEnd();
    MaskedAssign(inPhi && start < end, 0, &side);
    MaskedAssign(inPhi && start >= end, polyhedron.GetSideCount()-1, &side);
  }
  return side;
}

template <class PolyhedronType>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
Vector3D<typename Backend::precision_v>
PolyhedronImplementation<PolyhedronType>::VectorFromSOA(
    SOA3D<Precision> const &soa,
    size_t index) {
  Vector3D<typename Backend::precision_v> vector(
    Backend::Convert(soa.x(index)),
    Backend::Convert(soa.y(index)),
    Backend::Convert(soa.z(index))
  );
  return vector;
}

template <class PolyhedronType>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::InsideSegment(
    UnplacedPolyhedron const &unplaced,
    UnplacedPolyhedron::PolyhedronSegment const &segment,
    Vector3D<Precision> const &point,
    Inside_t &inside,
    Precision &distance) {

  int side = FindPhiSegment<kScalar>(point.Phi(), unplaced);

  Precision normal;
  DistanceToSide<kScalar>(segment, side, point, distance, normal);

  inside = EInside::kOutside;
  if (normal < 0.) inside = EInside::kInside;
  if ((Abs(normal) < kTolerance) && (distance < 2.*kTolerance)) {
    inside = EInside::kSurface;
  }
}

template <class PolyhedronType>
template <typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::DistanceToSide(
    UnplacedPolyhedron::PolyhedronSegment const &segment,
    int sideIndex,
    Vector3D<Precision> const &point,
    typename Backend::precision_v &distance,
    typename Backend::precision_v &distanceNormal) {
  
  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  // Manual SOA handling. There should be a better solution.

  Vector3D<Float_t> sideCenter =
      VectorFromSOA<Backend>(segment.center, sideIndex);
  Vector3D<Float_t> sideNormal =
      VectorFromSOA<Backend>(segment.normal, sideIndex);
  Vector3D<Float_t> sideSurfRZ =
      VectorFromSOA<Backend>(segment.surfRZ, sideIndex);
  Vector3D<Float_t> sideSurfPhi =
      VectorFromSOA<Backend>(segment.surfPhi, sideIndex);
  Vector3D<Float_t> sideEdgeNormal[2];
  Vector3D<Float_t> edgeNormal[2];
  Vector3D<Float_t> sideCorner[2][2];
  Vector3D<Float_t> sideCornerNormal[2][2];
  for (int i = 0; i < 2; ++i) {
    sideEdgeNormal[i] =
        VectorFromSOA<Backend>(segment.edgeNormal[i], sideIndex);
    edgeNormal[i] =
        VectorFromSOA<Backend>(segment.edge[i].normal, sideIndex);
    for (int j = 0; j < 2; ++j) {
      sideCorner[i][j] = VectorFromSOA<Backend>(segment.edge[i].corner[j],
                                                sideIndex);
      sideCornerNormal[i][j]
          = VectorFromSOA<Backend>(segment.edge[i].cornerNormal[j], sideIndex);
    }
  }

  Vector3D<Float_t> centerDiff = point - sideCenter;

  distanceNormal = sideNormal.Dot(centerDiff);

  //        A = above, B = below, I = inside
  //                                                   Phi
  //               |              |                     ^
  //           BA  |      IA      |  AA                 |
  //        ------[1]------------[3]-----               |
  //               |              |                     +----> RZ
  //           BA  |      II      |  AI
  //               |              |
  //        ------[0]------------[2]----
  //           BB  |      IB      |  AB
  //               |              |

  Float_t dotRZ = centerDiff.Dot(sideSurfRZ);
  Float_t dotPhi = centerDiff.Dot(sideSurfPhi);

  // Arguments to determine for final calculation
  Float_t distSquared;
  Vector3D<Float_t> corner, cornerNormal;

  // Intermediates
  Bool_t belowRZ, aboveRZ, insideRZ, belowPhiZ, abovePhiZ, insidePhiZ;
  Bool_t aa, ab, ai, ba, bb, bi, ia, ib; // Corresponding to graph above
  Float_t distZ, distZSquared, phiZLength;
  Float_t distPhi, distPhiSquared;
  Float_t temp;

  // R-Z conditions
  belowRZ = dotRZ < -segment.rZLength;
  aboveRZ = dotRZ > segment.rZLength;
  insideRZ = !(belowRZ || aboveRZ);

  // Calculate Phi-Z length
  temp = segment.rZLength * segment.phiLength[1];
  phiZLength = segment.phiLength[0] + dotRZ * segment.phiLength[1]; // Inside
  MaskedAssign(belowRZ, segment.phiLength[0] - temp, &phiZLength);
  MaskedAssign(aboveRZ, segment.phiLength[0] + temp, &phiZLength);

  // Phi-Z conditions
  belowPhiZ = dotPhi < -phiZLength;
  abovePhiZ = dotPhi > phiZLength;
  insidePhiZ = !(belowPhiZ || abovePhiZ);

  // Determine segment
  aa = aboveRZ && abovePhiZ;
  ab = aboveRZ && belowPhiZ;
  ai = aboveRZ && insidePhiZ;
  ba = belowRZ && abovePhiZ;
  bb = belowRZ && belowPhiZ;
  bi = belowRZ && insidePhiZ;
  ia = insideRZ && abovePhiZ;
  ib = insideRZ && belowPhiZ;
  
  // Distance for coordinates
  distZ = dotRZ + segment.rZLength;
  distPhi = dotPhi + phiZLength;
  MaskedAssign(aboveRZ, dotRZ - segment.rZLength, &distZ);
  MaskedAssign(abovePhiZ, dotPhi - phiZLength, &distPhi);
  distZSquared = distZ*distZ;
  distPhiSquared = distPhi*distPhi;

  // Compute distance to out
  distSquared = 0;
  MaskedAssign(aa || ab || ba || bb, distPhiSquared + distZSquared,
               &distSquared);
  MaskedAssign(ai || bi, distZSquared, &distSquared);
  temp = segment.rZPhiNormal*(dotPhi + phiZLength);
  MaskedAssign(ia || ib, temp*temp, &distSquared);

  // Determine corner
  MaskedAssign(aa || ia, sideCorner[1][1], &corner);
  MaskedAssign(ab || ai || ib, sideCorner[0][1], &corner);
  MaskedAssign(ba, sideCorner[1][0], &corner);
  MaskedAssign(bb || bi, sideCorner[0][0], &corner);

  // Determine corner normal
  MaskedAssign(aa, sideCornerNormal[1][1], &cornerNormal);
  MaskedAssign(ab, sideCornerNormal[0][1], &cornerNormal);
  MaskedAssign(bb, sideCornerNormal[0][0], &cornerNormal);
  MaskedAssign(ba, sideCornerNormal[1][0], &cornerNormal);
  MaskedAssign(ai, sideEdgeNormal[1], &cornerNormal);
  MaskedAssign(bi, sideEdgeNormal[0], &cornerNormal);
  MaskedAssign(ia, edgeNormal[1], &cornerNormal);
  MaskedAssign(ib, edgeNormal[0], &cornerNormal);

  // Return values
  distance = Sqrt(distanceNormal*distanceNormal + distSquared);
  distanceNormal = corner.Dot(cornerNormal);
}

template <class PolyhedronType>
template <typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::Inside(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::UnplacedContains(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::Contains(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<PolyhedronType>::DistanceToIn(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<PolyhedronType>::DistanceToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<PolyhedronType>::SafetyToIn(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<PolyhedronType>::SafetyToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<PolyhedronType>::ContainsKernel(
    Vector3D<Precision> const &polyhedronDimensions,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::bool_v &inside) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<PolyhedronType>::InsideKernel(
    Vector3D<Precision> const &polyhedronDimensions,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<PolyhedronType>::DistanceToInKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<PolyhedronType>::DistanceToOutKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<PolyhedronType>::SafetyToInKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v & safety) {
  Assert(0, "NYI");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<PolyhedronType>::SafetyToOutKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "NYI");
}

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_