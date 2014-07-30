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

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::int_v FindPhiSegment(
      typename Backend::precision_v phi0,
      UnplacedPolyhedron const &polyhedron);

  // template <class Backend>
  // VECGEOM_INLINE
  // VECGEOM_CUDA_HEADER_BOTH
  // static Vector3D<typename Backend::precision_v> VectorFromSOA(
  //     SOA3D<Precision> const &soa,
  //     size_t index);

  template<class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      UnplacedPolyhedron const &polyhedron,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static bool UnplacedContainsScalar(
    UnplacedPolyhedron const &unplaced,
    Vector3D<Precision> const &localPoint);

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static bool ContainsScalar(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<Precision> const &point,
    Vector3D<Precision> &localPoint);

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      UnplacedPolyhedron const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  // VECGEOM_INLINE
  // VECGEOM_CUDA_HEADER_BOTH
  // static Inside_t InsideScalar(
  //   UnplacedPolyhedron const &unplaced,
  //   Transformation3D const &transformation,
  //   Vector3D<Precision> const &point);

  // template <bool treatSurfaceT>
  // VECGEOM_INLINE
  // VECGEOM_CUDA_HEADER_BOTH
  // static Inside_t InsideScalarKernel(
  //   UnplacedPolyhedron const &unplaced,
  //   Vector3D<Precision> const &point);

  // template <bool treatSurfaceT>
  // VECGEOM_INLINE
  // VECGEOM_CUDA_HEADER_BOTH
  // static Inside_t InsideSegment(
  //     UnplacedPolyhedron const &unplaced,
  //     UnplacedPolyhedron::PolyhedronSegment const &segment,
  //     Vector3D<Precision> const &point,
  //     Precision &distance);

  // template <class Backend>
  // VECGEOM_INLINE
  // VECGEOM_CUDA_HEADER_BOTH
  // static void DistanceToSide(
  //   UnplacedPolyhedron::PolyhedronSegment const &segment,
  //   UnplacedPolyhedron::PolyhedronSide const &side,
  //   Vector3D<typename Backend::precision_v> const &point,
  //   typename Backend::precision_v &distance,
  //   typename Backend::precision_v &distanceNormal);

  // template <bool outgoingT, class Backend>
  // VECGEOM_INLINE
  // VECGEOM_CUDA_HEADER_BOTH
  // static typename Backend::bool_v DistanceToSegment(
  //     UnplacedPolyhedron::PolyhedronSegment const &segment,
  //     Vector3D<typename Backend::precision_v> const &point,
  //     Vector3D<typename Backend::precision_v> const &direction,
  //     typename Backend::precision_v &distance,
  //     typename Backend::precision_v &surfaceDistance);

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

}; // End struct PolyhedronImplementation

// /// \return Index to the phi segment in which a particle is located.
// template <class PolyhedronType>
// template <class Backend>
// VECGEOM_CUDA_HEADER_BOTH
// typename Backend::int_v
// PolyhedronImplementation<PolyhedronType>::FindPhiSegment(
//     typename Backend::precision_v phi0,
//     UnplacedPolyhedron const &polyhedron) {

//   typedef typename Backend::precision_v Float_t;
//   typedef typename Backend::int_v Int_t;
//   typedef typename Backend::bool_v Bool_t;

//   Float_t phi = GenericKernels<Backend>::NormalizeAngle(
//       phi0 - polyhedron.GetPhiStart());
//   Int_t side = Int_t(phi / polyhedron.GetPhiDelta());
//   if (PolyhedronType::phiTreatment) {
//     Bool_t inPhi = side > polyhedron.GetSideCount();
//     if (inPhi == Backend::kZero) return side;
//     phi = GenericKernels<Backend>::NormalizeAngle(phi);
//     Float_t start = polyhedron.GetPhiStart() - phi;
//     Float_t end = phi - polyhedron.GetPhiEnd();
//     MaskedAssign(inPhi && start < end, 0, &side);
//     MaskedAssign(inPhi && start >= end, polyhedron.GetSideCount()-1, &side);
//   }
//   return side;
// }

// template <class PolyhedronType>
// template <class Backend>
// VECGEOM_CUDA_HEADER_BOTH
// Vector3D<typename Backend::precision_v>
// PolyhedronImplementation<PolyhedronType>::VectorFromSOA(
//     SOA3D<Precision> const &soa,
//     size_t index) {
//   Vector3D<typename Backend::precision_v> vector(
//     Backend::Convert(soa.x(index)),
//     Backend::Convert(soa.y(index)),
//     Backend::Convert(soa.z(index))
//   );
//   return vector;
// }

// template <class PolyhedronType>
// Inside_t PolyhedronImplementation<PolyhedronType>::InsideScalar(
//     UnplacedPolyhedron const &unplaced,
//     Transformation3D const &transformation,
//     Vector3D<Precision> const &point) {

//   Vector3D<Precision> localPoint = transformation.Transform(point);
//   return InsideScalarKernel<true>(unplaced, localPoint);
// }

// template <class PolyhedronType>
// template <bool treatSurfaceT>
// Inside_t PolyhedronImplementation<PolyhedronType>::InsideScalarKernel(
//     UnplacedPolyhedron const &unplaced,
//     Vector3D<Precision> const &point) {

//   Inside_t output = EInside::kOutside;
//   Precision bestDistance = kInfinity;

//   Array<UnplacedPolyhedron::PolyhedronSegment> const &segments =
//       unplaced.GetSegments();
//   for (Array<UnplacedPolyhedron::PolyhedronSegment>::const_iterator s =
//        segments.cbegin(), sEnd = segments.cend(); s != sEnd; ++s) {
//     Inside_t insideResult;
//     Precision distanceResult;
//     insideResult = InsideSegment<treatSurfaceT>(
//       unplaced, *s, point, distanceResult
//     );
//     if (treatSurfaceT) {
//       if (insideResult == EInside::kSurface) return EInside::kSurface;
//     }
//     if (distanceResult < bestDistance) {
//       bestDistance = distanceResult;
//       output = insideResult;
//     }
//   }

//   return output;
// }

// template <class PolyhedronType>
// template <bool treatSurfaceT>
// VECGEOM_CUDA_HEADER_BOTH
// Inside_t PolyhedronImplementation<PolyhedronType>::InsideSegment(
//     UnplacedPolyhedron const &unplaced,
//     UnplacedPolyhedron::PolyhedronSegment const &segment,
//     Vector3D<Precision> const &point,
//     Precision &distance) {

//   int side = FindPhiSegment<kScalar>(point.Phi(), unplaced);

//   Precision normal;
//   DistanceToSide<kScalar>(
//     segment, segment.sides[side], point, distance, normal
//   );

//   if (treatSurfaceT) {
//     if (Abs(normal) < kTolerance && distance < 2.*kTolerance) {
//       return EInside::kSurface;
//     }
//   }
//   if (normal < 0.) return EInside::kInside;
//   return EInside::kOutside;
// }

// /// Kernel mostly suitable for sequential evaluation. Vectorization can only be
// /// if all particles check with the same side.
// template <class PolyhedronType>
// template <class Backend>
// VECGEOM_INLINE
// VECGEOM_CUDA_HEADER_BOTH
// void PolyhedronImplementation<PolyhedronType>::DistanceToSide(
//     UnplacedPolyhedron::PolyhedronSegment const &segment,
//     UnplacedPolyhedron::PolyhedronSide const &side,
//     Vector3D<typename Backend::precision_v> const &point,
//     typename Backend::precision_v &distance,
//     typename Backend::precision_v &distanceNormal) {

//   // This kernel is the product of several failed attempts at vectorization.
//   //
//   // The function can vectorize over sides in a segment, but for the Inside
//   // method, only one side has to be checked for a given particle, so there is
//   // no opportunity to vectorize.
//   //
//   // If one wanted to vectorize over segments, the data would have to be
//   // structured in an obscure way, which would clash with the vectorization of
//   // other polyhedron algorithms.
//   //
//   // Even for vectorizing over particles the function performs poorly, as each
//   // particle to be evaluated has an individual segment index. Since these will
//   // most likely not be adjacent in memory, they would have to be gathered.
//   //
//   // In the end, no vectorization is done, unless a vector of particles could be
//   // assumed to have the same segment index.

//   typedef typename Backend::precision_v Float_t;
//   typedef typename Backend::bool_v Bool_t;

//   // Manual SOA handling. There should be a better solution.

//   // Vector3D<Segment_t> sideCenter =
//   //     VectorFromSOA<Backend>(segment.center, sideIndex);
//   // Vector3D<Segment_t> sideNormal =
//   //     VectorFromSOA<Backend>(segment.normal, sideIndex);
//   // Vector3D<Segment_t> sideSurfRZ =
//   //     VectorFromSOA<Backend>(segment.surfRZ, sideIndex);
//   // Vector3D<Segment_t> sideSurfPhi =
//   //     VectorFromSOA<Backend>(segment.surfPhi, sideIndex);
//   // Vector3D<Segment_t> sideEdgeNormal[2];
//   // Vector3D<Segment_t> edgeNormal[2];
//   // Vector3D<Segment_t> sideCorner[2][2];
//   // Vector3D<Segment_t> side.corner[2].normal[2];
//   // for (int i = 0; i < 2; ++i) {
//   //   sideEdgeNormal[i] =
//   //       VectorFromSOA<Backend>(segment.edgeNormal[i], sideIndex);
//   //   edgeNormal[i] =
//   //       VectorFromSOA<Backend>(segment.edge[i].normal, sideIndex);
//   //   for (int j = 0; j < 2; ++j) {
//   //     sideCorner[i][j] = VectorFromSOA<Backend>(segment.edge[i].corner[j],
//   //                                               sideIndex);
//   //     side.corner[i].normal[j]
//   //         = VectorFromSOA<Backend>(segment.edge[i].cornerNormal[j], sideIndex);
//   //   }
//   // }

//   Vector3D<Float_t> centerDiff = point - side.center;

//   distanceNormal = side.normal.Dot(centerDiff);

//   //        A = above, B = below, I = inside
//   //                                                   Phi
//   //               |              |                     ^
//   //           BA  |      IA      |  AA                 |
//   //        ------[1]------------[3]-----               |
//   //               |              |                     +----> RZ
//   //           BA  |      II      |  AI
//   //               |              |
//   //        ------[0]------------[2]----
//   //           BB  |      IB      |  AB
//   //               |              |

//   Float_t dotRZ = centerDiff.Dot(side.surfRZ);
//   Float_t dotPhi = centerDiff.Dot(side.surfPhi);

//   // Arguments to determine for final calculation
//   Float_t distSquared;
//   Vector3D<Float_t> corner, cornerNormal;

//   // Intermediates
//   Bool_t belowRZ, aboveRZ, insideRZ, belowPhiZ, abovePhiZ, insidePhiZ;
//   Bool_t aa, ab, ai, ba, bb, bi, ia, ib; // Corresponding to graph above
//   Float_t distZ, distZSquared, phiZLength;
//   Float_t distPhi, distPhiSquared;
//   Float_t temp;

//   // R-Z conditions
//   belowRZ = dotRZ < -segment.rZLength;
//   aboveRZ = dotRZ > segment.rZLength;
//   insideRZ = !(belowRZ || aboveRZ);

//   // Calculate Phi-Z length
//   temp = segment.rZLength * segment.phiLength[1];
//   phiZLength = segment.phiLength[0] + dotRZ * segment.phiLength[1]; // Inside
//   MaskedAssign(belowRZ, segment.phiLength[0] - temp, &phiZLength);
//   MaskedAssign(aboveRZ, segment.phiLength[0] + temp, &phiZLength);

//   // Phi-Z conditions
//   belowPhiZ = dotPhi < -phiZLength;
//   abovePhiZ = dotPhi > phiZLength;
//   insidePhiZ = !(belowPhiZ || abovePhiZ);

//   // Determine segment
//   aa = aboveRZ && abovePhiZ;
//   ab = aboveRZ && belowPhiZ;
//   ai = aboveRZ && insidePhiZ;
//   ba = belowRZ && abovePhiZ;
//   bb = belowRZ && belowPhiZ;
//   bi = belowRZ && insidePhiZ;
//   ia = insideRZ && abovePhiZ;
//   ib = insideRZ && belowPhiZ;
  
//   // Distance for coordinates
//   distZ = dotRZ + segment.rZLength;
//   distPhi = dotPhi + phiZLength;
//   MaskedAssign(aboveRZ, dotRZ - segment.rZLength, &distZ);
//   MaskedAssign(abovePhiZ, dotPhi - phiZLength, &distPhi);
//   distZSquared = distZ*distZ;
//   distPhiSquared = distPhi*distPhi;

//   // Compute distance to out
//   distSquared = 0;
//   MaskedAssign(aa || ab || ba || bb, distPhiSquared + distZSquared,
//                &distSquared);
//   MaskedAssign(ai || bi, distZSquared, &distSquared);
//   temp = segment.rZPhiNormal*(dotPhi + phiZLength);
//   MaskedAssign(ia || ib, temp*temp, &distSquared);

//   // Determine corner
//   MaskedAssign(aa || ia, side.edges[1].corner[1], &corner);
//   MaskedAssign(ab || ai || ib, side.edges[0].corner[1], &corner);
//   MaskedAssign(ba, side.edges[1].corner[0], &corner);
//   MaskedAssign(bb || bi, side.edges[0].corner[0], &corner);

//   // Determine corner normal
//   MaskedAssign(aa, side.edges[1].cornerNormal[1], &cornerNormal);
//   MaskedAssign(ab, side.edges[0].cornerNormal[1], &cornerNormal);
//   MaskedAssign(bb, side.edges[0].cornerNormal[0], &cornerNormal);
//   MaskedAssign(ba, side.edges[1].cornerNormal[0], &cornerNormal);
//   MaskedAssign(ai, side.edges[1].normal, &cornerNormal);
//   MaskedAssign(bi, side.edges[0].normal, &cornerNormal);
//   MaskedAssign(ia, side.edgeNormal[1], &cornerNormal);
//   MaskedAssign(ib, side.edgeNormal[0], &cornerNormal);

//   // Return values
//   distance = Sqrt(distanceNormal*distanceNormal + distSquared);
//   distanceNormal = corner.Dot(cornerNormal);
// }

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::UnplacedContains(
    UnplacedPolyhedron const &polyhedron,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {
  Assert(0, "Not implemented.\n");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::Contains(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {
  Assert(0, "Not implemented.\n");
}

// template <class PolyhedronType>
// VECGEOM_CUDA_HEADER_BOTH
// bool PolyhedronImplementation<PolyhedronType>::UnplacedContainsScalar(
//   UnplacedPolyhedron const &unplaced,
//   Vector3D<Precision> const &localPoint) {
//   Inside_t inside = InsideScalarKernel<false>(unplaced, localPoint);
//   return (inside == EInside::kInside) ? true : false;
// }

// template <class PolyhedronType>
// VECGEOM_INLINE
// VECGEOM_CUDA_HEADER_BOTH
// bool PolyhedronImplementation<PolyhedronType>::ContainsScalar(
//     UnplacedPolyhedron const &unplaced,
//     Transformation3D const &transformation,
//     Vector3D<Precision> const &point,
//     Vector3D<Precision> &localPoint) {
//   localPoint = transformation.Transform(point);
//   return UnplacedContainsScalar(unplaced, localPoint);
// }

template <class PolyhedronType>
template <class Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::Inside(
    UnplacedPolyhedron const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {
  Assert(0, "Not implemented.\n");
}

// template <class PolyhedronType>
// template <class Backend>
// VECGEOM_CUDA_HEADER_BOTH
// void PolyhedronImplementation<PolyhedronType>::DistanceToIn(
//     UnplacedPolyhedron const &unplaced,
//     Transformation3D const &transformation,
//     Vector3D<typename Backend::precision_v> const &point,
//     Vector3D<typename Backend::precision_v> const &direction,
//     typename Backend::precision_v const &stepMax,
//     typename Backend::precision_v &distance) {

//   typedef typename Backend::precision_v Float_t;
//   typedef typename Backend::bool_v Bool_t;

//   Vector3D<Float_t> localPoint = transformation.Transform(point);
//   Vector3D<Float_t> localDirection =
//       transformation.TransformDirection(direction);

//   Array<UnplacedPolyhedron::PolyhedronSegment> const &segments =
//       unplaced.GetSegments();

//   Array<UnplacedPolyhedron::PolyhedronSegment>::const_iterator bestSegment =
//       NULL;
//   Float_t distFromSurface = kInfinity;
//   distance = kInfinity;

//   for (Array<UnplacedPolyhedron::PolyhedronSegment>::const_iterator s =
//        segments.cbegin(), sEnd = segments.cend(); s != sEnd; ++s) {
//     Float_t segmentDistance = kInfinity;
//     Float_t segmentSurface = kInfinity;
//     Bool_t hit = DistanceToSegment<false, Backend>(
//       *s, localPoint, localDirection, segmentDistance, segmentSurface
//     );
//     Bool_t better = hit && segmentDistance < distance;
//     MaskedAssign(better, segmentDistance, &distance);
//     MaskedAssign(better, segmentSurface, &distFromSurface);
//     MaskedAssign(better, s, &bestSegment);
//     MaskedAssign(segmentSurface <= 0., 0., &distance);
//   }
// }

namespace {

template <bool outgoing>
struct OutgoingTraits;

template <>
struct OutgoingTraits<true> {
  static VECGEOM_CONSTEXPR Precision sign = 1;
  static VECGEOM_CONSTEXPR Precision invSign = -1;
};

template <>
struct OutgoingTraits<false> {
  static VECGEOM_CONSTEXPR Precision sign = -1;
  static VECGEOM_CONSTEXPR Precision invSign = 1;
};

} // End anonymous namespace

// template <class PolyhedronType>
// template <bool outgoingT, class Backend>
// VECGEOM_CUDA_HEADER_BOTH
// typename Backend::bool_v
// PolyhedronImplementation<PolyhedronType>::DistanceToSegment(
//     UnplacedPolyhedron::PolyhedronSegment const &segment,
//     Vector3D<typename Backend::precision_v> const &point,
//     Vector3D<typename Backend::precision_v> const &direction,
//     typename Backend::precision_v &distance,
//     typename Backend::precision_v &surfaceDistance) {

//   typedef typename Backend::precision_v Float_t;
//   typedef typename Backend::bool_v Bool_t;

//   Bool_t hit = Backend::kFalse;

//   Float_t dotProduct;
//   Vector3D<Float_t> qA, qB, qC, qD, surfPhi, surfRZ, diff;

//   for (Array<UnplacedPolyhedron::PolyhedronSide>::const_iterator s =
//        segment.sides.cbegin(), sEnd = segment.sides.cend(); s != sEnd; ++s) {

//     Bool_t wrongSegment = Backend::kTrue;

//     dotProduct = OutgoingTraits<outgoingT>::sign * direction.Dot(s->normal);
//     wrongSegment |= dotProduct <= 0;
//     if (wrongSegment == Backend::kTrue) continue;

//     Vector3D<Float_t> delta = point - s->center;
//     surfaceDistance = OutgoingTraits<outgoingT>::invSign * delta.Dot(s->normal);
//     wrongSegment |= surfaceDistance < -kHalfTolerance;
//     if (wrongSegment == Backend::kTrue) continue;

//     Vector3D<Float_t> q = point + direction;

//     qC = q - s->edges[1].corner[0];
//     qD = q - s->edges[1].corner[1];
//     wrongSegment |= OutgoingTraits<outgoingT>::sign
//                     * qC.Cross(qD).Dot(direction) < 0;
//     if (wrongSegment == Backend::kTrue) continue;

//     qA = q - s->edges[0].corner[0];
//     qB = q - s->edges[0].corner[1];
//     wrongSegment |= OutgoingTraits<outgoingT>::sign
//                     * qA.Cross(qB).Dot(direction) > 0;
//     if (wrongSegment == Backend::kTrue) continue;

//     // Any remaining particlers will have found the only possible side
//     MaskedAssign(!wrongSegment, s->surfPhi, &surfPhi);
//     MaskedAssign(!wrongSegment, s->surfRZ, &surfRZ);
//     MaskedAssign(!wrongSegment, delta, &diff);
//     // MaskedAssign(!wrongSegment, s->normal, &normal);
//     hit |= !wrongSegment;
//     if (hit == Backend::kTrue) break;
//   }

//   // See if there is anything to treat
//   if (hit == Backend::kFalse) return hit;

//   hit &= (segment.start[0] <= kTolerance) ||
//          (OutgoingTraits<outgoingT>::sign * qA.Cross(qC).Dot(direction) >= 0.);
//   hit &= segment.end[0] <= kTolerance ||
//          OutgoingTraits<outgoingT>::sign * qB.Cross(qD).Dot(direction) <= 0.;
//   if (hit == Backend::kFalse) return hit;

//   Float_t rZ = Abs(diff.Dot(surfRZ));
//   Float_t phi = Abs(diff.Dot(surfPhi));
//   hit &= surfaceDistance >= 0 || rZ <= segment.rZLength + kHalfTolerance;
//   hit &= phi <= segment.phiLength[0] + segment.phiLength[1]*rZ + kHalfTolerance;

//   distance = surfaceDistance / dotProduct;
//   return hit;
// }

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void PolyhedronImplementation<PolyhedronType>::DistanceToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {
  Assert(0, "DistanceToOut not implemented.\n");
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
  Assert(0, "SafetyToIn not implemented.\n");
}

template <class PolyhedronType>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void PolyhedronImplementation<PolyhedronType>::SafetyToOut(
    UnplacedPolyhedron const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {
  Assert(0, "SafetyToOut not implemented.\n");
}

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_POLYHEDRONIMPLEMENTATION_H_