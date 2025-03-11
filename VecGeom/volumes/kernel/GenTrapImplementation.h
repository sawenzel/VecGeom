/*
 * GenTrapImplementation.h
 *
 *  Created on: Aug 2, 2014
 *      Author: swenzel
 *   Review/completion: Nov 4, 2015
 *      Author: mgheata
 */

#ifndef VECGEOM_VOLUMES_KERNEL_GENTRAPIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_GENTRAPIMPLEMENTATION_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"
#include "VecGeom/volumes/UnplacedGenTrap.h"

#include <iostream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct GenTrapImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, GenTrapImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedGenTrap;
class UnplacedGenTrap;

template <typename T>
struct GenTrapStruct;

struct GenTrapImplementation {

  using Vertex_t         = Vector3D<Precision>;
  using PlacedShape_t    = PlacedGenTrap;
  using UnplacedStruct_t = GenTrapStruct<Precision>;
  using UnplacedVolume_t = UnplacedGenTrap;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside);

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void ContainsGeneric(UnplacedStruct_t const &unplaced,
                                                                           Vector3D<Real_v> const &point,
                                                                           Bool_v &inside);

  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside);

  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void InsideGeneric(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Inside_t &inside);

  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, vecCore::Mask_v<Real_v> &completelyinside,
      vecCore::Mask_v<Real_v> &completelyoutside);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToInGeneric(UnplacedStruct_t const &unplaced,
                                                                               Vector3D<Real_v> const &point,
                                                                               Vector3D<Real_v> const &direction,
                                                                               Real_v const &stepMax, Real_v &distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &unplaced,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOutGeneric(UnplacedStruct_t const &unplaced,
                                                                                Vector3D<Real_v> const &point,
                                                                                Vector3D<Real_v> const &direction,
                                                                                Real_v const &stepMax,
                                                                                Real_v &distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &unplaced,
                                                                      Vector3D<Real_v> const &point, Real_v &safety);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToInGeneric(UnplacedStruct_t const &unplaced,
                                                                             Vector3D<Real_v> const &point,
                                                                             Real_v &safety);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOutGeneric(UnplacedStruct_t const &unplaced,
                                                                              Vector3D<Real_v> const &point,
                                                                              Real_v &safety);

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void NormalKernel(UnplacedStruct_t const &unplaced,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> &normal, Bool_v &valid);

  template <class Real_v>
  VECCORE_ATT_HOST_DEVICE static void GetClosestEdge(Vector3D<Real_v> const &point, Real_v vertexX[4],
                                                     Real_v vertexY[4], Real_v &iseg, Real_v &fraction);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static vecCore::Mask_v<Real_v> IsInTopOrBottomPolygon(
      UnplacedStruct_t const &unplaced, Real_v const &pointx, Real_v const &pointy, vecCore::Mask_v<Real_v> top);
}; // End struct GenTrapImplementation

//********************************
//**** implementations start here
//********************************/

template <typename Real_v>
struct FillPlaneDataHelper {
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static void FillPlaneData(GenTrapStruct<Precision> const &unplaced, Real_v &cornerx, Real_v &cornery, Real_v &deltax,
                            Real_v &deltay, vecCore::Mask_v<Real_v> const &top, int edgeindex)
  {

    // no vectorized data lookup for SIMD
    // need to fill the SIMD types individually

    for (size_t i = 0; i < vecCore::VectorSize<Real_v>(); ++i) {
      int index  = edgeindex + top[i] * 4;
      deltax[i]  = unplaced.fDeltaX[index];
      deltay[i]  = unplaced.fDeltaY[index];
      cornerx[i] = unplaced.fVerticesX[index];
      cornery[i] = unplaced.fVerticesY[index];
    }
  }
};

//______________________________________________________________________________
/** @brief A partial template specialization for nonSIMD cases (scalar, cuda, ... ) */
template <>
struct FillPlaneDataHelper<Precision> {
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static void FillPlaneData(GenTrapStruct<Precision> const &unplaced, Precision &cornerx, Precision &cornery,
                            Precision &deltax, Precision &deltay, bool const &top, int edgeindex)
  {
    int index = edgeindex + top * 4;
    deltax    = unplaced.fDeltaX[index];
    deltay    = unplaced.fDeltaY[index];
    cornerx   = unplaced.fVerticesX[index];
    cornery   = unplaced.fVerticesY[index];
  }
};

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE vecCore::Mask_v<Real_v> GenTrapImplementation::IsInTopOrBottomPolygon(
    UnplacedStruct_t const &unplaced, Real_v const &pointx, Real_v const &pointy, vecCore::Mask_v<Real_v> top)
{
  // optimized "inside" check for top or bottom z-surfaces
  // this is a bit tricky if different tracks check different planes
  // ( for example in case of Backend = Vc when top is mixed )
  // ( this is because vector data lookup is tricky )

  // stripped down version of the Contains kernel ( not yet shared with that kernel )
  // std::cerr << "IsInTopOrBottom: pointx: " << pointx << "  pointy: " << pointy << "  top: " << top << "\n";

  using Bool_v             = vecCore::Mask_v<Real_v>;
  Bool_v completelyoutside = Bool_v(false);
  Bool_v degenerate        = Bool_v(true);
  for (int i = 0; i < 4; ++i) {
    Real_v deltaX;
    Real_v deltaY;
    Real_v cornerX;
    Real_v cornerY;

    // thats the only place where scalar and vector code diverge
    // IsSIMD misses...replaced with early_returns
    FillPlaneDataHelper<Real_v>::FillPlaneData(unplaced, cornerX, cornerY, deltaX, deltaY, top, i);

    // std::cerr << i << " CORNERS " << cornerX << " " << cornerY << " " << deltaX << " " << deltaY << "\n";

    Real_v cross = (pointx - cornerX) * deltaY;
    cross -= (pointy - cornerY) * deltaX;
    degenerate =
        degenerate && (deltaX < Real_v(MakePlusTolerant<true>(0.))) && (deltaY < Real_v(MakePlusTolerant<true>(0.)));
    completelyoutside = completelyoutside ||
                        (cross < Real_v(MakeMinusTolerantCrossProduct<true, Real_v>(0., Abs(deltaX) + Abs(deltaY))));
    // if (vecCore::EarlyReturnAllowed()) {
    if (vecCore::MaskFull(completelyoutside)) {
      return Bool_v(false);
    }
    // }
  }
  completelyoutside = completelyoutside || degenerate;
  return (!completelyoutside);
}

//______________________________________________________________________________
template <typename Real_v, bool ForInside>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::GenericKernelForContainsAndInside(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, vecCore::Mask_v<Real_v> &completelyinside,
    vecCore::Mask_v<Real_v> &completelyoutside)
{

  using Bool_v                        = vecCore::Mask_v<Real_v>;
  VECGEOM_CONST Precision tolerancesq = 10000. * kTolerance * kTolerance;
  // Local point has to be translated in the bbox local frame.
  Vector3D<Real_v> halfsize(unplaced.fBBdimensions[0], unplaced.fBBdimensions[1], unplaced.fBBdimensions[2]);
  BoxImplementation::GenericKernelForContainsAndInside<Real_v, true>(halfsize, point - unplaced.fBBorigin,
                                                                     completelyinside, completelyoutside);
  //  if (vecCore::EarlyReturnAllowed()) {
  if (vecCore::MaskFull(completelyoutside)) {
    return;
  }
  //  }

  // analyse z
  Real_v cf = unplaced.fHalfInverseDz * (unplaced.fDz - point.z());
  // analyse if x-y coordinates of point are within polygon at z-height

  //  loop over edges connecting points i with i+4
  Real_v vertexX[4];
  Real_v vertexY[4];
  // vectorizes for scalar backend
  for (int i = 0; i < 4; i++) {
    // calculate x-y positions of vertex i at this z-height
    vertexX[i] = unplaced.fVerticesX[i + 4] + cf * unplaced.fConnectingComponentsX[i];
    vertexY[i] = unplaced.fVerticesY[i + 4] + cf * unplaced.fConnectingComponentsY[i];
  }

  for (int i = 0; i < 4; i++) {
    // this is based on the following idea:
    // we decided for each edge whether the point is above or below the
    // 2d line defined by that edge
    // In fact, this calculation is part of the calculation of the distance
    // of point to that line which is a cross product. In this case it is
    // an embedded cross product of 2D vectors in 3D. The resulting vector always points
    // in z-direction whose z-magnitude is directly related to the distance.
    // see, e.g.,  http://geomalgorithms.com/a02-_lines.html
    if (unplaced.fDegenerated[i]) continue;
    int j         = (i + 1) % 4;
    Real_v DeltaX = vertexX[j] - vertexX[i];
    Real_v DeltaY = vertexY[j] - vertexY[i];
    Real_v cross  = (point.x() - vertexX[i]) * DeltaY - (point.y() - vertexY[i]) * DeltaX;
    if (ForInside) {
      Bool_v onsurf     = (cross * cross < tolerancesq * (DeltaX * DeltaX + DeltaY * DeltaY));
      completelyoutside = completelyoutside || (((cross < Real_v(MakeMinusTolerant<true>(0.))) && (!onsurf)));
      completelyinside  = completelyinside && (cross > Real_v(MakePlusTolerant<true>(0.))) && (!onsurf);
    } else {
      completelyoutside = completelyoutside || (cross < Real_v(MakeMinusTolerant<true>(0.)));
    }

    //    if (vecCore::EarlyReturnAllowed()) {
    if (vecCore::MaskFull(completelyoutside)) {
      return;
    }
    //    }
  }
}

//______________________________________________________________________________
template <typename Real_v, typename Bool_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::ContainsGeneric(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Bool_v &inside)
{
  // Generic implementation for contains
  Bool_v unused(false);
  Bool_v outside(false);
  GenericKernelForContainsAndInside<Real_v, false>(unplaced, point, unused, outside);
  inside = !outside;
}

//______________________________________________________________________________
template <typename Real_v, typename Bool_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::Contains(UnplacedStruct_t const &unplaced,
                                                                                  Vector3D<Real_v> const &point,
                                                                                  Bool_v &inside)
{
  GenTrapImplementation::ContainsGeneric<Real_v, Bool_v>(unplaced, point, inside);
}

//______________________________________________________________________________
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::Contains<Precision, bool>(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, bool &inside)
{
// Scalar specialization for Contains function
#ifndef VECCORE_CUDA
  // The tessellated section helper is built only for the planar cases
  if (unplaced.fTslHelper) {
    // Check Z range
    inside = false;
    if (vecCore::math::Abs(point.z()) > unplaced.fDz) return;
    inside = unplaced.fTslHelper->Contains(point);
    return;
  }
#endif
  GenTrapImplementation::ContainsGeneric<Precision, bool>(unplaced, point, inside);
}

//______________________________________________________________________________
template <typename Real_v, typename Inside_t>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::InsideGeneric(UnplacedStruct_t const &unplaced,
                                                                                       Vector3D<Real_v> const &point,
                                                                                       Inside_t &inside)
{

  using Bool_v       = vecCore::Mask_v<Real_v>;
  using InsideBool_v = vecCore::Mask_v<Inside_t>;
  Bool_v completelyinside;
  Bool_v completelyoutside;
  GenericKernelForContainsAndInside<Real_v, true>(unplaced, point, completelyinside, completelyoutside);

  inside = Inside_t(EInside::kSurface);
  vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
  vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
}

//______________________________________________________________________________
template <typename Real_v, typename Inside_t>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::Inside(UnplacedStruct_t const &unplaced,
                                                                                Vector3D<Real_v> const &point,
                                                                                Inside_t &inside)
{
  // Generic dispatcher to Inside
  GenTrapImplementation::InsideGeneric<Real_v, Inside_t>(unplaced, point, inside);
}

//______________________________________________________________________________
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::Inside<Precision, Inside_t>(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, Inside_t &inside)
{
// Scalar specialization for Inside function
#ifndef VECCORE_CUDA
  if (unplaced.fTslHelper) {
    inside         = EInside::kOutside;
    Precision safZ = vecCore::math::Abs(point.z()) - unplaced.fDz;
    if (safZ > kTolerance) return;
    bool insideZ = safZ < -kTolerance;
    inside       = unplaced.fTslHelper->Inside(point);
    if (insideZ || inside == EInside::kOutside) return;
    inside = EInside::kSurface;
    return;
  }
#endif
  GenTrapImplementation::InsideGeneric<Precision, Inside_t>(unplaced, point, inside);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::DistanceToInGeneric(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction,
    Real_v const &stepMax, Real_v &distance)
{

  using Bool_v = vecCore::Mask_v<Real_v>;

  // do a quick boundary box check (Arb8, USolids) is doing this
  // unplaced.GetBBox();
  // actually this could also give us some indication which face is likely to be hit

  Real_v bbdistance = Real_v(kInfLength);
  BoxImplementation::DistanceToIn(BoxStruct<Precision>(unplaced.fBBdimensions), point - unplaced.fBBorigin, direction,
                                  stepMax, bbdistance);

  distance = InfinityLength<Real_v>();

  // do a check on bbdistance
  // if none of the tracks can hit even the bounding box; just return
  Bool_v done = bbdistance >= InfinityLength<Real_v>();
  if (vecCore::MaskFull(done)) return;

  // some particle could hit z
  Real_v zsafety = Abs(point.z()) - unplaced.fDz;
  Bool_v canhitz = zsafety > Real_v(MakeMinusTolerant<true>(0.));
  canhitz        = canhitz && (point.z() * direction.z() < 0); // coming towards the origin
  canhitz        = canhitz && (!done);

  if (!vecCore::MaskEmpty(canhitz)) {
    //   std::cerr << "can potentially hit\n";
    // calculate distance to z-plane ( see Box algorithm )
    // check if hit point is inside top or bottom polygon
    Real_v next = zsafety / NonZeroAbs(direction.z());
    // transport to z-height of planes
    Real_v coord1 = point.x() + next * direction.x();
    Real_v coord2 = point.y() + next * direction.y();
    Bool_v top    = direction.z() < 0;
    Bool_v hits   = IsInTopOrBottomPolygon<Real_v>(unplaced, coord1, coord2, top);
    hits          = hits && canhitz;
    vecCore::MaskedAssign(distance, hits, bbdistance);
    done = done || hits;
    if (vecCore::MaskFull(done)) return;
  }

  // now treat lateral surfaces
  Real_v disttoplanes = unplaced.fSurfaceShell.DistanceToIn<Real_v>(point, direction, done);
  vecCore__MaskedAssignFunc(distance, !done, Min(disttoplanes, distance));
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::DistanceToIn(UnplacedStruct_t const &unplaced,
                                                                                      Vector3D<Real_v> const &point,
                                                                                      Vector3D<Real_v> const &direction,
                                                                                      Real_v const &stepMax,
                                                                                      Real_v &distance)
{
  GenTrapImplementation::DistanceToInGeneric<Real_v>(unplaced, point, direction, stepMax, distance);
}

//______________________________________________________________________________
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::DistanceToIn(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
    Precision const &stepMax, Precision &distance)
{
// Scalar specialization of DistanceToIn
#ifndef VECCORE_CUDA
  if (unplaced.fTslHelper) {
    Precision invdirz = 1. / NonZero(direction.z());
    distance          = unplaced.fTslHelper->DistanceToIn<false>(point, direction, invdirz, stepMax);
    return;
  }
#endif
  GenTrapImplementation::DistanceToInGeneric<Precision>(unplaced, point, direction, stepMax, distance);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::DistanceToOutGeneric(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction,
    Real_v const & /* stepMax */, Real_v &distance)
{

  using Bool_v = vecCore::Mask_v<Real_v>;

  // we should check here the compilation condition
  // that treatNormal=true can only happen when Backend=kScalar
  // TODO: do this with some nice template features

  Bool_v negDirMask = direction.z() < 0;
  Real_v sign(1.0);
  vecCore__MaskedAssignFunc(sign, negDirMask, Real_v(-1.));
  //    Real_v invDirZ = 1./NonZero(direction.z());
  // this construct costs one multiplication more
  Real_v distmin = (sign * unplaced.fDz - point.z()) / NonZero(direction.z());

  Real_v distplane = unplaced.fSurfaceShell.DistanceToOut<Real_v>(point, direction);
  distance         = Min(distmin, distplane);
}

template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::DistanceToOut(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction,
    Real_v const &stepMax, Real_v &distance)
{
  GenTrapImplementation::DistanceToOutGeneric<Real_v>(unplaced, point, direction, stepMax, distance);
}

//______________________________________________________________________________
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::DistanceToOut(
    UnplacedStruct_t const &unplaced, Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
    Precision const &stepMax, Precision &distance)

{
#ifndef VECCORE_CUDA
  if (unplaced.fTslHelper) {
    distance = unplaced.fTslHelper->DistanceToOut(point, direction);
    return;
  }
#endif
  GenTrapImplementation::DistanceToOutGeneric<Precision>(unplaced, point, direction, stepMax, distance);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::SafetyToInGeneric(
    UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety)
{
  // Generic implementation for SafetyToIn
  using Bool_v = vecCore::Mask_v<Real_v>;

  Bool_v inside;
  // Check if all points are outside bounding box
  BoxImplementation::Contains(BoxStruct<Precision>(unplaced.fBBdimensions), point - unplaced.fBBorigin, inside);
  if (vecCore::MaskEmpty(inside)) {
    // All points outside, so compute safety using the bounding box
    // This is not optimal if top and bottom faces are not on top of each other
    BoxImplementation::SafetyToIn(BoxStruct<Precision>(unplaced.fBBdimensions), point - unplaced.fBBorigin, safety);
    return;
  }

  // Do Z
  safety = Abs(point[2]) - unplaced.fDz;
  safety = unplaced.fSurfaceShell.SafetyToIn<Real_v>(point, safety);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::SafetyToIn(UnplacedStruct_t const &unplaced,
                                                                                    Vector3D<Real_v> const &point,
                                                                                    Real_v &safety)
{
  // Dispatcher for SafetyToIn
  GenTrapImplementation::SafetyToInGeneric<Real_v>(unplaced, point, safety);
}

//______________________________________________________________________________
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::SafetyToIn(UnplacedStruct_t const &unplaced,
                                                                                    Vector3D<Precision> const &point,
                                                                                    Precision &safety)
{
// Scalar specialization for SafetyToIn
#ifndef VECCORE_CUDA
  if (unplaced.fTslHelper) {
    safety = unplaced.fTslHelper->SafetyToIn(point);
    return;
  }
#endif
  GenTrapImplementation::SafetyToInGeneric<Precision>(unplaced, point, safety);
}

//______________________________________________________________________________
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::SafetyToOutGeneric(UnplacedStruct_t const &unplaced,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
{
  // Generic implementation for SafetyToOut
  // Do Z
  safety = unplaced.fDz - Abs(point[2]);
  safety = unplaced.fSurfaceShell.SafetyToOut<Real_v>(point, safety);
}

//______________________________________________________________________________
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                Vector3D<Real_v> const &point, Real_v &safety)
{
  // Dispatcher for SafetyToOut
  GenTrapImplementation::SafetyToOutGeneric<Real_v>(unplaced, point, safety);
}

//______________________________________________________________________________
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::SafetyToOut(UnplacedStruct_t const &unplaced,
                                                                                     Vector3D<Precision> const &point,
                                                                                     Precision &safety)
{
// Scalar specialization for SafetyToOut
#ifndef VECCORE_CUDA
  if (unplaced.fTslHelper) {
    safety = unplaced.fTslHelper->SafetyToOut(point);
    return;
  }
#endif
  GenTrapImplementation::SafetyToOutGeneric<Precision>(unplaced, point, safety);
}

//______________________________________________________________________________
template <typename Real_v, typename Bool_v>
VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::NormalKernel(UnplacedStruct_t const &unplaced,
                                                                 Vector3D<Real_v> const &point,
                                                                 Vector3D<Real_v> &normal, Bool_v &valid)
{

  // Computes the normal on a surface and returns it as a unit vector
  //   In case a point is further than tolerance_normal from a surface, set validNormal=false
  //   Must return a valid vector. (even if the point is not on the surface.)

  using Index_v = vecCore::Index_v<Real_v>;

  valid = Bool_v(true);
  normal.Set(0., 0., 0.);
  // Do bottom and top faces
  Real_v safz = Abs(unplaced.fDz - Abs(point.z()));
  Bool_v onZ  = (safz < 10. * kTolerance);
  vecCore__MaskedAssignFunc(normal[2], onZ && (point.z() > Real_v(0.)), Real_v(1.));
  vecCore__MaskedAssignFunc(normal[2], onZ && (point.z() < Real_v(0.)), Real_v(-1.));

  //    if (vecCore::EarlyReturnAllowed()) {
  if (vecCore::MaskFull(onZ)) {
    return;
  }
  //    }
  //  Real_v done = onZ;
  // Get the closest edge (point should be on this edge within tolerance)
  Real_v cf = unplaced.fHalfInverseDz * (unplaced.fDz - point.z());
  Real_v vertexX[4];
  Real_v vertexY[4];
  for (int i = 0; i < 4; i++) {
    // calculate x-y positions of vertex i at this z-height
    vertexX[i] = unplaced.fVerticesX[i + 4] + cf * unplaced.fConnectingComponentsX[i];
    vertexY[i] = unplaced.fVerticesY[i + 4] + cf * unplaced.fConnectingComponentsY[i];
  }
  Real_v seg;
  Real_v frac;
  GetClosestEdge<Real_v>(point, vertexX, vertexY, seg, frac);
  vecCore__MaskedAssignFunc(frac, frac < Real_v(0.), Real_v(0.));
  Index_v iseg = (Index_v)seg;
  if (unplaced.IsPlanar()) {
    // Normals for the planar case are pre-computed
    Vertex_t const *normals = unplaced.fSurfaceShell.GetNormals();
    normal                  = normals[iseg];
    return;
  }
  Index_v jseg = (iseg + 1) % 4;
  Real_v x0    = vertexX[iseg];
  Real_v y0    = vertexY[iseg];
  Real_v x2    = vertexX[jseg];
  Real_v y2    = vertexY[jseg];
  x0 += frac * (x2 - x0);
  y0 += frac * (y2 - y0);
  Real_v x1 = unplaced.fVerticesX[iseg + 4];
  Real_v y1 = unplaced.fVerticesY[iseg + 4];
  x1 += frac * (unplaced.fVerticesX[jseg + 4] - x1);
  y1 += frac * (unplaced.fVerticesY[jseg + 4] - y1);
  Real_v ax = x1 - x0;
  Real_v ay = y1 - y0;
  Real_v az = unplaced.fDz - point.z();
  Real_v bx = x2 - x0;
  Real_v by = y2 - y0;
  Real_v bz = Real_v(0.);
  // Cross product of the vector given by the section segment (that contains the
  // point) at z=point[2] and the vector connecting the point projection to its
  // correspondent on the top edge.
  normal.Set(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
  normal.Normalize();
}

//______________________________________________________________________________
template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void GenTrapImplementation::GetClosestEdge(Vector3D<Real_v> const &point, Real_v vertexX[4],
                                                                   Real_v vertexY[4], Real_v &iseg, Real_v &fraction)
{
  /// Get index of the edge of the quadrilater represented by vert closest to point.
  /// If [P1,P2] is the closest segment and P is the point, the function returns the fraction of the
  /// projection of (P1P) over (P1P2). If projection of P is not in range [P1,P2] return -1.

  using Bool_v = vecCore::Mask_v<Real_v>;

  iseg = Real_v(0.);
  //  Real_v p1X, p1Y, p2X, p2Y;
  Real_v lsq, dx, dy, dpx, dpy, u;
  fraction    = Real_v(-1.);
  Real_v safe = InfinityLength<Real_v>();
  Real_v ssq  = InfinityLength<Real_v>();
  for (int i = 0; i < 4; ++i) {
    int j = (i + 1) % 4;
    dx    = vertexX[j] - vertexX[i];
    dy    = vertexY[j] - vertexY[i];
    dpx   = point.x() - vertexX[i];
    dpy   = point.y() - vertexY[i];
    lsq   = dx * dx + dy * dy;
    // Current segment collapsed to a point
    Bool_v collapsed = lsq < kTolerance;
    if (!vecCore::MaskEmpty(collapsed)) {
      vecCore__MaskedAssignFunc(ssq, lsq < kTolerance, dpx * dpx + dpy * dpy);
      // Missing a masked assign allowing to perform multiple assignments...
      vecCore::MaskedAssign(iseg, ssq < safe, (Precision)i);
      vecCore__MaskedAssignFunc(fraction, ssq < safe, Real_v(-1.));
      vecCore::MaskedAssign(safe, ssq < safe, ssq);
      if (vecCore::MaskFull(collapsed)) continue;
    }
    // Projection fraction
    u = (dpx * dx + dpy * dy) / NonZero(lsq);
    vecCore__MaskedAssignFunc(dpx, u > 1 && !collapsed, point.x() - vertexX[j]);
    vecCore__MaskedAssignFunc(dpy, u > 1 && !collapsed, point.y() - vertexY[j]);
    vecCore__MaskedAssignFunc(dpx, u >= 0 && u <= 1 && !collapsed, dpx - u * dx);
    vecCore__MaskedAssignFunc(dpy, u >= 0 && u <= 1 && !collapsed, dpy - u * dy);
    vecCore__MaskedAssignFunc(u, (u > 1 || u < 0) && !collapsed, Real_v(-1.));
    ssq = dpx * dpx + dpy * dpy;
    vecCore::MaskedAssign(iseg, ssq < safe, (Precision)i);
    vecCore::MaskedAssign(fraction, ssq < safe, u);
    vecCore::MaskedAssign(safe, ssq < safe, ssq);
  }
}

//*****************************
//**** Implementations end here
//*****************************
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* GENTRAPIMPLEMENTATION_H_ */
