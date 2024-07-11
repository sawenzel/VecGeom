#ifndef VECGEOM_PLANAR_POLYGON_H
#define VECGEOM_PLANAR_POLYGON_H

#include "VecGeom/base/Global.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Vector.h"
#include <VecCore/VecCore>
#include <iostream>
#include <limits>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlanarPolygon;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlanarPolygon);

inline namespace VECGEOM_IMPL_NAMESPACE {

// a class representing a 2D convex or concav polygon
class PlanarPolygon {

  friend struct SExtruImplementation;

protected:
  // we have to work on the "ideal" memory layout/placement for this
  // this is WIP
  SOA3D<Precision> fVertices; // a vector of vertices with links between
                              // note that the z component will hold the slopes between 2 links
                              // We assume a clockwise order of points
  Vector<Precision> fShiftedXJ;
  Vector<Precision> fShiftedYJ;
  Vector<Precision> fLengthSqr;    // the lenghts of each segment
  Vector<Precision> fInvLengthSqr; // the inverse square lengths of each segment
  Vector<Precision> fA;            // the "a"=x coefficient in the plane equation
  Vector<Precision> fB;            // the "b"=y coefficient in the plane equation
  Vector<Precision> fD;            // the "d" coefficient in the plane equation

  bool fIsConvex;  // convexity property to be calculated a construction time
  Precision fMinX; // the extent of the polygon
  Precision fMinY;
  Precision fMaxX;
  Precision fMaxY;

  size_t fNVertices; // the actual number of vertices
  friend class PolygonalShell;

public:
  VECCORE_ATT_HOST_DEVICE
  PlanarPolygon()
      : fVertices(), fShiftedXJ({}), fShiftedYJ({}), fLengthSqr({}), fInvLengthSqr({}), fA({}), fB({}), fD({}),
        fIsConvex(false), fMinX(kInfLength), fMinY(kInfLength), fMaxX(-kInfLength), fMaxY(-kInfLength), fNVertices(0)
  {
  }

  // constructor (not taking ownership of the pointers)
  VECCORE_ATT_HOST_DEVICE
  PlanarPolygon(int nvertices, Precision *x, Precision *y)
      : fVertices(nvertices), fShiftedXJ(nvertices), fShiftedYJ(nvertices), fLengthSqr(nvertices),
        fInvLengthSqr(nvertices), fA(nvertices), fB(nvertices), fD(nvertices), fIsConvex(false), fMinX(kInfLength),
        fMinY(kInfLength), fMaxX(-kInfLength), fMaxY(-kInfLength), fNVertices(nvertices)
  {
    Init(nvertices, x, y);
  }

  VECCORE_ATT_HOST_DEVICE
  void Init(int nvertices, Precision *x, Precision *y)
  {
    // allocating more space than nvertices, in order
    // to accomodate an internally vectorized treatment without tails
    // --> the size comes from this formula:
    const size_t kVS                = vecCore::VectorSize<vecgeom::VectorBackend::Real_v>();
    const auto numberOfVectorChunks = (nvertices / kVS + nvertices % kVS);
    // actual buffersize
    const auto bs = numberOfVectorChunks * kVS;
    assert(bs > 0);
    fNVertices = nvertices;
    fVertices.reserve(bs);
    fVertices.resize(nvertices);
    fShiftedXJ.resize(bs, 0);
    fShiftedYJ.resize(bs, 0);
    fLengthSqr.resize(bs, 0);
    fInvLengthSqr.resize(bs, 0);
    fA.resize(bs, 0);
    fB.resize(bs, 0);
    fD.resize(bs, 0);

    int inc = (GetOrientation(x, y, nvertices) > 0) ? -1 : 1;
    size_t i, j;
    // init the vertices (wrapping around periodically)
    for (i = 0; i < (size_t)fNVertices; ++i) {
      const size_t k = (i * inc + fNVertices) % fNVertices;
      fVertices.set(i, x[k], y[k], 0);
      fMinX = vecCore::math::Min(x[k], fMinX);
      fMinY = vecCore::math::Min(y[k], fMinY);
      fMaxX = vecCore::math::Max(x[k], fMaxX);
      fMaxY = vecCore::math::Max(y[k], fMaxY);
    }

    // initialize and cache the slopes as a "hidden" component
    auto slopes      = fVertices.z();
    const auto S     = fNVertices;
    const auto vertx = fVertices.x();
    const auto verty = fVertices.y();
    for (i = 0, j = S - 1; i < S; j = i++) {
      const auto vertxI = vertx[i];
      const auto vertxJ = vertx[j];

      const auto vertyI = verty[i];
      const auto vertyJ = verty[j];

      slopes[i]     = (vertxJ - vertxI) / NonZero(vertyJ - vertyI);
      fShiftedYJ[i] = vertyJ;
      fShiftedXJ[i] = vertxJ;
    }

    for (i = 0; i < (size_t)S; ++i) {
      fLengthSqr[i] = (vertx[i] - fShiftedXJ[i]) * (vertx[i] - fShiftedXJ[i]) +
                      (verty[i] - fShiftedYJ[i]) * (verty[i] - fShiftedYJ[i]);
      fInvLengthSqr[i] = 1. / fLengthSqr[i];
    }

    // init normals
    // this is taken from UnplacedTrapezoid
    // we should make this a standalone function outside any volume class
    for (i = 0; i < (size_t)S; ++i) {
      const auto xi = fVertices.x();
      const auto yi = fVertices.y();

      // arbitary choice of normal for the moment
      auto a = -(fShiftedYJ[i] - yi[i]);
      auto b = +(fShiftedXJ[i] - xi[i]);

      auto norm = 1.0 / std::sqrt(a * a + b * b); // normalization factor, always positive
      a *= norm;
      b *= norm;

      auto d = -(a * xi[i] + b * yi[i]);

      // fix (sign of zero (avoid -0 ))
      if (std::abs(a) < kTolerance) a = 0.;
      if (std::abs(b) < kTolerance) b = 0.;
      if (std::abs(d) < kTolerance) d = 0.;

      //      std::cerr << a << "," << b << "," << d << "\n";

      fA[i] = a;
      fB[i] = b;
      fD[i] = d;
    }

    // fill rest of data buffers periodically (for safe internal vectorized treatment)
    for (i = S; i < bs; ++i) {
      const size_t k = i % fNVertices;
      fVertices.set(i, fVertices.x()[k], fVertices.y()[k], fVertices.z()[k]);
      fShiftedXJ[i]    = fShiftedXJ[k];
      fShiftedYJ[i]    = fShiftedYJ[k];
      fLengthSqr[i]    = fLengthSqr[k];
      fInvLengthSqr[i] = fInvLengthSqr[k];
      fA[i]            = fA[k];
      fB[i]            = fB[k];
      fD[i]            = fD[k];
    }

    // set convexity
    CalcConvexity();

// check orientation
#ifndef VECCORE_CUDA
    if (Area() < 0.) {
      throw std::runtime_error("Polygon not given in clockwise order");
    }
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetMinX() const { return fMinX; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetMinY() const { return fMinY; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetMaxX() const { return fMaxX; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetMaxY() const { return fMaxY; }

  VECCORE_ATT_HOST_DEVICE
  SOA3D<Precision> const &GetVertices() const { return fVertices; }

  VECCORE_ATT_HOST_DEVICE
  size_t GetNVertices() const { return fNVertices; }

  // checks if 2D coordinates (x,y) are on the line segment given by index i
  template <typename Real_v, typename InternalReal_v, typename Bool_v>
  VECCORE_ATT_HOST_DEVICE Bool_v OnSegment(size_t i, Real_v const &px, Real_v const &py) const
  {
    using vecCore::FromPtr;

    // static assert ( cannot have Real_v == InternalReal_v )

    Bool_v result(false);
    //
    const auto vertx = fVertices.x();
    const auto verty = fVertices.y();
    // const auto slopes = fVertices.z();

    // check if cross close to zero
    const Real_v bx(FromPtr<InternalReal_v>(&vertx[i]));
    const Real_v by(FromPtr<InternalReal_v>(&verty[i]));
    const Real_v ax(FromPtr<InternalReal_v>(&fShiftedXJ[i]));
    const Real_v ay(FromPtr<InternalReal_v>(&fShiftedYJ[i]));
    // const Real_v slope(FromPtr<InternalReal_v>(&slopes[i]));

    const Real_v pymay(py - ay);
    const Real_v pxmax(px - ax);
    const Real_v epsilon(1E-9);

    // optimized crossproduct
    const Real_v cross     = (pymay * (bx - ax) - pxmax * (by - ay));
    const Bool_v collinear = Abs(cross) < epsilon;
    // TODO: can we use the slope?
    // const Bool_v collinear = Abs(pymay - slope * pxmax) < epsilon;

    if (vecCore::MaskFull(!collinear)) {
      return result;
    }
    result |= collinear;

    // can we do this with the slope??
    const auto dotproduct = pxmax * (bx - ax) + pymay * (by - ay);

    // check if length correct (use MakeTolerant templates)
    const Real_v tol(kTolerance);
    result &= (dotproduct >= -tol);
    result &= (dotproduct <= tol + Real_v(FromPtr<InternalReal_v>(&fLengthSqr[i])));
    return result;
  }

  template <typename Real_v, typename Bool_v = vecCore::Mask_v<Real_v>>
  VECCORE_ATT_HOST_DEVICE inline Bool_v ContainsConvex(Vector3D<Real_v> const &point) const
  {
    const size_t S = fVertices.size();
    Bool_v result(false);
    Real_v distance = -InfinityLength<Real_v>();
    for (size_t i = 0; i < S; ++i) {
      Real_v dseg = -(fA[i] * point.x() + fB[i] * point.y() + fD[i]);
      vecCore__MaskedAssignFunc(distance, dseg > distance, dseg);
    }
    result = distance < Real_v(kTolerance);
    return result;
  }

  template <typename Real_v, typename Bool_v = vecCore::Mask_v<Real_v>>
  VECCORE_ATT_HOST_DEVICE inline Bool_v Contains(Vector3D<Real_v> const &point) const
  {
    const size_t S = fVertices.size();
    Bool_v result(false);
    // implementation based on the point-polygon test after Jordan
    const auto vertx  = fVertices.x();
    const auto verty  = fVertices.y();
    const auto slopes = fVertices.z();
    const auto py     = point.y();
    const auto px     = point.x();
    for (size_t i = 0; i < S; ++i) {
      const auto vertyI       = verty[i];
      const auto vertyJ       = fShiftedYJ[i];
      const Bool_v condition1 = (vertyI > py) ^ (vertyJ > py);

      // early return leads to performance slowdown
      //  if (vecCore::MaskEmpty(condition1))
      //    continue;
      const auto vertxI     = vertx[i];
      const auto condition2 = px < (slopes[i] * (py - vertyI) + vertxI);

      result = (condition1 & condition2) ^ result;
    }
    return result;
  }

  template <typename Real_v, typename Inside_v = int /*vecCore::Index_v<Real_v>*/>
  VECCORE_ATT_HOST_DEVICE inline Inside_v InsideConvex(Vector3D<Real_v> const &point) const
  {
    assert(fIsConvex);
    const size_t S  = fVertices.size();
    Inside_v result = Inside_v(vecgeom::kOutside);
    Real_v distance = -InfinityLength<Real_v>();
    for (size_t i = 0; i < S; ++i) {
      Real_v dseg = -(fA[i] * point.x() + fB[i] * point.y() + fD[i]);
      vecCore__MaskedAssignFunc(distance, dseg > distance, dseg);
    }
    vecCore__MaskedAssignFunc(result, distance < Real_v(-kTolerance), Real_v(vecgeom::kInside));
    vecCore__MaskedAssignFunc(result, distance < Real_v(kTolerance), Real_v(vecgeom::kSurface));
    return result;
  }

  // calculate an underestimate of safety for the convex case
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v SafetyConvex(Vector3D<Real_v> const &point, bool inside) const
  {
    assert(fIsConvex);
    const size_t S  = fVertices.size();
    Real_v distance = -InfinityLength<Real_v>();
    for (size_t i = 0; i < S; ++i) {
      Real_v dseg = -(fA[i] * point.x() + fB[i] * point.y() + fD[i]);
      vecCore__MaskedAssignFunc(distance, dseg > distance, dseg);
      if (inside) distance *= Real_v(-1.);
    }
    return distance;
  }

  // calculate precise safety sqr to the polygon; return the closest "line" id
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v SafetySqr(Vector3D<Real_v> const &point, int &closestid) const
  {
    // implementation based on TGeoPolygone@ROOT
    Real_v safe(1E30);
    int isegmin = -1;

    const auto vertx = fVertices.x();
    const auto verty = fVertices.y();
    const auto S     = fVertices.size();
    for (size_t i = 0; i < S; ++i) {

      // could use the slope information to calc
      const Real_v p1[2] = {vertx[i], verty[i]};
      const Real_v p2[2] = {fShiftedXJ[i], fShiftedYJ[i]};

      const auto dx = p2[0] - p1[0];
      const auto dy = p2[1] - p1[1];
      auto dpx      = point.x() - p1[0];
      auto dpy      = point.y() - p1[1];

      // degenerate edge?
      // const auto lsq = dx * dx + dy * dy;

      // I don't think this is useful -- its a pure static property
      //         if ( ClostToZero(lsq,0)) {
      //            ssq = dpx*dpx + dpy*dpy;
      //            if (ssq < safe) {
      //               safe = ssq;
      //               isegmin = i;
      //            }
      //            continue;
      //         }

      const auto u = (dpx * dx + dpy * dy) * fInvLengthSqr[i];
      if (u > 1) {
        dpx = point.x() - p2[0];
        dpy = point.y() - p2[1];
      } else {
        if (u >= 0) {
          // need to divide by lsq now
          // since this is a static property of the polygon
          // we could actually cache it;
          dpx -= u * dx;
          dpy -= u * dy;
        }
      }
      const auto ssq = dpx * dpx + dpy * dpy;
      if (ssq < safe) {
        safe    = ssq;
        isegmin = i;
      }

      // check if we are done early ( on surface )
      if (Abs(safe) < kTolerance * kTolerance) {
        closestid = isegmin;
        return Real_v(0.);
      }
    }
    closestid = isegmin;
    return safe;
  }

  VECCORE_ATT_HOST_DEVICE
  bool IsConvex() const { return fIsConvex; }

  // check clockwise/counterclockwise condition (returns positive for anti-clockwise)
  // useful function to check orientation of points x,y
  // before calling the PlanarPolygon constructor
  VECCORE_ATT_HOST_DEVICE
  static Precision GetOrientation(Precision *x, Precision *y, size_t N)
  {
    Precision area(0.);
    for (size_t i = 0; i < N; ++i) {
      const Precision p1[2] = {x[i], y[i]};
      const size_t j        = (i + 1) % N;
      const Precision p2[2] = {x[j], y[j]};
      area += (p1[0] * p2[1] - p1[1] * p2[0]);
    }
    return area;
  }

  /* returns area of polygon */
  VECCORE_ATT_HOST_DEVICE
  Precision Area() const
  {
    const auto vertx = fVertices.x();
    const auto verty = fVertices.y();

    const auto kS = fVertices.size();
    Precision area(0.);
    for (size_t i = 0; i < kS; ++i) {
      const Precision p1[2] = {vertx[i], verty[i]};
      const Precision p2[2] = {fShiftedXJ[i], fShiftedYJ[i]};

      area += (p1[0] * p2[1] - p1[1] * p2[0]);
    }
    return 0.5 * area;
  }

private:
  VECCORE_ATT_HOST_DEVICE
  void CalcConvexity()
  {
    // check if we are always turning into the same sense
    // --> check if the sign of the cross product is always the same
    const auto vertx = fVertices.x();
    const auto verty = fVertices.y();

    const auto kS = fNVertices;
    int counter(0);
    for (size_t i = 0; i < kS; ++i) {
      size_t j              = (i + 1) % kS;
      size_t k              = (i + 2) % kS;
      const Precision p1[2] = {vertx[j] - vertx[i], verty[j] - verty[i]};
      const Precision p2[2] = {vertx[k] - vertx[j], verty[k] - verty[j]};
      counter += (p1[0] * p2[1] - p1[1] * p2[0]) < 0 ? -1 : 1;
    }
    fIsConvex = (size_t)std::abs(counter) == kS;
  }
};

// template specialization for scalar case (do internal vectorization)
#define SPECIALIZE
#ifdef SPECIALIZE

template <>
VECCORE_ATT_HOST_DEVICE inline bool PlanarPolygon::ContainsConvex(Vector3D<Precision> const &point) const
{
  const size_t S     = fVertices.size();
  Precision distance = -InfinityLength<Precision>();
  for (size_t i = 0; i < S; ++i) {
    Precision dseg = -(fA[i] * point.x() + fB[i] * point.y() + fD[i]);
    distance       = vecCore::math::Max(dseg, distance);
  }
  return (distance < 0.);
}

template <>
VECCORE_ATT_HOST_DEVICE inline bool PlanarPolygon::Contains(Vector3D<Precision> const &point) const
{

  using Real_v = vecgeom::VectorBackend::Real_v;
  using Bool_v = vecCore::Mask_v<Real_v>;
  using vecCore::FromPtr;

  const auto kVectorS = vecCore::VectorSize<Real_v>();

  Bool_v result(false);
  const Real_v px(point.x());
  const Real_v py(point.y());
  const size_t S       = fVertices.size();
  const size_t SVector = S - S % kVectorS;
  size_t i(0);
  const auto vertx  = fVertices.x();
  const auto verty  = fVertices.y();
  const auto slopes = fVertices.z();
  // treat vectorizable part of loop
  for (; i < SVector; i += kVectorS) {
    const Real_v vertyI(FromPtr<Real_v>(&verty[i]));      // init vectors
    const Real_v vertyJ(FromPtr<Real_v>(&fShiftedYJ[i])); // init vectors

    const auto condition1 = (vertyI > py) ^ (vertyJ > py); // xor

    const Real_v vertxI(FromPtr<Real_v>(&vertx[i]));
    const Real_v slope(FromPtr<Real_v>(&slopes[i]));
    const auto condition2 = px < (slope * (py - vertyI) + vertxI);

    result = result ^ (condition1 & condition2);
  }
  // reduction over vector lanes
  bool reduction(false);
  for (size_t j = 0; j < kVectorS; ++j) {
    if (vecCore::MaskLaneAt(result, j)) reduction = !reduction;
  }

  // treat tail
  using Real_s = vecCore::Scalar<Real_v>;
  for (; i < S; ++i) {
    const Real_s vertyI(FromPtr<Real_s>(&verty[i]));                     // init vectors
    const Real_s vertyJ(FromPtr<Real_s>(&fShiftedYJ[i]));                // init vectors
    const bool condition1 = (vertyI > point.y()) ^ (vertyJ > point.y()); // xor
    const Real_s vertxI(FromPtr<Real_s>(&vertx[i]));
    const Real_s slope(FromPtr<Real_s>(&slopes[i]));
    const bool condition2 = point.x() < (slope * (point.y() - vertyI) + vertxI);

    reduction = reduction ^ (condition1 & condition2);
  }
  return reduction;
}

template <>
VECCORE_ATT_HOST_DEVICE inline Inside_t PlanarPolygon::InsideConvex(Vector3D<Precision> const &point) const
{
  const size_t S = fVertices.size();
  assert(fIsConvex);
  Precision distance = -InfinityLength<Precision>();
  for (size_t i = 0; i < S; ++i) {
    Precision dseg = -(fA[i] * point.x() + fB[i] * point.y() + fD[i]);
    distance       = vecCore::math::Max(dseg, distance);
  }
  if (distance > kTolerance) return vecgeom::kOutside;
  if (distance < -kTolerance) return vecgeom::kInside;
  return vecgeom::kSurface;
}

// template specialization for convex safety
template <>
VECCORE_ATT_HOST_DEVICE inline Precision PlanarPolygon::SafetyConvex(Vector3D<Precision> const &point,
                                                                     bool inside) const
{
  const size_t S = fVertices.size();
  assert(fIsConvex);
  Precision distance = -InfinityLength<Precision>();
  for (size_t i = 0; i < S; ++i) {
    Precision dseg = -(fA[i] * point.x() + fB[i] * point.y() + fD[i]);
    distance       = vecCore::math::Max(dseg, distance);
  }
  if (inside) distance *= -1.;
  return distance;
}

// template specialization for scalar safety
template <>
VECCORE_ATT_HOST_DEVICE inline Precision PlanarPolygon::SafetySqr(Vector3D<Precision> const &point,
                                                                  int &closestid) const
{
  using Real_v = vecgeom::VectorBackend::Real_v;
  using vecCore::FromPtr;

  const auto kVectorS = vecCore::VectorSize<Real_v>();
  Precision safe(1E30);
  int isegmin(-1);

  const auto vertx = fVertices.x();
  const auto verty = fVertices.y();
  const auto S     = fVertices.size();
  const Real_v px(point.x());
  const Real_v py(point.y());
  for (size_t i = 0; i < S; i += kVectorS) {
    const Real_v p1[2] = {FromPtr<Real_v>(&vertx[i]), FromPtr<Real_v>(&verty[i])};
    const Real_v p2[2] = {FromPtr<Real_v>(&fShiftedXJ[i]), FromPtr<Real_v>(&fShiftedYJ[i])};

    const auto dx = p2[0] - p1[0];
    const auto dy = p2[1] - p1[1];
    auto dpx      = px - p1[0];
    auto dpy      = py - p1[1];

    // degenerate edge?
    const auto lsq = dx * dx + dy * dy;

    // I don't think this is useful -- its a pure static property
    //         if ( ClostToZero(lsq,0)) {
    //            ssq = dpx*dpx + dpy*dpy;
    //            if (ssq < safe) {
    //               safe = ssq;
    //               isegmin = i;
    //            }
    //            continue;
    //         }

    const auto u     = (dpx * dx + dpy * dy);
    const auto cond1 = (u > lsq);
    const auto cond2 = (!cond1 && (u >= Real_v(0.)));

    if (!vecCore::MaskEmpty(cond1)) {
      vecCore__MaskedAssignFunc(dpx, cond1, px - p2[0]);
      vecCore__MaskedAssignFunc(dpy, cond1, py - p2[1]);
    }
    if (!vecCore::MaskEmpty(cond2)) {
      const auto invlsq = Real_v(1.) / lsq;
      vecCore__MaskedAssignFunc(dpx, cond2, dpx - u * dx * invlsq);
      vecCore__MaskedAssignFunc(dpy, cond2, dpy - u * dy * invlsq);
    }
    const auto ssq = dpx * dpx + dpy * dpy;

// combined reduction is a bit tricky to translate:
// if (ssq < safe) {
//      safe = ssq;
//      isegmin = i;
// }

// a first try is serialized:
#ifndef VECCORE_CUDA
    using std::min;
#endif
    for (size_t j = 0; j < kVectorS; ++j) {
      Precision saftmp = vecCore::LaneAt(ssq, j);
      if (saftmp < safe) {
        safe    = saftmp;
        isegmin = i + j;
      }
    }
    if (Abs(safe) < kTolerance * kTolerance) {
      closestid = isegmin;
      return 0.;
    }
  }
  closestid = isegmin;
  return safe;
}
#endif

} // namespace VECGEOM_IMPL_NAMESPACE
} // end namespace vecgeom

#endif
