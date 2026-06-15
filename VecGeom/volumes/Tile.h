/// \file QuadrilateralFacet.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_TILE_H_
#define VECGEOM_VOLUMES_TILE_H_

#include "VecGeom/base/Vector3D.h"
#include "kernel/GenericKernels.h"

namespace vecgeom {

enum TileType { kTriangle = 3, kQuadrilateral = 4 };
namespace cuda {
template <typename Real_t>
struct TriangularTile;
}
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_1v_1t(struct, Tile, size_t, typename);
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(struct, TriangularTile, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <size_t, typename>
struct Tile;

template <typename T>
using TriangleFacet = Tile<3, T>;

template <typename T>
using QuadrilateralFacet = Tile<4, T>;

// a shorter, memory saving tile cmp to Tile
template <typename T>
struct TriangularTile {
  Vector3D<T> fVertices[3];
  Vector3D<T> fNormal;

  Vector3D<T> CalculateNormal()
  {
    const auto e1 = fVertices[1] - fVertices[0];
    const auto e2 = fVertices[2] - fVertices[0];
    auto n        = Vector3D<T>::Cross(e1, e2);
    n.Normalize();
    return n; //
  }

  // construct this from 3 vertices
  TriangularTile(Vector3D<T> v0, Vector3D<T> v1, Vector3D<T> v2) : fVertices{v0, v1, v2}
  {
    fNormal = CalculateNormal();
  }

  TriangularTile() = default;

  VECCORE_ATT_HOST_DEVICE
  T Distance(Vector3D<T> const &origin, Vector3D<T> const &dir, T rayEPS = 1e-8) const
  {
    // Moeller-Trumbore ray-triangle intersection
    using Vertex_t = Vector3D<T>;
    // Keep shallow but real facet crossings out of the "parallel" bucket.
    constexpr T EPS   = kToleranceDist<T>;
    const auto &v0    = fVertices[0];
    const auto &v1    = fVertices[1];
    const auto &v2    = fVertices[2];
    const T INF       = InfinityLength<T>();
    const Vertex_t e1 = v1 - v0; //
    const Vertex_t e2 = v2 - v0;
    auto p            = dir.Cross(e2);
    auto det          = e1.Dot(p);
    if (vecCore::math::Abs(det) <= EPS) {
      return INF;
    }

    const auto tvec = origin - v0;
    auto invDet     = 1.0 / det;
    auto u          = tvec.Dot(p) * invDet;
    if (u < T(0.0) || u > T(1.0)) {
      return INF;
    }
    auto q = tvec.Cross(e1);
    auto v = dir.Dot(q) * invDet;
    if (v < T(0.0) || u + v > T(1.0)) {
      return INF;
    }
    auto t = e2.Dot(q) * invDet;
    return (t > rayEPS) ? t : INF;
  }

  // calculate squared safety, possibly in a different precision that the storage precision
  template <typename P = float>
  VECCORE_ATT_HOST_DEVICE P SafetySq(Vector3D<P> const &p) const
  {
    using Vec3 = Vector3D<P>;
    const Vec3 a(fVertices[0]);
    const Vec3 b(fVertices[1]);
    const Vec3 c(fVertices[2]);

    // Edges in precision P ... constructed from original vertices in precision T
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = p - a;

    const P d1 = ab.Dot(ap);
    const P d2 = ac.Dot(ap);
    if (d1 <= P(0.0) && d2 <= P(0.0)) {
      return ap.Dot(ap); // barycentric (1,0,0)
    }

    const Vec3 bp = p - b;
    const P d3    = ab.Dot(bp);
    const P d4    = ac.Dot(bp);
    if (d3 >= P(0.0) && d4 <= d3) {
      return bp.Dot(bp); // (0,1,0)
    }

    const P vc = d1 * d4 - d3 * d2;
    if (vc <= P(0.0) && d1 >= P(0.0) && d3 <= P(0.0)) {
      const P v       = d1 / (d1 - d3);
      const Vec3 proj = a + v * ab;
      const Vec3 d    = p - proj;
      return d.Dot(d); // edge AB
    }

    const Vec3 cp = p - c;
    const P d5    = ab.Dot(cp);
    const P d6    = ac.Dot(cp);
    if (d6 >= P(0.0f) && d5 <= d6) {
      return cp.Dot(cp); // (0,0,1)
    }

    const P vb = d5 * d2 - d1 * d6;
    if (vb <= P(0.0) && d2 >= P(0.0) && d6 <= P(0.0)) {
      const P w = d2 / (d2 - d6);
      Vec3 proj = a + w * ac;
      Vec3 d    = p - proj;
      return d.Dot(d); // edge AC
    }

    const P va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
      const P w     = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      const Vec3 bc = c - b;
      Vec3 proj     = b + w * bc;
      Vec3 d        = p - proj;
      return d.Dot(d); // edge BC
    }

    // Inside face region
    const P denom = P(1.0) / (va + vb + vc);
    const P v     = vb * denom;
    const P w     = vc * denom;

    Vec3 proj = a;
    proj += v * ab;
    proj += w * ac;
    const Vec3 d = p - proj;
    return d.Dot(d);
  }

  // Is the point inside this triangle ?
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool Contains(Vector3D<T> const &point) const
  {
    constexpr int NVERT = 3;
    bool inside         = true;
    for (size_t i = 0; i < NVERT; ++i) {
      const auto sideVector = fVertices[(i + 1) % NVERT] - fVertices[i];
      T saf                 = (point - fVertices[i]).Dot(sideVector);
      inside &= saf > -kTolerance;
      if (inside) {
        break;
      }
    }
    return inside;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  T SurfaceArea() const
  {
    T area{0.};
    constexpr int NVERT = 3;
    for (size_t i = 1; i < NVERT - 1; ++i) {
      Vector3D<T> e1 = fVertices[i] - fVertices[0];
      Vector3D<T> e2 = fVertices[i + 1] - fVertices[0];
      area += 0.5 * (e1.Cross(e2)).Mag();
    }
    return area;
  }
};

//______________________________________________________________________________
// Basic facet tile structure having NVERT vertices making a convex polygon.
// The vertices making the tile have to be given in anti-clockwise
// order looking from the outsider of the solid where it belongs.
//______________________________________________________________________________
template <size_t NVERT, typename T = double>
struct Tile {
  size_t fNvert = 0;               ///< the tile is fully defined after adding the last vertex
  Vector3D<T> fVertices[NVERT];    ///< vertices of the tile
  Vector3D<T> fSideVectors[NVERT]; ///< side vectors perpendicular to edges
  Vector3D<T> fNormal;             ///< normal vector pointing outside
  Vector3D<T> fCenter;             ///< Center of the tile
  size_t fIndices[NVERT] = {0};    ///< indices for 3 distinct vertices
  T fSurfaceArea         = 0;      ///< surface area
  bool fConvex           = false;  ///< convexity of the facet with respect to the solid
  T fDistance            = 0;      ///< distance between the origin and the triangle plane

  VECCORE_ATT_HOST_DEVICE
  Tile() {}

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool SetVertices(Vector3D<T> const &vtx0, Vector3D<T> const &vtx1, Vector3D<T> const &vtx2, size_t ind0 = 0,
                   size_t ind1 = 0, size_t ind2 = 0)
  {
    VECGEOM_ASSERT(NVERT == 3);
    AddVertex(vtx0, ind0);
    AddVertex(vtx1, ind1);
    return AddVertex(vtx2, ind2);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool SetVertices(Vector3D<T> const &vtx0, Vector3D<T> const &vtx1, Vector3D<T> const &vtx2, Vector3D<T> const &vtx3,
                   size_t ind0 = 0, size_t ind1 = 0, size_t ind2 = 0, size_t ind3 = 0)
  {
    VECGEOM_ASSERT(NVERT == 4);
    AddVertex(vtx0, ind0);
    AddVertex(vtx1, ind1);
    AddVertex(vtx2, ind2);
    return AddVertex(vtx3, ind3);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool AddVertex(Vector3D<T> const &vtx, size_t ind = 0)
  {
    fVertices[fNvert] = vtx;
    fIndices[fNvert]  = ind;
    fNvert++;
    if (fNvert < NVERT) return true;
    // Check validity
    // Get number of different vertices
    size_t nvert = NVERT;
    for (size_t i = 0; i < NVERT; ++i) {
      const Vector3D<T> vi = fVertices[(i + 1) % NVERT] - fVertices[i];
      if (vi.Mag2() < kTolerance) {
        nvert--;
      }
    }

    if (nvert < 3) {
      // std::cerr << "Tile degenerated: Length of sides of facet are too small." << std::endl;
      return false;
    }

    // Compute normal using non-zero segments

    bool degenerated = true;
    for (size_t i = 0; i < NVERT - 1; ++i) {
      Vector3D<T> e1 = fVertices[i + 1] - fVertices[i];
      if (e1.Mag2() < kTolerance) continue;
      for (size_t j = i + 1; j < NVERT; ++j) {
        Vector3D<T> e2 = fVertices[(j + 1) % NVERT] - fVertices[j];
        if (e2.Mag2() < kTolerance) continue;
        fNormal = e1.Cross(e2);
        // e1 and e2 may be colinear
        if (fNormal.Mag2() < kTolerance) continue;
        fNormal.Normalize();
        degenerated = false;
        break;
      }
      if (!degenerated) break;
    }

    if (degenerated) {
      // std::cerr << "Tile degenerated 2: Length of sides of facet are too small." << std::endl;
      return false;
    }

    // Compute side vectors
    for (size_t i = 0; i < NVERT; ++i) {
      Vector3D<T> e1 = fVertices[(i + 1) % NVERT] - fVertices[i];
      if (e1.Mag2() < kTolerance) continue;
      fSideVectors[i] = fNormal.Cross(e1).Normalized();
      fDistance       = -fNormal.Dot(fVertices[i]);
      for (size_t j = i + 1; j < i + NVERT; ++j) {
        Vector3D<T> e2 = fVertices[(j + 1) % NVERT] - fVertices[j % NVERT];
        if (e2.Mag2() < kTolerance)
          fSideVectors[j % NVERT] = fSideVectors[(j - 1) % NVERT];
        else
          fSideVectors[j % NVERT] = fNormal.Cross(e2).Normalized();
      }
      break;
    }

    // Compute surface area
    fSurfaceArea = 0.;
    for (size_t i = 1; i < NVERT - 1; ++i) {
      Vector3D<T> e1 = fVertices[i] - fVertices[0];
      Vector3D<T> e2 = fVertices[i + 1] - fVertices[0];
      fSurfaceArea += 0.5 * (e1.Cross(e2)).Mag();
    }
    VECGEOM_ASSERT(fSurfaceArea > kTolerance * kTolerance);

    // Center of the tile
    for (size_t i = 0; i < NVERT; ++i)
      fCenter += fVertices[i];
    fCenter /= NVERT;
    return true;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<T> const &GetNormal() const { return fNormal; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t IsNeighbor(Tile<NVERT, T> const &other)
  {
    // Check if a segment is common
    size_t ncommon = 0;
    for (size_t ind1 = 0; ind1 < NVERT; ++ind1) {
      for (size_t ind2 = 0; ind2 < NVERT; ++ind2) {
        if (fIndices[ind1] == other.fIndices[ind2]) ncommon++;
      }
    }
    return ncommon;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool Contains(Vector3D<T> const &point) const
  {
    // Check id point within the triangle plane is inside the triangle.
    bool inside = true;
    for (size_t i = 0; i < NVERT; ++i) {
      T saf = (point - fVertices[i]).Dot(fSideVectors[i]);
      inside &= saf > -kTolerance;
    }
    return inside;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  T DistPlane(Vector3D<T> const &point) const
  {
    // Returns distance from point to plane. This is positive if the point is on
    // the outside halfspace, negative otherwise.
    return (point.Dot(fNormal) + fDistance);
  }

  VECCORE_ATT_HOST_DEVICE
  T DistanceToIn(Vector3D<T> const &point, Vector3D<T> const &direction) const
  {
    T ndd      = NonZero(direction.Dot(fNormal));
    T saf      = DistPlane(point);
    bool valid = ndd < 0. && saf > -kTolerance;
    if (!valid) return InfinityLength<T>();
    T distance = -saf / ndd;
    // Propagate the point with the distance to the plane.
    Vector3D<T> point_prop = point + distance * direction;
    // Check if propagated points hit the triangle
    if (!Contains(point_prop)) return InfinityLength<T>();
    return distance;
  }

  VECCORE_ATT_HOST_DEVICE
  Precision DistanceToOut(Vector3D<T> const &point, Vector3D<T> const &direction) const
  {
    T ndd      = NonZero(direction.Dot(fNormal));
    T saf      = DistPlane(point);
    bool valid = ndd > 0. && saf < kTolerance;
    if (!valid) return InfinityLength<T>();
    T distance = -saf / ndd;
    // Propagate the point with the distance to the plane.
    Vector3D<T> point_prop = point + distance * direction;
    // Check if propagated points hit the triangle
    if (!Contains(point_prop)) return InfinityLength<T>();
    return distance;
  }

  template <bool ToIn>
  VECCORE_ATT_HOST_DEVICE T SafetySq(Vector3D<T> const &point) const
  {
    T safety = DistPlane(point);
    // Find the projection of the point on each plane
    Vector3D<T> intersection = point - safety * fNormal;
    bool withinBound         = Contains(intersection);
    if (ToIn)
      withinBound &= safety > -kTolerance;
    else
      withinBound &= safety < kTolerance;
    safety *= safety;
    if (withinBound) return safety;

    Vector3D<T> safety_outbound = InfinityLength<T>();
    for (size_t ivert = 0; ivert < NVERT; ++ivert) {
      safety_outbound[ivert] =
          DistanceToLineSegmentSquared<kScalar>(fVertices[ivert], fVertices[(ivert + 1) % NVERT], point);
    }
    return (safety_outbound.Min());
  }
};

#ifndef VECCORE_CUDA
std::ostream &operator<<(std::ostream &os, TriangleFacet<double> const &facet);
std::ostream &operator<<(std::ostream &os, QuadrilateralFacet<double> const &facet);
#endif
} // namespace VECGEOM_IMPL_NAMESPACE
} // end namespace vecgeom

#endif // VECGEOM_VOLUMES_TILE_H_
