/*
 * PolyhedronStruct.h
 *
 *  Created on: 09.12.2016
 *      Author: mgheata
 */
#ifndef VECGEOM_POLYHEDRONSTRUCT_H_
#define VECGEOM_POLYHEDRONSTRUCT_H_

#include <ostream>

#include <VecCore/VecCore>
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Quadrilaterals.h"
#include "VecGeom/volumes/Wedge_Evolution.h"
#include "VecGeom/base/Array.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/volumes/TubeStruct.h"

// These enums should be in the scope vecgeom::Polyhedron, but when used in the
// shape implementation helper instantiations, nvcc gets confused:

enum struct EInnerRadii { kFalse = -1, kGeneric = 0, kTrue = 1 };
enum struct EPhiCutout { kFalse = -1, kGeneric = 0, kTrue = 1, kLarge = 2 };

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct ZSegment;);
VECGEOM_DEVICE_DECLARE_CONV(struct, ZSegment);

// Declare types shared by cxx and cuda.
namespace Polyhedron {
using ::EInnerRadii;
using ::EPhiCutout;
} // namespace Polyhedron

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Represents one segment along the Z-axis, containing one or more sets of
/// quadrilaterals that represent the outer, inner and phi shells.
struct ZSegment {
  Quadrilaterals outer; ///< Should always be non-empty.
  Quadrilaterals phi;   ///< Is empty if fHasPhiCutout is false.
  Quadrilaterals inner; ///< Is empty hasInnerRadius is false.

  VECCORE_ATT_HOST_DEVICE
  bool hasInnerRadius() const { return inner.size() > 0; }

  VECCORE_ATT_HOST_DEVICE
  size_t aligned_sizeof_data(size_t nOuter, size_t nInner, size_t nPhi)
  {
    return Quadrilaterals::aligned_sizeof_data(nOuter) + Quadrilaterals::aligned_sizeof_data(nInner) +
           Quadrilaterals::aligned_sizeof_data(nPhi);
  }

  ZSegment() = default;

  VECCORE_ATT_HOST_DEVICE
  ZSegment(size_t nOuter, size_t nInner, size_t nPhi, AlignedAllocator &a, bool convex = true)
      : outer(nOuter, a), phi(nPhi, a, convex), inner(nInner, a)
  {
  }
};

// a plain and lightweight struct to encapsulate data members of a polyhedron
template <typename T = double>
struct PolyhedronStruct {
  size_t fSize{0};                ///< Size of the buffer to hold the object, including alignment
  int fSideCount{0};              ///< Number of segments along phi.
  bool fHasInnerRadii{false};     ///< Has any Z-segments with an inner radius != 0.
  bool fHasPhiCutout{false};      ///< Has a cutout angle along phi.
  bool fHasLargePhiCutout{false}; ///< Phi cutout is larger than pi.
  T fPhiStart{0.};                ///< Phi start in radians (input to constructor)
  T fPhiDelta{0.};                ///< Phi delta in radians (input to constructor)
  evolution::Wedge fPhiWedge;     ///< Phi wedge
  Array<ZSegment> fZSegments;     ///< AOS'esque collections of quadrilaterals
  Array<T> fZPlanes;              ///< Z-coordinate of each plane separating segments
  Array<T> fRMin;                 ///< Inner radii as specified in constructor.
  Array<T> fRMax;                 ///< Outer radii as specified in constructor.
  Array<bool> fSameZ;             ///< Array of flags marking that the following plane is at same Z
  SOA3D<T> fPhiSections;          ///< Unit vectors marking the bounds between
                                  ///  phi segments, represented by planes
                                  ///  through the origin with the normal
                                  ///  point along the positive phi direction.
  TubeStruct<T> fBoundingTube;    ///< Tube enclosing the outer bounds of the
                                  ///  polyhedron. Used in Contains, Inside and
                                  ///  DistanceToIn.
  T fBoundingTubeOffset{0.};      ///< Offset in Z of the center of the bounding
                                  ///  tube. Used as a quick substitution for
                                  ///  running a full transformation.

  /// Internal structure to cache component surface areas per Z segment
  struct AreaStruct {
    Precision area        = 0.;      ///< Cached total surface area
    Precision top_area    = 0.;      ///< Area of top surface
    Precision bottom_area = 0.;      ///< Area of top surface
    Precision *outer      = nullptr; ///< Array of surface areas for the auter part
    Precision *inner      = nullptr; ///< Array of surface areas for the inner part
    Precision *phi        = nullptr; ///< Array of surface areas for the phi part

    AreaStruct(int nseg)
    {
      inner = new Precision[nseg];
      outer = new Precision[nseg];
      phi   = new Precision[nseg];
    }

    VECCORE_ATT_HOST_DEVICE
    ~AreaStruct()
    {
      delete[] inner;
      delete[] outer;
      delete[] phi;
    }
  };

  mutable AreaStruct *fAreaStruct = nullptr; ///< Cached surface area values
  mutable Precision fCapacity     = 0.;      ///< Stored Capacity

  // These data member and member functions are added for convexity detection
  bool fContinuousInSlope;
  bool fConvexityPossible;
  bool fEqualRmax;

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static bool ApproxEqual(const Precision &x, const Precision &y) { return vecCore::math::Abs(x - y) < kTolerance; }

  PolyhedronStruct() = default;

  VECCORE_ATT_HOST_DEVICE
  PolyhedronStruct(Precision phiStart, Precision phiDelta, const int sideCount, const int zPlaneCount,
                   Precision const zPlanes[], Precision const rMin[], Precision const rMax[])
      : fSideCount(sideCount), fHasInnerRadii(false), fHasPhiCutout(phiDelta < kTwoPi),
        fHasLargePhiCutout(phiDelta < kPi), fPhiStart(NormalizeAngle<kScalar>(phiStart)),
        fPhiDelta((phiDelta > kTwoPi) ? kTwoPi : phiDelta), fPhiWedge(fPhiDelta, fPhiStart),
        fZSegments(zPlaneCount - 1), fZPlanes(zPlaneCount), fRMin(zPlaneCount), fRMax(zPlaneCount), fSameZ(zPlaneCount),
        fPhiSections(sideCount + 1), fBoundingTube(0, 1, 1, fPhiStart, fPhiDelta), fContinuousInSlope(true),
        fConvexityPossible(true), fEqualRmax(true)
  {
    // initialize polyhedron internals
    Initialize(phiStart, phiDelta, sideCount, zPlaneCount, zPlanes, rMin, rMax);
  }

  VECCORE_ATT_HOST_DEVICE
  PolyhedronStruct(Precision phiStart, Precision phiDelta, const int sideCount, const int zPlaneCount,
                   Precision const zPlanes[], Precision const rMin[], Precision const rMax[], AlignedAllocator &a)
      : fSideCount(sideCount), fHasInnerRadii(false), fHasPhiCutout(phiDelta < kTwoPi),
        fHasLargePhiCutout(phiDelta < kPi), fPhiStart(NormalizeAngle<kScalar>(phiStart)),
        fPhiDelta((phiDelta > kTwoPi) ? kTwoPi : phiDelta), fPhiWedge(fPhiDelta, fPhiStart),
        fZSegments(zPlaneCount - 1, a), fZPlanes(zPlaneCount, a), fRMin(zPlaneCount, a), fRMax(zPlaneCount, a),
        fSameZ(zPlaneCount, a), fPhiSections(sideCount + 1), fBoundingTube(0, 1, 1, fPhiStart, fPhiDelta),
        fContinuousInSlope(true), fConvexityPossible(true), fEqualRmax(true)
  {
    // initialize polyhedron internals
    Initialize(phiStart, phiDelta, sideCount, zPlaneCount, zPlanes, rMin, rMax, a);
  }

  PolyhedronStruct(Precision phiStart, Precision phiDelta, const int sideCount, const int verticesCount,
                   Precision const r[], Precision const z[])
      : fSideCount(sideCount), fHasInnerRadii(false), fHasPhiCutout(phiDelta < kTwoPi),
        fHasLargePhiCutout(phiDelta < kPi), fPhiStart(NormalizeAngle<kScalar>(phiStart)),
        fPhiDelta((phiDelta > kTwoPi) ? kTwoPi : phiDelta), fPhiWedge(fPhiDelta, fPhiStart), fZSegments(), fZPlanes(),
        fRMin(), fRMax(), fSameZ(), fPhiSections(sideCount + 1), fBoundingTube(0, 1, 1, fPhiStart, fPhiDelta),
        fContinuousInSlope(true), fConvexityPossible(true), fEqualRmax(true)
  {
    if (verticesCount < 3) throw std::runtime_error("A Polyhedron needs at least 3 (rz) vertices");

    // Geant4-like construction (n = verticesCount). The rz section is described
    // as a sequence of connected vertices (r[i], z[i]). We have to associate
    // the vertices with (rmin, rmax, z) plane representation.

    // detect if vertices are defined clockwise
    Precision area = 0;
    for (int i = 0; i < verticesCount; ++i) {
      int j = (i + 1) % verticesCount;
      area += r[i] * z[j] - r[j] * z[i];
    }

    bool cw      = (area < 0);
    int inc      = cw ? -1 : 1;
    Precision zt = z[0];
    Precision zb = z[0];
    // Find min/max on Z
    for (int i = 0; i < verticesCount; ++i) {
      if (z[i] > zt) zt = z[i];
      if (z[i] < zb) zb = z[i];
    }

    // Add implicit vertices
    Precision *rnew    = new Precision[2 * verticesCount];
    Precision *znew    = new Precision[2 * verticesCount];
    int verticesCount1 = 0;
    for (int i0 = 0; i0 < verticesCount; ++i0) {
      rnew[verticesCount1]   = r[i0];
      znew[verticesCount1++] = z[i0];
      // Check if top/bottom vertex is singular
      if (vecCore::math::Abs(z[i0] - zt) < kTolerance || vecCore::math::Abs(z[i0] - zb) < kTolerance) {
        if (vecCore::math::Abs(z[i0] - z[(i0 + verticesCount - 1) % verticesCount]) > kTolerance &&
            vecCore::math::Abs(z[i0] - z[(i0 + 1) % verticesCount]) > kTolerance) {
          rnew[verticesCount1]   = r[i0];
          znew[verticesCount1++] = z[i0];
        }
      }
      int i1       = (i0 + 1) % verticesCount;
      Precision dz = z[i1] - z[i0];
      if (vecCore::math::Abs(dz) < kTolerance) continue;
      Precision zmin = vecCore::math::Min(z[i0], z[i1]);
      Precision zmax = vecCore::math::Max(z[i0], z[i1]);
      for (int j = 0; j < verticesCount - 2; ++j) {
        // go backward
        int k = (i0 - 1 - j + verticesCount) % verticesCount;
        if (z[k] > zmin + kTolerance && z[k] < zmax - kTolerance) {
          // Project the vertex on current segment to get a new vertex
          Precision rp = r[i0] + (r[i1] - r[i0]) * (z[k] - z[i0]) / dz;
          assert(rp >= 0);
          // We need to insert point (rp, z[k]) after i1
          rnew[verticesCount1]   = rp;
          znew[verticesCount1++] = z[k];
        }
      }
    }

    // detect index of outer vertex with minimum Z
    int i0 = -1;
    for (int i = 0; i < verticesCount1; ++i) {
      if (znew[i] == zb) {
        i0 = i;
        break;
      }
    }
    if (vecCore::math::Abs(zb - znew[(i0 + inc) % verticesCount1]) < kTolerance) i0 = (i0 + inc) % verticesCount1;

    if (phiDelta <= 0 || phiDelta > kTwoPi - kAngTolerance) phiDelta = kTwoPi;
    Precision sidePhi         = phiDelta / sideCount;
    Precision cosHalfDeltaPhi = cos(0.5 * sidePhi);

    // We count vertices starting from imin, making sure we move counter-clockwise

    int Nz          = verticesCount1 / 2;
    Precision *rMin = new Precision[Nz];
    Precision *rMax = new Precision[Nz];
    Precision *zArg = new Precision[Nz];

    for (int i = 0; i < Nz; ++i) {
      // Current vertex index going always ccw from (rmin,zmin)
      int j    = (i0 + verticesCount1 + inc * i) % verticesCount1;
      int jsim = (i0 + verticesCount1 + inc * (verticesCount1 - 1 - i)) % verticesCount1;
      assert(znew[j] == znew[jsim]);
      zArg[i] = znew[j];
      rMax[i] = rnew[j] * cosHalfDeltaPhi;
      rMin[i] = rnew[jsim] * cosHalfDeltaPhi;
      assert(rMax[i] >= rMin[i] &&
             "UnplPolycone ERROR: r[] provided has problems of the Rmax < Rmin type, please check!\n");
    }

    // Allocate arrays
    fZSegments.Allocate(Nz - 1);
    fZPlanes.Allocate(Nz);
    fRMin.Allocate(Nz);
    fRMax.Allocate(Nz);
    fSameZ.Allocate(Nz);

    // Delegate to full constructor
    Initialize(phiStart, phiDelta, sideCount, Nz, zArg, rMin, rMax);
    delete[] rnew;
    delete[] znew;
    delete[] rMin;
    delete[] rMax;
    delete[] zArg;
  }

  VECCORE_ATT_HOST_DEVICE
  ~PolyhedronStruct() { delete fAreaStruct; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static size_t aligned_sizeof_data(Precision /*phiStart*/, Precision phiDelta, const int sideCount,
                                    const int zPlaneCount, Precision const zPlanes[], Precision const rMin[],
                                    Precision const rMax[])
  {
    const bool hasPhiCutout = phiDelta < 2 * kPi;
    size_t aligned_size     = (zPlaneCount - 1) * sizeof(ZSegment);
    // Alignment of all Array data members, which are AlignedBase types
    aligned_size += 5 * kAlignmentBoundary;
    // fZsegments content
    for (int i = 0; i < zPlaneCount - 1; ++i) {
      // Z-planes must be monotonically increasing
      assert(zPlanes[i] <= zPlanes[i + 1]);
      bool hasInnerRadius = rMin[i] > kTolerance || rMin[i + 1] > kTolerance;
      int multiplier      = (ApproxEqual(zPlanes[i], zPlanes[i + 1]) && ApproxEqual(rMax[i], rMax[i + 1])) ? 0 : 1;
      aligned_size += Quadrilaterals::aligned_sizeof_data(sideCount * multiplier);
      // no phi segment here if degenerate z;
      if (hasPhiCutout) {
        multiplier = (zPlanes[i] == zPlanes[i + 1]) ? 0 : 1;
        aligned_size += Quadrilaterals::aligned_sizeof_data(2 * multiplier);
      }
      multiplier = (zPlanes[i] == zPlanes[i + 1] && rMin[i] == rMin[i + 1]) ? 0 : 1;
      if (hasInnerRadius && multiplier > 0) {
        aligned_size += Quadrilaterals::aligned_sizeof_data(sideCount * multiplier);
      }
    }
    // fZplanes, fRmin, fRmax
    aligned_size += 3 * Array<T>::aligned_sizeof_data(zPlaneCount);
    // fSameZ
    aligned_size += Array<bool>::aligned_sizeof_data(zPlaneCount);
    // fPhiSections
    aligned_size += SOA3D<T>::aligned_sizeof_data(sideCount + 1);
    return aligned_size;
  }

  VECCORE_ATT_HOST_DEVICE
  bool CheckContinuityInSlope(const Precision rOuter[], const Precision zPlane[], const unsigned int nz)
  {
    Precision prevSlope = kInfLength;
    for (unsigned int j = 0; j < nz - 1; ++j) {
      if (ApproxEqual(zPlane[j + 1], zPlane[j])) {
        if (!ApproxEqual(rOuter[j + 1], rOuter[j])) return false;
      } else {
        Precision currentSlope = (rOuter[j + 1] - rOuter[j]) / (zPlane[j + 1] - zPlane[j]);
        if (currentSlope > prevSlope) return false;
        prevSlope = currentSlope;
      }
    }
    return true;
  }

  VECCORE_ATT_HOST_DEVICE
  void Dump()
  {
    auto dump_planes = [](Planes const &planes) {
      auto const &normals   = planes.GetNormals();
      auto const &distances = planes.GetDistances();
      printf("[%d]: convex=%d normals[%p] distances[%p]", planes.size(), planes.IsConvex(), (void *)&normals,
             (void *)&distances);
      if (planes.size() == 0) printf(" : empty");
      printf("\n");
      for (size_t i = 0; i < planes.size(); ++i) {
        auto const &normal = normals[i];
        auto distance      = distances[i];
        printf("         [%lu]: fNormal { %g, %g, %g } fDistance %g\n", i, normal[0], normal[1], normal[2], distance);
      }
    };
    printf("== PolyhedronStruct at: %p", (void *)this);
    printf("   side count: %d  z_plane_count: %u phiStart: %g phiDelta: %g\n", fSideCount, fZPlanes.size(), fPhiStart,
           fPhiDelta);
    printf("   hasInnerRadii: %d hasPhiCutout: %d fHasLargePhiCutout: %d\n", fHasInnerRadii, fHasPhiCutout,
           fHasLargePhiCutout);
    for (size_t i = 0; i < fZSegments.size(); ++i) {
      printf("   ZSegments[%lu]:\n", i);
      auto const &zseg = fZSegments[i];
      printf("     outer\n");
      printf("       planes");
      dump_planes(zseg.outer.GetPlanes());
      for (size_t j = 0; j < 4; ++j) {
        printf("       side vectors[%lu]", j);
        dump_planes(zseg.outer.GetSideVectors()[j]);
        auto const &corners = zseg.outer.GetCorners()[j];
        if (corners.size()) printf("       corners[%lu]", j);
        for (size_t k = 0; k < corners.size(); ++k)
          printf(" %lu:{%g, %g, %g}", k, corners[k].x(), corners[k].y(), corners[k].z());
        if (corners.size()) printf("\n");
      }
      printf("     inner\n");
      printf("       planes");
      dump_planes(zseg.inner.GetPlanes());
      for (size_t j = 0; j < 4; ++j) {
        printf("       side vectors[%lu]", j);
        dump_planes(zseg.inner.GetSideVectors()[j]);
        auto const &corners = zseg.inner.GetCorners()[j];
        if (corners.size()) printf("       corners[%lu]", j);
        for (size_t k = 0; k < corners.size(); ++k)
          printf(" %lu:{%g, %g, %g}", k, corners[k].x(), corners[k].y(), corners[k].z());
        if (corners.size()) printf("\n");
      }
      printf("     phi\n");
      printf("       planes");
      dump_planes(zseg.phi.GetPlanes());
      for (size_t j = 0; j < 4; ++j) {
        printf("       side vectors[%lu]", j);
        dump_planes(zseg.phi.GetSideVectors()[j]);
        auto const &corners = zseg.phi.GetCorners()[j];
        if (corners.size()) printf("       corners[%lu]", j);
        for (size_t k = 0; k < corners.size(); ++k)
          printf(" %lu:{%g, %g, %g}", k, corners[k].x(), corners[k].y(), corners[k].z());
        if (corners.size()) printf("\n");
      }
    }
    printf("\n   fZPlanes: ");
    for (size_t i = 0; i < fZPlanes.size(); ++i)
      printf(" %lu: %g", i, fZPlanes[i]);
    printf("\n   fRMin: ");
    for (size_t i = 0; i < fRMin.size(); ++i)
      printf(" %lu: %g", i, fRMin[i]);
    printf("\n   fRMax: ");
    for (size_t i = 0; i < fRMax.size(); ++i)
      printf(" %lu: %g", i, fRMax[i]);
    printf("\n   fSameZ: ");
    for (size_t i = 0; i < fSameZ.size(); ++i)
      printf(" %lu: %d", i, fSameZ[i]);
    printf("\n   fPhiSections: ");
    for (size_t i = 0; i < fPhiSections.size(); ++i)
      printf(" %lu: {%g, %g, %g}", i, fPhiSections[i].x(), fPhiSections[i].y(), fPhiSections[i].z());
    printf("\n");
  }

  // This method does the proper construction of planes and segments.
  // Used by multiple constructors.
  VECCORE_ATT_HOST_DEVICE
  void Initialize(Precision phiStart, Precision phiDelta, const int sideCount, const int zPlaneCount,
                  Precision const zPlanes[], Precision const rMin[], Precision const rMax[])
  {
    typedef Vector3D<Precision> Vec_t;

    // Sanity check of input parameters
    assert(zPlaneCount > 1);
    assert(fSideCount > 0);
    fSize = PolyhedronStruct<T>::aligned_sizeof_data(phiStart, phiDelta, sideCount, zPlaneCount, zPlanes, rMin, rMax);

    for (auto i = 0; i < zPlaneCount; ++i) {
      fZPlanes[i] = 0.;
      fRMin[i]    = 0.;
      fRMax[i]    = 0.;
      fSameZ[i]   = false;
    }
    copy(zPlanes, zPlanes + zPlaneCount, &fZPlanes[0]);
    copy(rMin, rMin + zPlaneCount, &fRMin[0]);
    copy(rMax, rMax + zPlaneCount, &fRMax[0]);

    Precision startRmax = rMax[0];
    for (int i = 0; i < zPlaneCount; i++) {
      fConvexityPossible &= (rMin[i] < kTolerance);
      fEqualRmax &= (ApproxEqual(startRmax, rMax[i]));
      if (i > 0 && i < zPlaneCount - 2) {
        if (ApproxEqual(fZPlanes[i], fZPlanes[i + 1])) fSameZ[i] = true;
      }
    }
    fContinuousInSlope = CheckContinuityInSlope(rMax, zPlanes, zPlaneCount);

    // Initialize segments
    // sometimes there will be no quadrilaterals: for instance when
    // rmin jumps at some z and rmax remains continouus
    for (int i = 0; i < zPlaneCount - 1; ++i) {
      // Z-planes must be monotonically increasing
      assert(zPlanes[i] <= zPlanes[i + 1]);

      bool hasInnerRadius = rMin[i] > kTolerance || rMin[i + 1] > kTolerance;

      int multiplier = (ApproxEqual(zPlanes[i], zPlanes[i + 1]) && ApproxEqual(rMax[i], rMax[i + 1])) ? 0 : 1;

      // create quadrilaterals in a predefined place with placement new
      new (&fZSegments[i].outer) Quadrilaterals(sideCount * multiplier);

      // no phi segment here if degenerate z;
      if (fHasPhiCutout) {
        multiplier = (zPlanes[i] == zPlanes[i + 1]) ? 0 : 1;
        new (&fZSegments[i].phi) Quadrilaterals(2 * multiplier, phiDelta <= kPi);
      } else {
        new (&fZSegments[i].phi) Quadrilaterals(0);
      }

      multiplier = (zPlanes[i] == zPlanes[i + 1] && rMin[i] == rMin[i + 1]) ? 0 : 1;

      if (hasInnerRadius && multiplier > 0) {
        new (&fZSegments[i].inner) Quadrilaterals(sideCount * multiplier);
        fHasInnerRadii = true;
      } else {
        new (&fZSegments[i].inner) Quadrilaterals(0);
      }
    }

    // Compute the cylindrical coordinate phi along which the corners are placed
    if (phiDelta <= 0 || phiDelta > kTwoPi - kAngTolerance) phiDelta = kTwoPi;
    phiStart = NormalizeAngle<kScalar>(phiStart);
    if (phiDelta > kTwoPi) phiDelta = kTwoPi;
    Precision sidePhi = phiDelta / sideCount;

    auto getPhi = [&](int side) {
      if (!fHasPhiCutout && side == sideCount) {
        side = 0;
      }
      return NormalizeAngle<kScalar>(phiStart + side * sidePhi);
    };

    for (int i = 0, iMax = sideCount + 1; i < iMax; ++i) {
      Vector3D<Precision> cornerVector = Vec_t::FromCylindrical(1., getPhi(i), 0).Normalized().FixZeroes();
      fPhiSections.set(i, cornerVector.Normalized().Cross(Vector3D<Precision>(0, 0, -1)));
    }

    // Specified radii are to the sides, not to the corners. Change these values,
    // as corners and not sides are used to build the structure
    Precision cosHalfDeltaPhi = cos(0.5 * sidePhi);
    Precision innerRadius = kInfLength, outerRadius = -kInfLength;
    for (int i = 0; i < zPlaneCount; ++i) {
      // Use distance to side for minimizing inner radius of bounding tube
      if (rMin[i] < innerRadius) innerRadius = rMin[i];
      assert(rMin[i] >= 0 && rMax[i] >= 0);
      // Use distance to corner for minimizing outer radius of bounding tube
      if (rMax[i] > outerRadius) outerRadius = rMax[i];
    }
    // need to convert from distance to planes to real radius in case of outerradius
    // the inner radius of the bounding tube is given by min(rMin[])
    outerRadius /= cosHalfDeltaPhi;

    // Create bounding tube with biggest outer radius and smallest inner radius
    Precision boundingTubeZ = 0.5 * (zPlanes[zPlaneCount - 1] - zPlanes[0]) + kTolerance;
    // Make bounding tube phi range a bit larger to contain all points on phi boundaries
    const Precision kPhiTolerance = 100 * kTolerance;
    // The increase in the angle has to be large enough to contain most of
    // kSurface points. There will be some points close to the Z axis which will
    // not be contained. The value is empirical to satisfy ShapeTester
    Precision boundsPhiStart = !fHasPhiCutout ? 0 : phiStart - kPhiTolerance;
    Precision boundsPhiDelta = !fHasPhiCutout ? kTwoPi : phiDelta + 2 * kPhiTolerance;

    fBoundingTube = TubeStruct<Precision>(innerRadius - kHalfTolerance, outerRadius + kHalfTolerance, boundingTubeZ,
                                          boundsPhiStart, boundsPhiDelta);

    // The offset has to match the middle of the polyhedron
    fBoundingTubeOffset = 0.5 * (zPlanes[0] + zPlanes[zPlaneCount - 1]);

    auto getVertexImpl = [&](Precision const r[], int i, int j) {
      if (!fHasPhiCutout && j == sideCount) {
        j = 0;
      }
      return Vec_t::FromCylindrical(r[i] / cosHalfDeltaPhi, getPhi(j), zPlanes[i]).FixZeroes();
    };

    auto getInnerVertex = [&](int i, int j) { return getVertexImpl(rMin, i, j); };
    auto getOuterVertex = [&](int i, int j) { return getVertexImpl(rMax, i, j); };

    // Build segments by drawing quadrilaterals between vertices
    for (int iPlane = 0; iPlane < zPlaneCount - 1; ++iPlane) {

      auto WrongNormal = [](Vector3D<Precision> const &normal, Vector3D<Precision> const &corner) {
        return normal[0] * corner[0] + normal[1] * corner[1] < 0;
      };

      // Draw the regular quadrilaterals along phi
      for (int iSide = 0; iSide < fZSegments[iPlane].outer.size(); ++iSide) {
        fZSegments[iPlane].outer.Set(iSide, getOuterVertex(iPlane, iSide), getOuterVertex(iPlane, iSide + 1),
                                     getOuterVertex(iPlane + 1, iSide + 1), getOuterVertex(iPlane + 1, iSide));
        // Normal has to point away from Z-axis
        if (WrongNormal(fZSegments[iPlane].outer.GetNormal(iSide), getOuterVertex(iPlane, iSide))) {
          fZSegments[iPlane].outer.FlipSign(iSide);
        }
      }
      for (int iSide = 0; iSide < fZSegments[iPlane].inner.size(); ++iSide) {
        fZSegments[iPlane].inner.Set(iSide, getInnerVertex(iPlane, iSide), getInnerVertex(iPlane, iSide + 1),
                                     getInnerVertex(iPlane + 1, iSide + 1), getInnerVertex(iPlane + 1, iSide));
        // Normal has to point away from Z-axis
        if (WrongNormal(fZSegments[iPlane].inner.GetNormal(iSide), getInnerVertex(iPlane, iSide))) {
          fZSegments[iPlane].inner.FlipSign(iSide);
        }
      }

      if (fHasPhiCutout && fZSegments[iPlane].phi.size() == 2) {
        // If there's a phi cutout, draw two quadrilaterals connecting the four
        // corners (two inner, two outer) of the first and last phi coordinate,
        // respectively
        fZSegments[iPlane].phi.Set(0, getInnerVertex(iPlane, 0), getInnerVertex(iPlane + 1, 0),
                                   getOuterVertex(iPlane + 1, 0), getOuterVertex(iPlane, 0));
        // Make sure normal points backwards along phi
        if (fZSegments[iPlane].phi.GetNormal(0).Dot(fPhiSections[0]) > 0) {
          fZSegments[iPlane].phi.FlipSign(0);
        }
        fZSegments[iPlane].phi.Set(1, getOuterVertex(iPlane, sideCount), getOuterVertex(iPlane + 1, sideCount),
                                   getInnerVertex(iPlane + 1, sideCount), getInnerVertex(iPlane, sideCount));
        // Make sure normal points forwards along phi
        if (fZSegments[iPlane].phi.GetNormal(1).Dot(fPhiSections[fSideCount]) < 0) {
          fZSegments[iPlane].phi.FlipSign(1);
        }
      }

    } // End loop over segments
  }

  // This method does the proper construction of planes and segments.
  // Used by multiple constructors.
  VECCORE_ATT_HOST_DEVICE
  void Initialize(Precision phiStart, Precision phiDelta, const int sideCount, const int zPlaneCount,
                  Precision const zPlanes[], Precision const rMin[], Precision const rMax[], AlignedAllocator &a)
  {
    typedef Vector3D<Precision> Vec_t;

    // Sanity check of input parameters
    assert(zPlaneCount > 1);
    assert(fSideCount > 0);

    for (auto i = 0; i < zPlaneCount; ++i) {
      fZPlanes[i] = 0.;
      fRMin[i]    = 0.;
      fRMax[i]    = 0.;
      fSameZ[i]   = false;
    }
    copy(zPlanes, zPlanes + zPlaneCount, &fZPlanes[0]);
    copy(rMin, rMin + zPlaneCount, &fRMin[0]);
    copy(rMax, rMax + zPlaneCount, &fRMax[0]);

    Precision startRmax = rMax[0];
    for (int i = 0; i < zPlaneCount; i++) {
      fConvexityPossible &= (rMin[i] < kTolerance);
      fEqualRmax &= (ApproxEqual(startRmax, rMax[i]));
      if (i > 0 && i < zPlaneCount - 2) {
        if (ApproxEqual(fZPlanes[i], fZPlanes[i + 1])) fSameZ[i] = true;
      }
    }
    fContinuousInSlope = CheckContinuityInSlope(rMax, zPlanes, zPlaneCount);

    // Initialize segments
    // sometimes there will be no quadrilaterals: for instance when
    // rmin jumps at some z and rmax remains continouus
    for (int i = 0; i < zPlaneCount - 1; ++i) {
      // Z-planes must be monotonically increasing
      assert(zPlanes[i] <= zPlanes[i + 1]);

      bool hasInnerRadius = rMin[i] > kTolerance || rMin[i + 1] > kTolerance;
      bool convex         = phiDelta <= kPi;

      int multiplier = (ApproxEqual(zPlanes[i], zPlanes[i + 1]) && ApproxEqual(rMax[i], rMax[i + 1])) ? 0 : 1;
      size_t nouter  = sideCount * multiplier;

      // no phi segment here if degenerate z;
      size_t nphi = 0;
      if (fHasPhiCutout) {
        multiplier = (zPlanes[i] == zPlanes[i + 1]) ? 0 : 1;
        nphi       = 2 * multiplier;
      }

      multiplier    = (zPlanes[i] == zPlanes[i + 1] && rMin[i] == rMin[i + 1]) ? 0 : 1;
      size_t ninner = 0;

      if (hasInnerRadius && multiplier > 0) {
        ninner         = sideCount * multiplier;
        fHasInnerRadii = true;
      }

      // Create section
      new (&fZSegments[i]) ZSegment(nouter, ninner, nphi, a, convex);
    }

    // Compute the cylindrical coordinate phi along which the corners are placed
    if (phiDelta <= 0 || phiDelta > kTwoPi - kAngTolerance) phiDelta = kTwoPi;
    phiStart = NormalizeAngle<kScalar>(phiStart);
    if (phiDelta > kTwoPi) phiDelta = kTwoPi;
    Precision sidePhi = phiDelta / sideCount;

    auto getPhi = [&](int side) {
      if (!fHasPhiCutout && side == sideCount) {
        side = 0;
      }
      return NormalizeAngle<kScalar>(phiStart + side * sidePhi);
    };

    for (int i = 0, iMax = sideCount + 1; i < iMax; ++i) {
      Vector3D<Precision> cornerVector = Vec_t::FromCylindrical(1., getPhi(i), 0).Normalized().FixZeroes();
      fPhiSections.set(i, cornerVector.Normalized().Cross(Vector3D<Precision>(0, 0, -1)));
    }

    // Specified radii are to the sides, not to the corners. Change these values,
    // as corners and not sides are used to build the structure
    Precision cosHalfDeltaPhi = cos(0.5 * sidePhi);
    Precision innerRadius = kInfLength, outerRadius = -kInfLength;
    for (int i = 0; i < zPlaneCount; ++i) {
      // Use distance to side for minimizing inner radius of bounding tube
      if (rMin[i] < innerRadius) innerRadius = rMin[i];
      assert(rMin[i] >= 0 && rMax[i] >= 0);
      // Use distance to corner for minimizing outer radius of bounding tube
      if (rMax[i] > outerRadius) outerRadius = rMax[i];
    }
    // need to convert from distance to planes to real radius in case of outerradius
    // the inner radius of the bounding tube is given by min(rMin[])
    outerRadius /= cosHalfDeltaPhi;

    // Create bounding tube with biggest outer radius and smallest inner radius
    Precision boundingTubeZ = 0.5 * (zPlanes[zPlaneCount - 1] - zPlanes[0]) + kTolerance;
    // Make bounding tube phi range a bit larger to contain all points on phi boundaries
    const Precision kPhiTolerance = 100 * kTolerance;
    // The increase in the angle has to be large enough to contain most of
    // kSurface points. There will be some points close to the Z axis which will
    // not be contained. The value is empirical to satisfy ShapeTester
    Precision boundsPhiStart = !fHasPhiCutout ? 0 : phiStart - kPhiTolerance;
    Precision boundsPhiDelta = !fHasPhiCutout ? kTwoPi : phiDelta + 2 * kPhiTolerance;

    fBoundingTube = TubeStruct<Precision>(innerRadius - kHalfTolerance, outerRadius + kHalfTolerance, boundingTubeZ,
                                          boundsPhiStart, boundsPhiDelta);

    // The offset has to match the middle of the polyhedron
    fBoundingTubeOffset = 0.5 * (zPlanes[0] + zPlanes[zPlaneCount - 1]);

    auto getVertexImpl = [&](Precision const r[], int i, int j) {
      if (!fHasPhiCutout && j == sideCount) {
        j = 0;
      }
      return Vec_t::FromCylindrical(r[i] / cosHalfDeltaPhi, getPhi(j), zPlanes[i]).FixZeroes();
    };

    auto getInnerVertex = [&](int i, int j) { return getVertexImpl(rMin, i, j); };
    auto getOuterVertex = [&](int i, int j) { return getVertexImpl(rMax, i, j); };

    // Build segments by drawing quadrilaterals between vertices
    for (int iPlane = 0; iPlane < zPlaneCount - 1; ++iPlane) {

      auto WrongNormal = [](Vector3D<Precision> const &normal, Vector3D<Precision> const &corner) {
        return normal[0] * corner[0] + normal[1] * corner[1] < 0;
      };

      // Draw the regular quadrilaterals along phi
      for (int iSide = 0; iSide < fZSegments[iPlane].outer.size(); ++iSide) {
        fZSegments[iPlane].outer.Set(iSide, getOuterVertex(iPlane, iSide), getOuterVertex(iPlane, iSide + 1),
                                     getOuterVertex(iPlane + 1, iSide + 1), getOuterVertex(iPlane + 1, iSide));
        // Normal has to point away from Z-axis
        if (WrongNormal(fZSegments[iPlane].outer.GetNormal(iSide), getOuterVertex(iPlane, iSide))) {
          fZSegments[iPlane].outer.FlipSign(iSide);
        }
      }
      for (int iSide = 0; iSide < fZSegments[iPlane].inner.size(); ++iSide) {
        fZSegments[iPlane].inner.Set(iSide, getInnerVertex(iPlane, iSide), getInnerVertex(iPlane, iSide + 1),
                                     getInnerVertex(iPlane + 1, iSide + 1), getInnerVertex(iPlane + 1, iSide));
        // Normal has to point away from Z-axis
        if (WrongNormal(fZSegments[iPlane].inner.GetNormal(iSide), getInnerVertex(iPlane, iSide))) {
          fZSegments[iPlane].inner.FlipSign(iSide);
        }
      }

      if (fHasPhiCutout && fZSegments[iPlane].phi.size() == 2) {
        // If there's a phi cutout, draw two quadrilaterals connecting the four
        // corners (two inner, two outer) of the first and last phi coordinate,
        // respectively
        fZSegments[iPlane].phi.Set(0, getInnerVertex(iPlane, 0), getInnerVertex(iPlane + 1, 0),
                                   getOuterVertex(iPlane + 1, 0), getOuterVertex(iPlane, 0));
        // Make sure normal points backwards along phi
        if (fZSegments[iPlane].phi.GetNormal(0).Dot(fPhiSections[0]) > 0) {
          fZSegments[iPlane].phi.FlipSign(0);
        }
        fZSegments[iPlane].phi.Set(1, getOuterVertex(iPlane, sideCount), getOuterVertex(iPlane + 1, sideCount),
                                   getInnerVertex(iPlane + 1, sideCount), getInnerVertex(iPlane, sideCount));
        // Make sure normal points forwards along phi
        if (fZSegments[iPlane].phi.GetNormal(1).Dot(fPhiSections[fSideCount]) < 0) {
          fZSegments[iPlane].phi.FlipSign(1);
        }
      }

    } // End loop over segments
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
