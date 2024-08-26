#ifndef VECGEOM_SURFACE_COMMONTYPES_H
#define VECGEOM_SURFACE_COMMONTYPES_H

#include <cassert>
#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/base/Transformation3DMP.h>
#include <VecGeom/base/Vector2D.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/volumes/kernel/GenericKernels.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/management/Logger.h>

namespace vgbrep {

///< VecGeom type aliases
template <typename Real_t>
using Vector3D = vecgeom::Vector3D<Real_t>;

template <typename Real_t>
using Vector2D       = vecgeom::Vector2D<Real_t>;
using Transformation = vecgeom::Transformation3D;
template <typename Real_t>
using TransformationMP = vecgeom::Transformation3DMP<Real_t>;

using logic_int = int;

/// @brief Operators used inside Boolean expressions
enum OperatorToken : logic_int {
  lbegin = logic_int(0x7FFFFFFF - 8),
  lfalse = lbegin, ///< Push 'false'
  ltrue,           ///< Push 'true'
  lplus,           ///< increase logical depth
  lminus,          ///< decrease logical depth
  lnot,            ///< Unary negation
  lor,             ///< Binary logical OR
  land,            ///< Binary logical AND
  lend
};

struct LogicExpression {
  logic_int size_;
  logic_int *data_{nullptr};

  static VECCORE_ATT_HOST_DEVICE bool is_operator_token(logic_int lv) { return (lv >= lbegin); }

  VECCORE_ATT_HOST_DEVICE
  unsigned size() const { return (unsigned)size_; }

  VECCORE_ATT_HOST_DEVICE
  logic_int operator[](unsigned i) const { return data_[i]; }
};

///< Supported surface types
///< kPlanar        <- planar (xOy) half-space having the normal along z direction
///< kCylindrical   <- cylindrical half-space around the z-axis, having the normals pointing outwards
///< kConical       <- conical half-space around z-axis, having the normals pointing outwards
///< kSpherical     <- sphere centered in origin, normals pointing outwards
///< kTorus         <- toroidal surface hafing the median circle in the xy plane, cenetred in origin
///< kElliptical    <- elliptical half-space around the z-axis, having the normals pointing outwards
///< kArb4          <- twisted surface defined by 4 non co-planar vertices
enum class SurfaceType : char { kNoSurf, kPlanar, kCylindrical, kConical, kSpherical, kTorus, kElliptical, kArb4 };

///< Supported frame types
///< kNoFrame       <- no frame, used for Inside only
///< kRangeZ        <- range along z-axis
///< kRing          <- a "ring" range on a plane
///< kZPhi          <- z and phi range on a cylinder
///< kRangeSph      <- theta and phi range on a sphere
///< kWindow        <- rectangular range in xy-plane
///< kTriangle      <- triangular range in xy-plane
///< kQuadrilateral <- planar quadrilateral in xy-plane
enum class FrameType : char { kNoFrame, kRangeZ, kRing, kZPhi, kRangeSph, kWindow, kTriangle, kQuadrilateral };

///< Segment-segment intersection types
enum class SegmentIntersect : char { kNoIntersect, kEmbedding, kEmbedded, kOverlap, kIntersect, kEqual };

/// @brief A list of candidate surfaces
struct Candidates {
  int fNcand{0};             ///< Number of candidate surfaces
  int fNExiting{0};          ///< Number of exiting candidate surfaces. fNcand = NEntering + NExiting
  int *fCandidates{nullptr}; ///< [fNcand] Array of candidates
  int *fFrameInd{nullptr};   ///< [fNcand] Start index of the frame contributed by the touchable on the common surface
  char *fSides{nullptr};     ///< [fNcand] Side containing relevant frames for each candidate (0=left, 1=right, 2=both)

  VECCORE_ATT_HOST_DEVICE
  int operator[](int i) const { return fCandidates[i]; }
  VECCORE_ATT_HOST_DEVICE
  int operator[](int i) { return fCandidates[i]; }

  Candidates() = default;
};

/// @brief Framed surface locator
struct FSlocator {
  int common_id{0};               ///< Common surface id (positive = left side, negative = right side)
  int frame_id{-1};               ///< frame index on the side
  vecgeom::NavigationState state; ///< full state associated to the frame

  FSlocator() = default;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  FSlocator(int csind, int iframe, bool left) { Set(csind, iframe, left); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Set(int csind, int iframe, bool left)
  {
    common_id = (2 * int(left) - 1) * csind;
    frame_id  = iframe;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetCSindex() const { return vecCore::math::Abs(common_id); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsLeftSide() const { return common_id > 0; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetFSindex() const { return frame_id; }
};

/// @brief Exit surf data, contains all relevant surface information
struct CrossedSurface {
  FSlocator exit_surf; // containing the surface information of the highest exited frame
  FSlocator hit_surf;  // containing the surface data of the last frame hit (exited or entered)

  // Default constructor
  CrossedSurface() = default;

  // Constructor with initialization
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  CrossedSurface(int csind1, int iframe1, bool left1, int csind2, int iframe2, bool left2)
      : exit_surf(csind1, iframe1, left1), hit_surf(csind2, iframe2, left2)
  {
  }

  // Method to set both FSlocators
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Set(int csind1, int iframe1, bool left1, int csind2, int iframe2, bool left2)
  {
    exit_surf.Set(csind1, iframe1, left1);
    hit_surf.Set(csind2, iframe2, left2);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Set(int csind, int iframe, bool left)
  {
    exit_surf.Set(csind, iframe, left);
    hit_surf.Set(csind, iframe, left);
  }
};

struct SliceCand {
  int fNcand{0};             ///< Number of candidate frames in the slice
  int *fCandidates{nullptr}; ///< [fNcand] Array of candidates
};

enum AxisType : char {
  kX,     ///< X axis
  kY,     ///< Y axis
  kZ,     ///< Z axis
  kR,     ///< Radial axis
  kPhi,   ///< Phi axis
  kXY,    ///< XY grid
  kNoAxis ///< No axis
};

/// @brief Helper for dividing a side in equal slices along one axis, keeping frame candidates in each slice
/// @details The range given by the side extent is divided in equal slices along an axis. Each
///          slice intersects a number of frames. The slice is found besed on the crossing point, then the full
///          frame loops are reduced to the list of candidates in that slice.
template <typename Real_t>
struct SideDivision {
  Real_t fStartU{0.};                ///< Division start on primary division axis
  Real_t fStepU{0.};                 ///< Division step on primary division axis
  Real_t fStartV{0.};                ///< Division start on secondary axis (if any)
  Real_t fStepV{0.};                 ///< Division step on secondary axis (if any)
  unsigned short fNslices{0};        ///< Total number of slices
  unsigned short fNslicesU{0};       ///< Number of slices on primary division axis
  unsigned short fNslicesV{0};       ///< Number of slices on secondary division axis
  AxisType fAxis{AxisType::kNoAxis}; ///< Division axis
  SliceCand *fSlices{nullptr};       ///< Array of slices

  // Methods
  SideDivision() = default;

  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE int GetSliceIndex(Vector3D<Real_i> const &onsurf) const
  {
    int ind = -1;
    switch (fAxis) {
    case kXY: {
      int indU = (onsurf[0] - fStartU) / fStepU;
      int indV = (onsurf[1] - fStartV) / fStepV;
      ind      = (indU >= 0 && indV >= 0 && indU < fNslicesU && indV < fNslicesV) ? fNslicesV * indU + indV : -1;
      return ind;
    }
    case AxisType::kR: {
      ind = (onsurf.Perp() - fStartU) / fStepU;
      return (ind < fNslices) ? ind : -1;
    }
    case AxisType::kPhi: {
      ind = (onsurf.Phi() - fStartU) / fStepU;
      return (ind >= 0 && ind < fNslices) ? ind : -1;
    }
    default:
      ind = (onsurf[fAxis] - fStartU) / fStepU;
      return (ind >= 0 && ind < fNslices) ? ind : -1;
    }
  }
};

// Aliases for different usages of Vec2D.
template <typename Real_t>
using Range = Vector2D<Real_t>;

template <typename Real_t>
using AngleVector = Vector2D<Real_t>;

template <typename Real_t>
using Point2D = Vector2D<Real_t>;

template <typename Real_t>
struct Circle2D {
  Real_t fR;          ///< Circle radius
  Point2D<Real_t> fC; ///< Center
  Circle2D(Real_t r, Point2D<Real_t> const &center) : fR{r}, fC{center} {}
};

template <typename Real_t>
struct Segment2D {
  Point2D<Real_t> fP1;
  Point2D<Real_t> fP2;
  Vector2D<Real_t> fV;
  bool fDegen{false};

  Segment2D(Point2D<Real_t> const &p1, Point2D<Real_t> const &p2)
  {
    fP1    = p1;
    fP2    = p2;
    fV     = fP2 - fP1;
    fDegen = fV.Mag2() < vecgeom::kToleranceDistSquared<Real_t>;
  }

  /// @brief Segment intersection with a circle.
  /// @param circle Circle to check against
  /// @return Intersection result. Can be kEmbedded, kIntersect or kNoIntersect
  SegmentIntersect Intersect(Circle2D<Real_t> const &circle)
  {
    Vector2D<Real_t> v1 = circle.fC - fP1;
    Vector2D<Real_t> v2 = circle.fC - fP2;
    bool inside1        = false;
    bool inside2        = false;
    if (v1.Dot(fV) < vecgeom::MakePlusTolerant<true, Real_t>(0) ||
        v2.Dot(fV) > vecgeom::MakeMinusTolerant<true, Real_t>(0)) {
      // One of the segment points is the closest to the circle center (includes the degen case)
      // Check the inside of each point and compare them
      inside1 = v1.Mag() < vecgeom::MakeMinusTolerant<true, Real_t>(circle.fR);
      inside2 = v2.Mag() < vecgeom::MakeMinusTolerant<true, Real_t>(circle.fR);
      if (inside1 && inside2) return SegmentIntersect::kEmbedded;
      if (inside1 ^ inside2) return SegmentIntersect::kIntersect;
      return SegmentIntersect::kNoIntersect;
    }
    // There is a point on the segment closest to the center (no degen here)
    auto saf = std::abs(v1.CrossZ(fV)) / fV.Mag();
    if (saf > vecgeom::MakeMinusTolerant<true, Real_t>(circle.fR)) return SegmentIntersect::kNoIntersect;
    // The closest point is inside the circle
    inside1 = v1.Mag() < vecgeom::MakeMinusTolerant<true, Real_t>(circle.fR);
    inside2 = v2.Mag() < vecgeom::MakeMinusTolerant<true, Real_t>(circle.fR);
    if (inside1 && inside2) return SegmentIntersect::kEmbedded;
    return SegmentIntersect::kIntersect;
  }

  bool BoundingBoxOverlap(Segment2D<Real_t> const &other)
  {
    Real_t min_x1 = std::min(fP1.x(), fP2.x());
    Real_t max_x1 = std::max(fP1.x(), fP2.x());
    Real_t min_y1 = std::min(fP1.y(), fP2.y());
    Real_t max_y1 = std::max(fP1.y(), fP2.y());

    Real_t min_x2 = std::min(other.fP1.x(), other.fP2.x());
    Real_t max_x2 = std::max(other.fP1.x(), other.fP2.x());
    Real_t min_y2 = std::min(other.fP1.y(), other.fP2.y());
    Real_t max_y2 = std::max(other.fP1.y(), other.fP2.y());

    return !(max_x1 < min_x2 || max_x2 < min_x1 || max_y1 < min_y2 || max_y2 < min_y1);
  }

  /// @brief Intersection with another segment
  /// @param other Other segment
  /// @return Intersection type
  SegmentIntersect Intersect(Segment2D<Real_t> const &other)
  {
    // Return kNoIntersect if one of the segments is degenerated
    if (fDegen || other.fDegen) return SegmentIntersect::kNoIntersect;

    // Perform bounding box overlap check
    if (!BoundingBoxOverlap(other)) return SegmentIntersect::kNoIntersect;

    auto v12  = other.fP2 - fP1;
    auto s    = v12.CrossZ(other.fV);
    auto t    = v12.CrossZ(fV);
    auto norm = fV.CrossZ(other.fV);
    if (std::abs(norm) < vecgeom::kToleranceDist<Real_t>) {
      // the segments are parallel
      if (std::abs(s) < vecgeom::kToleranceDist<Real_t>) {
        // the segments are colinear
        auto inv_vsq = Real_t(1) / fV.Dot(fV);
        auto t1      = inv_vsq * fV.Dot(other.fP1 - fP1);
        auto t2      = inv_vsq * fV.Dot(other.fP2 - fP1);
        if (t1 > t2) std::swap(t1, t2);
        if (t1 > vecgeom::MakeMinusTolerant<true, Real_t>(1) || t2 < vecgeom::MakePlusTolerant<true, Real_t>(0))
          return SegmentIntersect::kNoIntersect;
        if (std::abs(t1) < vecgeom::kToleranceDist<Real_t> &&
            std::abs(t2 - Real_t(1)) < vecgeom::kToleranceDist<Real_t>)
          return SegmentIntersect::kEqual;
        if (t1 > vecgeom::MakeMinusTolerant<true, Real_t>(0) && t2 < vecgeom::MakePlusTolerant<true, Real_t>(1))
          return SegmentIntersect::kEmbedding;
        if (t1 < vecgeom::MakePlusTolerant<true, Real_t>(0) && t2 > vecgeom::MakeMinusTolerant<true, Real_t>(1))
          return SegmentIntersect::kEmbedded;
        return SegmentIntersect::kOverlap;
      } else
        // The segments are parallel
        return SegmentIntersect::kNoIntersect;
    }

    // Calculate the intersection points using both segments
    auto intersectionPoint1 = fP1 + s * fV;
    auto intersectionPoint2 = other.fP1 + t * other.fV;

    auto PointInSegmentBounds = [](const Vector2D<Real_t> &point, const Segment2D<Real_t> &seg) {
      return (point.x() >= vecgeom::MakeMinusTolerant<true, Real_t>(std::min(seg.fP1.x(), seg.fP2.x())) &&
              point.x() <= vecgeom::MakePlusTolerant<true, Real_t>(std::max(seg.fP1.x(), seg.fP2.x())) &&
              point.y() >= vecgeom::MakeMinusTolerant<true, Real_t>(std::min(seg.fP1.y(), seg.fP2.y())) &&
              point.y() <= vecgeom::MakePlusTolerant<true, Real_t>(std::max(seg.fP1.y(), seg.fP2.y())));
    };

    // Check if both intersection points are within the bounds of the respective segments
    bool isWithinBounds1 =
        PointInSegmentBounds(intersectionPoint1, *this) && PointInSegmentBounds(intersectionPoint1, other);
    bool isWithinBounds2 =
        PointInSegmentBounds(intersectionPoint2, *this) && PointInSegmentBounds(intersectionPoint2, other);

    if ((isWithinBounds1 != isWithinBounds2)) {
      VECGEOM_LOG(warning) << " Check for intersection points using segment 1 and segment 2 differ. This should really "
                              "not be happening! Check for numerical problems.";
    }

    // If both points are within bounds, consider it an intersection
    if (isWithinBounds1 && isWithinBounds2) {
      return SegmentIntersect::kIntersect;
    }

    // No intersection in range
    return SegmentIntersect::kNoIntersect;
  }
};

/// @brief Data for conical surfaces
/// @tparam Real_t Storage type
/// @tparam Real_s Interface type
template <typename Real_t>
struct EllipData {
  Real_t Rx{0}; ///< semi-axis in x
  Real_t Ry{0}; ///< semi-axis in y
  Real_t dz{0}; ///< half height of the elliptical tube, so it extends from -dz to dz
  Real_t R{0};  ///< radius after rescaling to circle

  // Precalculated cached values
  //
  Real_t fRsph; ///< Radius of bounding sphere
  Real_t fDDx;  ///< Dx squared
  Real_t fDDy;  ///< Dy squared
  Real_t fSx;   ///< X scale factor
  Real_t fSy;   ///< Y scale factor
  Real_t fQ1;   ///< Coefficient in the approximation of dist = Q1*(x^2+y^2) - Q2
  Real_t fQ2;   ///< Coefficient in the approximation of dist = Q1*(x^2+y^2) - Q2

  EllipData() = default;
  /// @tparam Real_i precision type of inputs
  template <typename Real_i>
  EllipData(Real_i radx, Real_i rady, Real_i half_z)
      : Rx(static_cast<Real_t>(radx)), Ry(static_cast<Real_t>(rady)), dz(static_cast<Real_t>(half_z))
  {

    R = vecCore::math::Min(Rx, Ry);

    // Precalculated values
    fRsph = vecCore::math::Sqrt(Rx * Rx + Ry * Ry + dz * dz);
    fDDx  = Rx * Rx;
    fDDy  = Ry * Ry;
    fSx   = R / Rx;
    fSy   = R / Ry;

    // Coefficient for approximation of distance : Q1 * (x^2 + y^2) - Q2
    fQ1 = 0.5 / R;
    fQ2 = 0.5 * (R + vecgeom::kHalfTolerance * vecgeom::kHalfTolerance / R);
  }
  template <typename Real_i>
  EllipData(const EllipData<Real_i> &other)
      : Rx(static_cast<Real_t>(other.Rx)), Ry(static_cast<Real_t>(other.Ry)), dz(static_cast<Real_t>(other.dz)),
        R(static_cast<Real_t>(other.R)), fRsph(static_cast<Real_t>(other.fRsph)), fDDx(static_cast<Real_t>(other.fDDx)),
        fDDy(static_cast<Real_t>(other.fDDy)), fSx(static_cast<Real_t>(other.fSx)), fSy(static_cast<Real_t>(other.fSy)),
        fQ1(static_cast<Real_t>(other.fQ1)), fQ2(static_cast<Real_t>(other.fQ2))
  {
  }
};

/// @brief Data for Arb4 surfaces
/// @tparam Real_t Storage type
/// @tparam Real_s Interface type
template <typename Real_t>
struct Arb4Data {
  using Vector3D = vecgeom::Vector3D<Real_t>;

  Real_t verticesX[4];
  Real_t verticesY[4];
  Real_t connecting_compX[2];
  Real_t connecting_compY[2];
  Real_t halfH;                  ///< half the height of the Arb4
  Real_t halfH_inv;              ///< inverse of half the height
  Real_t ftx1, fty1, ftx2, fty2; /** Connecting components top-bottom */

  Real_t ft1crosst2; /** Cross term ftx1[i]*fty2[i] - ftx2[i]*fty1[i] */
  Real_t fDeltatx;   /** Term ftx2[i] - ftx1[i] */
  Real_t fDeltaty;   /** Term fty2[i] - fty1[i] */

  // pre-computed cross products for normal computation
  Vector3D fViCrossHi0;  /** Pre-computed vi X hi0 */
  Vector3D fViCrossVj;   /** Pre-computed vi X vj */
  Vector3D fHi1CrossHi0; /** Pre-computed hi1 X hi0 */

#ifdef SURF_ACCURATE_SAFETY
  // pre-computed normals of the plane using 3 of the 4 points of the Arb4 for additional safety calcuation
  Vector3D normal0;
  Vector3D normal1;
  Vector3D normal2;
  Vector3D normal3;
#endif

  Arb4Data() = default;
  /// @tparam Real_i precision type of inputs
  template <typename Real_i>
  Arb4Data(Real_i v0_0, Real_i v0_1, Real_i v0_2, Real_i v1_0, Real_i v1_1, Real_i v2_0, Real_i v2_1, Real_i v2_2,
           Real_i v3_0, Real_i v3_1)
      : halfH(0.5 * static_cast<Real_t>(v2_2 - v0_2)), halfH_inv(1. / halfH)
  {

    verticesX[0] = static_cast<Real_t>(v0_0);
    verticesX[1] = static_cast<Real_t>(v1_0);
    verticesX[2] = static_cast<Real_t>(
        v3_0); // note the flip here to stick to the convention of GenTrapImplementation in the solid model
    verticesX[3] = static_cast<Real_t>(v2_0);
    verticesY[0] = static_cast<Real_t>(v0_1);
    verticesY[1] = static_cast<Real_t>(v1_1);
    verticesY[2] = static_cast<Real_t>(v3_1);
    verticesY[3] = static_cast<Real_t>(v2_1);

    for (int i = 0; i < 2; ++i) {
      connecting_compX[i] = verticesX[i] - verticesX[i + 2];
      connecting_compY[i] = verticesY[i] - verticesY[i + 2];
    }

    ftx1 = 0.5 * halfH_inv * -connecting_compX[1];
    fty1 = 0.5 * halfH_inv * -connecting_compY[1];
    ftx2 = 0.5 * halfH_inv * -connecting_compX[0];
    fty2 = 0.5 * halfH_inv * -connecting_compY[0];

    ft1crosst2 = ftx1 * fty2 - ftx2 * fty1;
    fDeltatx   = ftx2 - ftx1;
    fDeltaty   = fty2 - fty1;

    // temporary vertices
    Vector3D va = {verticesX[1], verticesY[1], -halfH};
    Vector3D vb = {verticesX[3], verticesY[3], halfH};
    Vector3D vc = {verticesX[0], verticesY[0], -halfH};
    Vector3D vd = {verticesX[2], verticesY[2], halfH};

    // Cross products used for normal computation
    fViCrossHi0  = (vb - va).Cross(vc - va);
    fViCrossVj   = (vb - va).Cross(vd - vc);
    fHi1CrossHi0 = (vd - vb).Cross(vc - va);

#ifdef SURF_ACCURATE_SAFETY
    normal0 = (vd - vc).Cross(va - vc);
    normal0.Normalize();
    if ((vb - vc).Dot(normal0) < 0) normal0 *= -1;

    normal1 = (vb - va).Cross(vc - va);
    normal1.Normalize();
    if ((vd - va).Dot(normal1) < 0) normal1 *= -1;

    normal2 = (vc - vd).Cross(vb - vd);
    normal2.Normalize();
    if ((va - vd).Dot(normal2) < 0) normal2 *= -1;

    normal3 = (vd - vb).Cross(vc - vb);
    normal3.Normalize();
    if ((vc - vb).Dot(normal3) < 0) normal3 *= -1;
#endif
  }
  template <typename Real_i>
  Arb4Data(const Arb4Data<Real_i> &other)
      : halfH(static_cast<Real_t>(other.halfH)), halfH_inv(static_cast<Real_t>(other.halfH_inv)),
        ftx1(static_cast<Real_t>(other.ftx1)), fty1(static_cast<Real_t>(other.fty1)),
        ftx2(static_cast<Real_t>(other.ftx2)), fty2(static_cast<Real_t>(other.fty2)),
        ft1crosst2(static_cast<Real_t>(other.ft1crosst2)), fDeltatx(static_cast<Real_t>(other.fDeltatx)),
        fDeltaty(static_cast<Real_t>(other.fDeltaty)),
        fViCrossHi0(Vector3D(static_cast<Real_t>(other.fViCrossHi0[0]), static_cast<Real_t>(other.fViCrossHi0[1]),
                             static_cast<Real_t>(other.fViCrossHi0[2]))),
        fViCrossVj(Vector3D(static_cast<Real_t>(other.fViCrossVj[0]), static_cast<Real_t>(other.fViCrossVj[1]),
                            static_cast<Real_t>(other.fViCrossVj[2]))),
        fHi1CrossHi0(Vector3D(static_cast<Real_t>(other.fHi1CrossHi0[0]), static_cast<Real_t>(other.fHi1CrossHi0[1]),
                              static_cast<Real_t>(other.fHi1CrossHi0[2])))
#ifdef SURF_ACCURATE_SAFETY
        ,
        normal0(Vector3D(static_cast<Real_t>(other.normal0[0]), static_cast<Real_t>(other.normal0[1]),
                         static_cast<Real_t>(other.normal0[2]))),
        normal1(Vector3D(static_cast<Real_t>(other.normal1[0]), static_cast<Real_t>(other.normal1[1]),
                         static_cast<Real_t>(other.normal1[2]))),
        normal2(Vector3D(static_cast<Real_t>(other.normal2[0]), static_cast<Real_t>(other.normal2[1]),
                         static_cast<Real_t>(other.normal2[2]))),
        normal3(Vector3D(static_cast<Real_t>(other.normal3[0]), static_cast<Real_t>(other.normal3[1]),
                         static_cast<Real_t>(other.normal3[2])))

#endif
  {
    for (int i = 0; i < 4; ++i) {
      verticesX[i] = static_cast<Real_t>(other.verticesX[i]);
      verticesY[i] = static_cast<Real_t>(other.verticesY[i]);
    }

    for (int i = 0; i < 2; ++i) {
      connecting_compX[i] = static_cast<Real_t>(other.connecting_compX[i]);
      connecting_compY[i] = static_cast<Real_t>(other.connecting_compY[i]);
    }
  }
};

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool ApproxEqual(Real_t t1, Real_t t2)
{
  return std::abs(t1 - t2) <= vecgeom::kToleranceDist<Real_t>;
}

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool ApproxEqualVector(Vector3D<Real_t> const &v1, Vector3D<Real_t> const &v2)
{
  return ApproxEqual(v1[0], v2[0]) && ApproxEqual(v1[1], v2[1]) && ApproxEqual(v1[2], v2[2]);
}

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool ApproxEqualVector2(Vector2D<Real_t> const &v1, Vector2D<Real_t> const &v2)
{
  return ApproxEqual(v1[0], v2[0]) && ApproxEqual(v1[1], v2[1]);
}

template <typename Real_t>
bool ApproxEqualTransformation(const vecgeom::Transformation3DMP<Real_t> &t1,
                               const vecgeom::Transformation3DMP<Real_t> &t2)
{
  if (!ApproxEqualVector(t1.Translation(), t2.Translation())) return false;
  for (int i = 0; i < 9; ++i)
    if (!ApproxEqual(t1.Rotation(i), t2.Rotation(i))) return false;
  return true;
}

/// @brief Truncate a value to as many significant digits as a given tolerance.
///  For example, if the tolerance is 1e-9, truncate the vlaue to 9 significant digits
/// @tparam Real_t Precision type
/// @param x Value to truncate
/// @param tolerance Tolerance with as many significant digits as the truncation result
/// @return Truncated value
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t TruncateValue(Real_t x, Real_t tolerance = vecgeom::kToleranceDist<Real_t>())
{
  auto div = std::abs(x);
  while (int(div) > 0) {
    div /= 10;
    tolerance *= 10;
  }
  return tolerance * std::lround(x / tolerance);
}

/// @brief Return the rounding error of a value, assuming as many significant digits as a provided tolerance
/// @tparam Real_t Precision type
/// @param x Value subject to rounding
/// @param tolerance Tolerance with as many significant digits as the truncation result
/// @return Truncation error
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t RoundingError(Real_t x, Real_t tolerance = vecgeom::kToleranceDist<Real_t>())
{
  auto div = std::abs(x);
  while (int(div) > 0) {
    div /= 10;
    tolerance *= 10;
  }
  return tolerance;
}
/// @brief Data for torus surfaces
/// @tparam Real_t Storage type
/// @tparam Real_s Interface type
template <typename Real_t>
struct TorusData {
  Real_t rTor{0};  ///< radius to the center the torus.
  Real_t rTube{0}; ///< radius of the tube around the torus center, if negative, the torus is flipped
  AngleVector<Real_t> vecSPhi{vecCore::NumericLimits<Real_t>::Max(),
                              vecCore::NumericLimits<Real_t>::Max()}; ///< Cartesian coordinates of vectors that
                                                                      ///< represents the start of the phi-cut.
  AngleVector<Real_t> vecEPhi{vecCore::NumericLimits<Real_t>::Max(),
                              vecCore::NumericLimits<Real_t>::Max()}; ///< Cartesian coordinates of vectors that
                                                                      ///< represents the end of the phi-cut.
  Real_t inner_BC_radius{0}; ///< Cylindrical data for the inner bouding cylinder, normalized to rTor
  Real_t outer_BC_radius{0}; ///< Cylindrical data for the outer bouding cylinder, normalized to rTor

  /// @brief Check if local point is in the phi range
  /// @param local Point in local coordinates
  /// @return Point inside phi
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool InsidePhi(Vector3D<Real_t> const &local, bool flip) const
  {
    if (vecSPhi[0] >= vecgeom::InfinityLength<Real_t>() - vecgeom::kToleranceStrict<Real_t> &&
        vecEPhi[0] >= vecgeom::InfinityLength<Real_t>() - vecgeom::kToleranceStrict<Real_t>)
      return true;
    int flipsign = flip ? -1 : 1;
    AngleVector<Real_t> localAngle{local[0], local[1]};
    auto convex = vecSPhi.CrossZ(vecEPhi) > Real_t(0);
    auto in1    = vecSPhi.CrossZ(localAngle) > -flipsign * vecgeom::kToleranceStrict<Real_t>;
    auto in2    = localAngle.CrossZ(vecEPhi) > -flipsign * vecgeom::kToleranceStrict<Real_t>;
    return convex ? in1 && in2 : in1 || in2;
  }

  TorusData() = default;
  /// @tparam Real_i precision type of inputs
  template <typename Real_i>
  TorusData(Real_i rad, Real_i rad_tube, Real_i sphi = Real_i{0}, Real_i ephi = Real_i{0}, bool flip = false)
      : rTor(static_cast<Real_t>(rad)), rTube(flip ? static_cast<Real_t>(-rad_tube) : static_cast<Real_t>(rad_tube)),
        vecSPhi(static_cast<Real_t>(vecgeom::Cos(sphi)), static_cast<Real_t>(vecgeom::Sin(sphi))),
        vecEPhi(static_cast<Real_t>(vecgeom::Cos(ephi)), static_cast<Real_t>(vecgeom::Sin(ephi))),
        inner_BC_radius(-(1. - rad_tube / vecgeom::NonZero(rad))),
        outer_BC_radius(1. + rad_tube / vecgeom::NonZero(rad))
  {
  }
  template <typename Real_i>
  TorusData(const TorusData<Real_i> &other)
      : rTor(static_cast<Real_t>(other.rTor)), rTube(static_cast<Real_t>(other.rTube)),
        vecSPhi(static_cast<Real_t>(other.vecSPhi[0]), static_cast<Real_t>(other.vecSPhi[1])),
        vecEPhi(static_cast<Real_t>(other.vecEPhi[0]), static_cast<Real_t>(other.vecEPhi[1])),
        inner_BC_radius(static_cast<Real_t>(other.inner_BC_radius)),
        outer_BC_radius(static_cast<Real_t>(other.outer_BC_radius))
  {
  }

  VECCORE_ATT_HOST_DEVICE
  Real_t Radius() const { return std::abs(rTor); }
  VECCORE_ATT_HOST_DEVICE
  Real_t RadiusTube() const { return std::abs(rTube); }
  VECCORE_ATT_HOST_DEVICE
  VECCORE_ATT_HOST_DEVICE
  bool IsFlipped() const { return rTube < 0; }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Real_t InnerBCRadius() const { return std::abs(inner_BC_radius); }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Real_t OuterBCRadius() const { return std::abs(outer_BC_radius); }
};

} // namespace vgbrep

#endif
