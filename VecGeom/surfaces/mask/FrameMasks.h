#ifndef VECGEOM_SURFACE_FRAMEMASKS_H
#define VECGEOM_SURFACE_FRAMEMASKS_H

#ifndef SURF_ACCURATE_SAFETY
#define SURF_ACCURATE_SAFETY 0
#endif

#define WINDOW_ACCURATE_SAFETY SURF_ACCURATE_SAFETY
#define QUAD_ACCURATE_SAFETY SURF_ACCURATE_SAFETY

#include <VecGeom/surfaces/mask/WindowMask.h>
#include <VecGeom/surfaces/mask/RingMask.h>
#include <VecGeom/surfaces/mask/ZPhiMask.h>
#include <VecGeom/surfaces/mask/TriangleMask.h>
#include <VecGeom/surfaces/mask/QuadrilateralMask.h>

namespace vgbrep {

namespace detail {
template <typename Real_t>
static void ConvertAngleToFrame1(Vector3D<Real_t> &sPhi, Vector3D<Real_t> &ePhi, TransformationMP<Real_t> const &trans)
{
  bool flip       = trans.Rotation(8) < 0.;
  auto trans_sPhi = trans.InverseTransformDirection(sPhi);
  auto trans_ePhi = trans.InverseTransformDirection(ePhi);
  sPhi            = flip ? trans_ePhi : trans_sPhi;
  ePhi            = flip ? trans_sPhi : trans_ePhi;
}
} // namespace detail

/// @brief Frame checker utility
/// @tparam Real_t Precision type
/// @tparam FrameType1 Parent frame type
/// @tparam FrameType2 hild frame type
template <typename Real_t, typename FrameType1, typename FrameType2>
struct FrameChecker {

  /// @brief Check if a child frame is embedded in a parent frame
  /// @param f1 Parent frame
  /// @param f2 Child frame
  /// @param trans Transformation of the child in the parent reference system
  /// @return Child frame contained in the parent one
  static bool IsEmbedding(FrameType1 const &f1, FrameType2 const &f2, TransformationMP<Real_t> const &trans)
  {
    VECGEOM_LOG(error) << "Frame embedding not implemented for " << typeid(FrameType1).name() << " - "
                       << typeid(FrameType2).name();
    return false;
  }

  /// @brief Check if two frames of arbitrary types are disjoint
  /// @param f1 First frame
  /// @param f2 Second frame
  /// @param trans Transformation of the second frame in the reference frame of the first
  /// @return Frames are disjoint
  static bool AreDisjoint(FrameType1 const &f1, FrameType2 const &f2, TransformationMP<Real_t> const &trans)
  {
    VECGEOM_LOG(error) << "Frame disjoint check not implemented for " << typeid(FrameType1).name() << " - "
                       << typeid(FrameType2).name();
    return false;
  }

  /// @brief Full check of a frame against the other
  /// @param f1 First frame
  /// @param f2 Second frame
  /// @param trans Transformation of the second frame in the reference frame of the first
  /// @return Intersection type
  static FrameIntersect CheckFrames(FrameType1 const &f1, FrameType2 const &f2, TransformationMP<Real_t> const &trans)
  {
    if (FrameChecker<Real_t, FrameType1, FrameType2>::AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (FrameChecker<Real_t, FrameType1, FrameType2>::IsEmbedding(f1, f2, trans))
      intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, FrameType2, FrameType1>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, RingMask<Real_t>, RingMask<Real_t>> {
  static bool IsEmbedding(RingMask<Real_t> const &frame1, RingMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Special case where trans is identity (most frequent)
    if (!trans.HasTranslation()) {
      // Radial embedding
      if (frame2.rangeR[0] < vecgeom::MakeMinusTolerant<true, Real_t>(frame1.rangeR[0]) ||
          frame2.rangeR[1] > vecgeom::MakePlusTolerant<true, Real_t>(frame1.rangeR[1]))
        return false;
      // Phi embedding
      if (frame1.isFullCirc) return true;
      if (frame2.isFullCirc) return false;
      Vector3D<Real_t> sPhi{frame2.vecSPhi[0], frame2.vecSPhi[1], 0};
      Vector3D<Real_t> ePhi{frame2.vecEPhi[0], frame2.vecEPhi[1], 0};
      detail::ConvertAngleToFrame1(sPhi, ePhi, trans);
      AngleInterval<Real_t> ang1(frame1.vecSPhi.Phi(), frame1.vecEPhi.Phi());
      AngleInterval<Real_t> ang2(sPhi.Phi(), ePhi.Phi());
      bool embedding = ang1.Intersect(ang2) == ang2;
      return embedding;
    }
    // General case: check if Safety inside for the center of the frame2 circle is large enough
    Vector3D<Real_t> center;
    Real_t safety = frame1.SafetyInside(trans.InverseTransform(center));
    return frame2.rangeR[1] < vecgeom::MakePlusTolerant<true, Real_t>(safety);
  }

  static bool AreDisjoint(RingMask<Real_t> const &frame1, RingMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Concentric rings
    if (!trans.HasTranslation()) {
      // non-overlapping on radius
      if (frame2.rangeR[0] > vecgeom::MakeMinusTolerant<true, Real_t>(frame1.rangeR[1]) ||
          frame1.rangeR[0] > vecgeom::MakeMinusTolerant<true, Real_t>(frame2.rangeR[1]))
        return true;
      // one of the two is a full circle, while there is an overlap on R
      if (frame1.isFullCirc || frame2.isFullCirc) return false;
      // Phi check
      Vector3D<Real_t> sPhi{frame2.vecSPhi[0], frame2.vecSPhi[1], 0};
      Vector3D<Real_t> ePhi{frame2.vecEPhi[0], frame2.vecEPhi[1], 0};
      detail::ConvertAngleToFrame1(sPhi, ePhi, trans);
      AngleInterval<Real_t> ang1(frame1.vecSPhi.Phi(), frame1.vecEPhi.Phi());
      AngleInterval<Real_t> ang2(sPhi.Phi(), ePhi.Phi());
      bool disjoint = ang1.Intersect(ang2).IsNull();
      return disjoint;
    }
    // Two rings are disjoint if:
    // 1. The distance between their centers (norm of the translation) is larger than the sum of outer radii
    auto dist = trans.Translation().Length();
    if (frame1.rangeR[1] + frame2.rangeR[1] < dist + vecgeom::kToleranceDist<Real_t> * dist) return true;
    // 2. One of the inner holes of one ring embeds the other ring
    if (frame1.rangeR[0] - frame2.rangeR[1] > dist - vecgeom::kToleranceDist<Real_t> * dist ||
        frame2.rangeR[0] - frame1.rangeR[1] > dist - vecgeom::kToleranceDist<Real_t> * dist)
      return true;
    // 3. The bounding boxes are disjoint (relevant for rings having acute angles) **NOT CHECKED YET**
    return false;
  }

  static FrameIntersect CheckFrames(RingMask<Real_t> const &f1, RingMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, RingMask<Real_t>, WindowMask<Real_t>> {
  static bool IsEmbedding(RingMask<Real_t> const &frame1, WindowMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    Vector3D<Real_t> vlocal[4] = {{frame2.rangeU[0], frame2.rangeV[0], 0},
                                  {frame2.rangeU[0], frame2.rangeV[1], 0},
                                  {frame2.rangeU[1], frame2.rangeV[1], 0},
                                  {frame2.rangeU[1], frame2.rangeV[0], 0}};

    Vector3D<Real_t> v[4];
    // If any vertex is not embedded, the frame is not embedded
    for (auto i = 0; i < 4; ++i) {
      v[i] = trans.InverseTransform(vlocal[i]);
      if (!frame1.Inside(v[i])) return false;
    }
    // Less than 180 deg ring having Rmin = 0 (convex)
    if (frame1.IsConvex()) return true;
    // General case: All segments of frame2 must not cross frame1 edges, and the center of frame1 must not be contained
    // in frame2
    Segment2D<Real_t> seg1_phi1({frame1.rangeR[0] * frame1.vecSPhi}, {frame1.rangeR[1] * frame1.vecSPhi});
    Segment2D<Real_t> seg1_phi2({frame1.rangeR[0] * frame1.vecEPhi}, {frame1.rangeR[1] * frame1.vecEPhi});
    for (size_t i = 0; i < 4; ++i) {
      Segment2D<Real_t> seg2({v[i].x(), v[i].y()}, {v[(i + 1) % 4].x(), v[(i + 1) % 4].y()});
      if (frame1.HasRmin()) {
        auto embedded = seg2.Intersect(Circle2D<Real_t>{frame1.rangeR[0], {Real_t(0), Real_t{0}}});
        if (embedded == SegmentIntersect::kEmbedded || embedded == SegmentIntersect::kIntersect) return false;
      }
      if (!frame1.isFullCirc) {
        auto embedded = seg1_phi1.Intersect(seg2);
        if (embedded != SegmentIntersect::kNoIntersect && embedded != SegmentIntersect::kEmbedding) return false;
        embedded = seg1_phi2.Intersect(seg2);
        if (embedded != SegmentIntersect::kNoIntersect && embedded != SegmentIntersect::kEmbedding) return false;
      }
    }
    if (frame1.isFullCirc && frame1.HasRmin()) {
      // The center of the circle must not be inside the frame
      auto center = trans.Transform(Vector3D<Real_t>{});
      if (frame2.Inside(center)) return false;
    }

    return true;
  }

  static bool AreDisjoint(RingMask<Real_t> const &frame1, WindowMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Fast check: The safety from outside of the center of the ring in the window frame is larger than the outer radius
    auto center = trans.Transform(Vector3D<Real_t>{});
    if (frame2.SafetyOutside(center) > frame1.rangeR[1] - vecgeom::kToleranceDist<Real_t> * frame1.rangeR[1])
      return true;
    // Fast check: The center of the box is inside the inner radius of the ring and its safety from outside is less than
    // the box diagonal
    center      = trans.InverseTransform(Vector3D<Real_t>{});
    auto diag2  = frame2.GetHalfSize().Length2();
    auto safety = frame1.SafetyOutside(center);
    if (center.Length2() < frame1.rangeR[0] * frame1.rangeR[0] &&
        diag2 < safety * safety + vecgeom::kToleranceDistSquared<Real_t>)
      return true;
    return false;
  }

  static FrameIntersect CheckFrames(RingMask<Real_t> const &f1, WindowMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, WindowMask<Real_t>, RingMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, RingMask<Real_t>, TriangleMask<Real_t>> {
  static bool IsEmbedding(RingMask<Real_t> const &frame1, TriangleMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    Vector3D<Real_t> vlocal[3] = {{frame2.p_[0].x(), frame2.p_[0].y(), 0},
                                  {frame2.p_[1].x(), frame2.p_[1].y(), 0},
                                  {frame2.p_[2].x(), frame2.p_[2].y(), 0}};

    Vector3D<Real_t> v[3];
    // If any vertex is not embedded, the frame is not embedded
    for (auto i = 0; i < 3; ++i) {
      v[i] = trans.InverseTransform(vlocal[i]);
      if (!frame1.Inside(v[i])) return false;
    }
    // Less than 180 deg ring having Rmin = 0 (convex)
    if (frame1.IsConvex()) return true;
    // General case: All segments of frame2 must not cross frame1 edges, and the center of frame1 must not be contained
    // in frame2
    for (size_t i = 0; i < 3; ++i) {
      Segment2D<Real_t> seg2({v[i].x(), v[i].y()}, {v[(i + 1) % 3].x(), v[(i + 1) % 3].y()});
      if (frame1.HasRmin()) {
        auto embedded = seg2.Intersect(Circle2D<Real_t>{frame1.rangeR[0], {Real_t(0), Real_t{0}}});
        if (embedded == SegmentIntersect::kEmbedded || embedded == SegmentIntersect::kIntersect) return false;
      }
      if (!frame1.isFullCirc) {
        Segment2D<Real_t> seg1_phi1({frame1.rangeR[0] * frame1.vecSPhi}, {frame1.rangeR[1] * frame1.vecSPhi});
        Segment2D<Real_t> seg1_phi2({frame1.rangeR[0] * frame1.vecEPhi}, {frame1.rangeR[1] * frame1.vecEPhi});
        auto embedded = seg1_phi1.Intersect(seg2);
        if (embedded != SegmentIntersect::kNoIntersect && embedded != SegmentIntersect::kEmbedding) return false;
        embedded = seg1_phi2.Intersect(seg2);
        if (embedded != SegmentIntersect::kNoIntersect && embedded != SegmentIntersect::kEmbedding) return false;
      }
    }
    // The center of the circle must not be inside the frame
    if (!frame1.isFullCirc) {
      auto center = trans.Transform(Vector3D<Real_t>{});
      if (frame2.Inside(center)) return false;
    }

    return true;
  }

  static bool AreDisjoint(RingMask<Real_t> const &frame1, TriangleMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Fast check: The safety form outside of the center of the ring in the window frame is larger than the outer radius
    auto center = trans.Transform(Vector3D<Real_t>{});
    return (frame2.SafetyOutside(center) > frame1.rangeR[1] - vecgeom::kToleranceDist<Real_t> * frame1.rangeR[1]);
  }

  static FrameIntersect CheckFrames(RingMask<Real_t> const &f1, TriangleMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, TriangleMask<Real_t>, RingMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, RingMask<Real_t>, QuadrilateralMask<Real_t>> {
  static bool IsEmbedding(RingMask<Real_t> const &frame1, QuadrilateralMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    Vector3D<Real_t> vlocal[4] = {{frame2.p_[0].x(), frame2.p_[0].y(), 0},
                                  {frame2.p_[1].x(), frame2.p_[1].y(), 0},
                                  {frame2.p_[2].x(), frame2.p_[2].y(), 0},
                                  {frame2.p_[3].x(), frame2.p_[3].y(), 0}};

    Vector3D<Real_t> v[4];
    // If any vertex is not embedded, the frame is not embedded
    for (auto i = 0; i < 4; ++i) {
      v[i] = trans.InverseTransform(vlocal[i]);
      if (!frame1.Inside(v[i])) return false;
    }
    // Less than 180 deg ring having Rmin = 0 (convex)
    if (frame1.IsConvex()) return true;
    // General case: All segments of frame2 must not cross frame1 edges, and the center of frame1 must not be contained
    // in frame2
    for (size_t i = 0; i < 4; ++i) {
      Segment2D<Real_t> seg2({v[i].x(), v[i].y()}, {v[(i + 1) % 4].x(), v[(i + 1) % 4].y()});
      if (frame1.HasRmin()) {
        auto embedded = seg2.Intersect(Circle2D<Real_t>{frame1.rangeR[0], {Real_t(0), Real_t{0}}});
        if (embedded == SegmentIntersect::kEmbedded || embedded == SegmentIntersect::kIntersect) return false;
      }
      if (!frame1.isFullCirc) {
        Segment2D<Real_t> seg1_phi1({frame1.rangeR[0] * frame1.vecSPhi}, {frame1.rangeR[1] * frame1.vecSPhi});
        Segment2D<Real_t> seg1_phi2({frame1.rangeR[0] * frame1.vecEPhi}, {frame1.rangeR[1] * frame1.vecEPhi});
        auto embedded = seg1_phi1.Intersect(seg2);
        if (embedded != SegmentIntersect::kNoIntersect && embedded != SegmentIntersect::kEmbedding) return false;
        embedded = seg1_phi2.Intersect(seg2);
        if (embedded != SegmentIntersect::kNoIntersect && embedded != SegmentIntersect::kEmbedding) return false;
      }
    }
    // The center of the circle must not be inside the frame
    auto center = trans.Transform(Vector3D<Real_t>{});
    if (frame2.Inside(center)) return false;

    return true;
  }

  static bool AreDisjoint(RingMask<Real_t> const &frame1, QuadrilateralMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Fast check: The safety form outside of the center of the ring in the window frame is larger than the outer radius
    auto center = trans.Transform(Vector3D<Real_t>{});
    return (frame2.SafetyOutside(center) > frame1.rangeR[1] - vecgeom::kToleranceDist<Real_t> * frame1.rangeR[1]);
  }

  static FrameIntersect CheckFrames(RingMask<Real_t> const &f1, QuadrilateralMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, QuadrilateralMask<Real_t>, RingMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, ZPhiMask<Real_t>, ZPhiMask<Real_t>> {
  static bool IsEmbedding(ZPhiMask<Real_t> const &frame1, ZPhiMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // embedding in Z must be transformed in z
    Vector3D<Real_t> zmin(0., 0., frame2.rangeZ[0]);
    Vector3D<Real_t> zmax(0., 0., frame2.rangeZ[1]);

    zmin = trans.InverseTransform(zmin);
    zmax = trans.InverseTransform(zmax);

    // Z embedding
    if (zmin[2] < vecgeom::MakeMinusTolerant<true, Real_t>(frame1.rangeZ[0]) ||
        zmax[2] > vecgeom::MakePlusTolerant<true, Real_t>(frame1.rangeZ[1]))
      return false;

    // Phi embedding
    if (frame1.isFullCirc) return true;
    if (frame2.isFullCirc) return false;
    Vector3D<Real_t> sPhi{frame2.vecSPhi[0], frame2.vecSPhi[1], 0};
    Vector3D<Real_t> ePhi{frame2.vecEPhi[0], frame2.vecEPhi[1], 0};
    detail::ConvertAngleToFrame1(sPhi, ePhi, trans);
    AngleInterval<Real_t> ang1(frame1.vecSPhi.Phi(), frame1.vecEPhi.Phi());
    AngleInterval<Real_t> ang2(sPhi.Phi(), ePhi.Phi());
    bool embedding = ang1.Intersect(ang2) == ang2;
    return embedding;
  }

  static bool AreDisjoint(ZPhiMask<Real_t> const &frame1, ZPhiMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // embedding in Z must be transformed in z
    Vector3D<Real_t> zmin(0., 0., frame2.rangeZ[0]);
    Vector3D<Real_t> zmax(0., 0., frame2.rangeZ[1]);

    zmin = trans.InverseTransform(zmin);
    zmax = trans.InverseTransform(zmax);

    // Z embedding
    if (zmin[2] > vecgeom::MakeMinusTolerant<true, Real_t>(frame1.rangeZ[1]) ||
        zmax[2] < vecgeom::MakePlusTolerant<true, Real_t>(frame1.rangeZ[0]))
      return true;

    // Check if the masks are phi-disjoint
    if (frame1.isFullCirc || frame2.isFullCirc) return false;
    // Phi check
    Vector3D<Real_t> sPhi{frame2.vecSPhi[0], frame2.vecSPhi[1], 0};
    Vector3D<Real_t> ePhi{frame2.vecEPhi[0], frame2.vecEPhi[1], 0};
    detail::ConvertAngleToFrame1(sPhi, ePhi, trans);
    AngleInterval<Real_t> ang1(frame1.vecSPhi.Phi(), frame1.vecEPhi.Phi());
    AngleInterval<Real_t> ang2(sPhi.Phi(), ePhi.Phi());
    bool disjoint = ang1.Intersect(ang2).IsNull();
    return disjoint;
  }

  static FrameIntersect CheckFrames(ZPhiMask<Real_t> const &f1, ZPhiMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, WindowMask<Real_t>, RingMask<Real_t>> {
  static bool IsEmbedding(WindowMask<Real_t> const &frame1, RingMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Compute SafetyInside for the center of the circle
    // NOTE: for now checking only if the full circle is embedded
    Vector3D<Real_t> center;
    Real_t safety = frame1.SafetyInside(trans.InverseTransform(center));
    return safety > frame2.rangeR[1] - vecgeom::kToleranceDist<Real_t> * frame2.rangeR[1];
  }

  static bool AreDisjoint(WindowMask<Real_t> const &frame1, RingMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    return FrameChecker<Real_t, RingMask<Real_t>, WindowMask<Real_t>>::AreDisjoint(frame2, frame1, trans.Inverse());
  }

  static FrameIntersect CheckFrames(WindowMask<Real_t> const &f1, RingMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, RingMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, WindowMask<Real_t>, WindowMask<Real_t>> {
  static bool IsEmbedding(WindowMask<Real_t> const &frame1, WindowMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v[4] = {{frame2.rangeU[0], frame2.rangeV[0], 0},
                             {frame2.rangeU[0], frame2.rangeV[1], 0},
                             {frame2.rangeU[1], frame2.rangeV[1], 0},
                             {frame2.rangeU[1], frame2.rangeV[0], 0}};

    for (auto i = 0; i < 4; ++i) {
      if (!frame1.Inside(trans.InverseTransform(v[i]))) return false;
    }
    return true;
  }

  static bool AreDisjoint(WindowMask<Real_t> const &frame1, WindowMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v1[4] = {{frame1.rangeU[0], frame1.rangeV[0], 0},
                              {frame1.rangeU[0], frame1.rangeV[1], 0},
                              {frame1.rangeU[1], frame1.rangeV[1], 0},
                              {frame1.rangeU[1], frame1.rangeV[0], 0}};
    Vector3D<Real_t> v2[4] = {trans.InverseTransform(Vector3D<Real_t>(frame2.rangeU[0], frame2.rangeV[0], 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.rangeU[0], frame2.rangeV[1], 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.rangeU[1], frame2.rangeV[1], 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.rangeU[1], frame2.rangeV[0], 0))};
    Polygon<Real_t, 4> poly1(v1);
    Polygon<Real_t, 4> poly2(v2);
    auto intersect = poly1.Intersect(poly2);
    if (intersect == FrameIntersect::kIntersect) return false;
    // Check if the centers are also disjoint
    auto center1 = poly1.GetCenter();
    auto center2 = poly2.GetCenter();
    if (frame1.Inside(Vector3D<Real_t>(center2[0], center2[1], 0)) ||
        frame2.Inside(trans.Transform(Vector3D<Real_t>(center1[0], center1[1], 0))))
      return false;
    return true;
  }

  static FrameIntersect CheckFrames(WindowMask<Real_t> const &f1, WindowMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, WindowMask<Real_t>, TriangleMask<Real_t>> {
  static bool IsEmbedding(WindowMask<Real_t> const &frame1, TriangleMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v[3] = {{frame2.p_[0].x(), frame2.p_[0].y(), 0},
                             {frame2.p_[1].x(), frame2.p_[1].y(), 0},
                             {frame2.p_[2].x(), frame2.p_[2].y(), 0}};
    for (auto i = 0; i < 3; ++i) {
      if (!frame1.Inside(trans.InverseTransform(v[i]))) return false;
    }
    return true;
  }

  static bool AreDisjoint(WindowMask<Real_t> const &frame1, TriangleMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v1[4] = {{frame1.rangeU[0], frame1.rangeV[0], 0},
                              {frame1.rangeU[0], frame1.rangeV[1], 0},
                              {frame1.rangeU[1], frame1.rangeV[1], 0},
                              {frame1.rangeU[1], frame1.rangeV[0], 0}};
    Vector3D<Real_t> v2[3] = {trans.InverseTransform(Vector3D<Real_t>(frame2.p_[0].x(), frame2.p_[0].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[1].x(), frame2.p_[1].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[2].x(), frame2.p_[2].y(), 0))};
    Polygon<Real_t, 4> poly1(v1);
    Polygon<Real_t, 3> poly2(v2);
    auto intersect = poly1.Intersect(poly2);
    if (intersect == FrameIntersect::kIntersect) return false;
    // Check if the centers are also disjoint
    auto center1 = poly1.GetCenter();
    auto center2 = poly2.GetCenter();
    if (frame1.Inside(Vector3D<Real_t>(center2[0], center2[1], 0)) ||
        frame2.Inside(trans.Transform(Vector3D<Real_t>(center1[0], center1[1], 0))))
      return false;
    return true;
  }

  static FrameIntersect CheckFrames(WindowMask<Real_t> const &f1, TriangleMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, TriangleMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, WindowMask<Real_t>, QuadrilateralMask<Real_t>> {
  static bool IsEmbedding(WindowMask<Real_t> const &frame1, QuadrilateralMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v[4] = {{frame2.p_[0].x(), frame2.p_[0].y(), 0},
                             {frame2.p_[1].x(), frame2.p_[1].y(), 0},
                             {frame2.p_[2].x(), frame2.p_[2].y(), 0},
                             {frame2.p_[3].x(), frame2.p_[3].y(), 0}};
    for (auto i = 0; i < 4; ++i) {
      if (!frame1.Inside(trans.InverseTransform(v[i]))) return false;
    }
    return true;
  }

  static bool AreDisjoint(WindowMask<Real_t> const &frame1, QuadrilateralMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v1[4] = {{frame1.rangeU[0], frame1.rangeV[0], 0},
                              {frame1.rangeU[0], frame1.rangeV[1], 0},
                              {frame1.rangeU[1], frame1.rangeV[1], 0},
                              {frame1.rangeU[1], frame1.rangeV[0], 0}};
    Vector3D<Real_t> v2[4] = {trans.InverseTransform(Vector3D<Real_t>(frame2.p_[0].x(), frame2.p_[0].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[1].x(), frame2.p_[1].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[2].x(), frame2.p_[2].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[3].x(), frame2.p_[3].y(), 0))};
    Polygon<Real_t, 4> poly1(v1);
    Polygon<Real_t, 4> poly2(v2);
    auto intersect = poly1.Intersect(poly2);
    if (intersect == FrameIntersect::kIntersect) return false;
    // Check if the centers are also disjoint
    auto center1 = poly1.GetCenter();
    auto center2 = poly2.GetCenter();
    if (frame1.Inside(Vector3D<Real_t>(center2[0], center2[1], 0)) ||
        frame2.Inside(trans.Transform(Vector3D<Real_t>(center1[0], center1[1], 0))))
      return false;
    return true;
  }

  static FrameIntersect CheckFrames(WindowMask<Real_t> const &f1, QuadrilateralMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, QuadrilateralMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, TriangleMask<Real_t>, RingMask<Real_t>> {
  static bool IsEmbedding(TriangleMask<Real_t> const &frame1, RingMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Compute SafetyInside for the center of the circle
    // NOTE: for now checking only if the full circle is embedded
    Vector3D<Real_t> center;
    Real_t safety = frame1.SafetyInside(trans.InverseTransform(center));
    return safety > frame2.rangeR[1] - vecgeom::kToleranceDist<Real_t> * frame2.rangeR[1];
  }

  static bool AreDisjoint(TriangleMask<Real_t> const &frame1, RingMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    return FrameChecker<Real_t, RingMask<Real_t>, TriangleMask<Real_t>>::AreDisjoint(frame2, frame1, trans.Inverse());
  }

  static FrameIntersect CheckFrames(TriangleMask<Real_t> const &f1, RingMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, RingMask<Real_t>, TriangleMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, TriangleMask<Real_t>, WindowMask<Real_t>> {
  static bool IsEmbedding(TriangleMask<Real_t> const &frame1, WindowMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v[4] = {{frame2.rangeU[0], frame2.rangeV[0], 0},
                             {frame2.rangeU[0], frame2.rangeV[1], 0},
                             {frame2.rangeU[1], frame2.rangeV[1], 0},
                             {frame2.rangeU[1], frame2.rangeV[0], 0}};

    for (auto i = 0; i < 4; ++i) {
      if (!frame1.Inside(trans.InverseTransform(v[i]))) return false;
    }
    return true;
  }

  static bool AreDisjoint(TriangleMask<Real_t> const &frame1, WindowMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    return FrameChecker<Real_t, WindowMask<Real_t>, TriangleMask<Real_t>>::AreDisjoint(frame2, frame1, trans.Inverse());
  }

  static FrameIntersect CheckFrames(TriangleMask<Real_t> const &f1, WindowMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, WindowMask<Real_t>, TriangleMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, TriangleMask<Real_t>, TriangleMask<Real_t>> {
  static bool IsEmbedding(TriangleMask<Real_t> const &frame1, TriangleMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v[3] = {{frame2.p_[0].x(), frame2.p_[0].y(), 0},
                             {frame2.p_[1].x(), frame2.p_[1].y(), 0},
                             {frame2.p_[2].x(), frame2.p_[2].y(), 0}};
    for (auto i = 0; i < 3; ++i) {
      if (!frame1.Inside(trans.InverseTransform(v[i]))) return false;
    }
    return true;
  }

  static bool AreDisjoint(TriangleMask<Real_t> const &frame1, TriangleMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v1[3] = {{frame1.p_[0].x(), frame1.p_[0].y(), 0},
                              {frame1.p_[1].x(), frame1.p_[1].y(), 0},
                              {frame1.p_[2].x(), frame1.p_[2].y(), 0}};
    Vector3D<Real_t> v2[3] = {trans.InverseTransform(Vector3D<Real_t>(frame2.p_[0].x(), frame2.p_[0].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[1].x(), frame2.p_[1].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[2].x(), frame2.p_[2].y(), 0))};
    Polygon<Real_t, 3> poly1(v1);
    Polygon<Real_t, 3> poly2(v2);
    auto intersect = poly1.Intersect(poly2);
    if (intersect == FrameIntersect::kIntersect) return false;
    // Check if the centers are also disjoint
    auto center1 = poly1.GetCenter();
    auto center2 = poly2.GetCenter();
    if (frame1.Inside(Vector3D<Real_t>(center2[0], center2[1], 0)) ||
        frame2.Inside(trans.Transform(Vector3D<Real_t>(center1[0], center1[1], 0))))
      return false;
    return true;
  }

  static FrameIntersect CheckFrames(TriangleMask<Real_t> const &f1, TriangleMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, TriangleMask<Real_t>, QuadrilateralMask<Real_t>> {
  static bool IsEmbedding(TriangleMask<Real_t> const &frame1, QuadrilateralMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v[4] = {{frame2.p_[0].x(), frame2.p_[0].y(), 0},
                             {frame2.p_[1].x(), frame2.p_[1].y(), 0},
                             {frame2.p_[2].x(), frame2.p_[2].y(), 0},
                             {frame2.p_[3].x(), frame2.p_[3].y(), 0}};
    for (auto i = 0; i < 4; ++i) {
      if (!frame1.Inside(trans.InverseTransform(v[i]))) return false;
    }
    return true;
  }

  static bool AreDisjoint(TriangleMask<Real_t> const &frame1, QuadrilateralMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v1[3] = {{frame1.p_[0].x(), frame1.p_[0].y(), 0},
                              {frame1.p_[1].x(), frame1.p_[1].y(), 0},
                              {frame1.p_[2].x(), frame1.p_[2].y(), 0}};
    Vector3D<Real_t> v2[4] = {trans.InverseTransform(Vector3D<Real_t>(frame2.p_[0].x(), frame2.p_[0].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[1].x(), frame2.p_[1].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[2].x(), frame2.p_[2].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[3].x(), frame2.p_[3].y(), 0))};
    Polygon<Real_t, 3> poly1(v1);
    Polygon<Real_t, 4> poly2(v2);
    auto intersect = poly1.Intersect(poly2);
    if (intersect == FrameIntersect::kIntersect) return false;
    // Check if the centers are also disjoint
    auto center1 = poly1.GetCenter();
    auto center2 = poly2.GetCenter();
    if (frame1.Inside(Vector3D<Real_t>(center2[0], center2[1], 0)) ||
        frame2.Inside(trans.Transform(Vector3D<Real_t>(center1[0], center1[1], 0))))
      return false;
    return true;
  }

  static FrameIntersect CheckFrames(TriangleMask<Real_t> const &f1, QuadrilateralMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, QuadrilateralMask<Real_t>, TriangleMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, QuadrilateralMask<Real_t>, RingMask<Real_t>> {
  static bool IsEmbedding(QuadrilateralMask<Real_t> const &frame1, RingMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Compute SafetyInside for the center of the circle
    // NOTE: for now checking only if the full circle is embedded
    Vector3D<Real_t> center;
    Real_t safety = frame1.SafetyInside(trans.InverseTransform(center));
    return safety > frame2.rangeR[1] - vecgeom::kToleranceDist<Real_t> * frame2.rangeR[1];
  }

  static bool AreDisjoint(QuadrilateralMask<Real_t> const &frame1, RingMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    return FrameChecker<Real_t, RingMask<Real_t>, QuadrilateralMask<Real_t>>::AreDisjoint(frame2, frame1,
                                                                                          trans.Inverse());
  }

  static FrameIntersect CheckFrames(QuadrilateralMask<Real_t> const &f1, RingMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, RingMask<Real_t>, QuadrilateralMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, QuadrilateralMask<Real_t>, WindowMask<Real_t>> {
  static bool IsEmbedding(QuadrilateralMask<Real_t> const &frame1, WindowMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v[4] = {{frame2.rangeU[0], frame2.rangeV[0], 0},
                             {frame2.rangeU[0], frame2.rangeV[1], 0},
                             {frame2.rangeU[1], frame2.rangeV[1], 0},
                             {frame2.rangeU[1], frame2.rangeV[0], 0}};

    for (auto i = 0; i < 4; ++i) {
      if (!frame1.Inside(trans.InverseTransform(v[i]))) return false;
    }
    return true;
  }

  static bool AreDisjoint(QuadrilateralMask<Real_t> const &frame1, WindowMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    return FrameChecker<Real_t, WindowMask<Real_t>, QuadrilateralMask<Real_t>>::AreDisjoint(frame2, frame1,
                                                                                            trans.Inverse());
  }

  static FrameIntersect CheckFrames(QuadrilateralMask<Real_t> const &f1, WindowMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, WindowMask<Real_t>, QuadrilateralMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, QuadrilateralMask<Real_t>, TriangleMask<Real_t>> {
  static bool IsEmbedding(QuadrilateralMask<Real_t> const &frame1, TriangleMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v[3] = {{frame2.p_[0].x(), frame2.p_[0].y(), 0},
                             {frame2.p_[1].x(), frame2.p_[1].y(), 0},
                             {frame2.p_[2].x(), frame2.p_[2].y(), 0}};
    for (auto i = 0; i < 3; ++i) {
      if (!frame1.Inside(trans.InverseTransform(v[i]))) return false;
    }
    return true;
  }

  static bool AreDisjoint(QuadrilateralMask<Real_t> const &frame1, TriangleMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    return FrameChecker<Real_t, TriangleMask<Real_t>, QuadrilateralMask<Real_t>>::AreDisjoint(frame2, frame1,
                                                                                              trans.Inverse());
  }

  static FrameIntersect CheckFrames(QuadrilateralMask<Real_t> const &f1, TriangleMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (FrameChecker<Real_t, TriangleMask<Real_t>, QuadrilateralMask<Real_t>>::IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

template <typename Real_t>
struct FrameChecker<Real_t, QuadrilateralMask<Real_t>, QuadrilateralMask<Real_t>> {
  static bool IsEmbedding(QuadrilateralMask<Real_t> const &frame1, QuadrilateralMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v[4] = {{frame2.p_[0].x(), frame2.p_[0].y(), 0},
                             {frame2.p_[1].x(), frame2.p_[1].y(), 0},
                             {frame2.p_[2].x(), frame2.p_[2].y(), 0},
                             {frame2.p_[3].x(), frame2.p_[3].y(), 0}};
    for (auto i = 0; i < 4; ++i) {
      if (!frame1.Inside(trans.InverseTransform(v[i]))) return false;
    }
    return true;
  }

  static bool AreDisjoint(QuadrilateralMask<Real_t> const &frame1, QuadrilateralMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Convert all vertices of the child in the parent frame, and check if they are contained
    Vector3D<Real_t> v1[4] = {{frame1.p_[0].x(), frame1.p_[0].y(), 0},
                              {frame1.p_[1].x(), frame1.p_[1].y(), 0},
                              {frame1.p_[2].x(), frame1.p_[2].y(), 0},
                              {frame1.p_[3].x(), frame1.p_[3].y(), 0}};
    Vector3D<Real_t> v2[4] = {trans.InverseTransform(Vector3D<Real_t>(frame2.p_[0].x(), frame2.p_[0].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[1].x(), frame2.p_[1].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[2].x(), frame2.p_[2].y(), 0)),
                              trans.InverseTransform(Vector3D<Real_t>(frame2.p_[3].x(), frame2.p_[3].y(), 0))};
    Polygon<Real_t, 4> poly1(v1);
    Polygon<Real_t, 4> poly2(v2);
    auto intersect = poly1.Intersect(poly2);
    if (intersect == FrameIntersect::kIntersect) return false;
    // Check if the centers are also disjoint
    auto center1 = poly1.GetCenter();
    auto center2 = poly2.GetCenter();
    if (frame1.Inside(Vector3D<Real_t>(center2[0], center2[1], 0)) ||
        frame2.Inside(trans.Transform(Vector3D<Real_t>(center1[0], center1[1], 0))))
      return false;
    return true;
  }

  static FrameIntersect CheckFrames(QuadrilateralMask<Real_t> const &f1, QuadrilateralMask<Real_t> const &f2,
                                    TransformationMP<Real_t> const &trans)
  {
    if (AreDisjoint(f1, f2, trans)) return FrameIntersect::kNoIntersect;
    auto intersect_type = FrameIntersect::kIntersect;
    if (IsEmbedding(f1, f2, trans)) intersect_type = FrameIntersect::kEmbedding;
    if (IsEmbedding(f2, f1, trans.Inverse())) {
      if (intersect_type == FrameIntersect::kEmbedding)
        intersect_type = FrameIntersect::kEqual;
      else
        intersect_type = FrameIntersect::kEmbedded;
    }
    return intersect_type;
  }
};

/// More FrameChecker specializations follow

} // namespace vgbrep

#endif
