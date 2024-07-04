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

/// @brief Frame checker utility
/// @tparam Real_t Precision type
/// @tparam FrameType1 Parent frame type
/// @tparam FrameType2 hild frame type
template <typename Real_t, typename FrameType1, typename FrameType2>
struct FrameChecker {

  /// @brief heck if a child frame is embedded in a parent frame
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
};

template <typename Real_t>
struct FrameChecker<Real_t, RingMask<Real_t>, RingMask<Real_t>> {
  static bool IsEmbedding(RingMask<Real_t> const &frame1, RingMask<Real_t> const &frame2,
                          TransformationMP<Real_t> const &trans)
  {
    // Special case where trans is identity (most frequent)
    if (trans.IsIdentity() || (!(trans.HasTranslation()) && trans.IsXYRotation())) {
      // Radial embedding
      if (frame2.rangeR[0] < vecgeom::MakeMinusTolerant<true, Real_t>(frame1.rangeR[0]) ||
          frame2.rangeR[1] > vecgeom::MakePlusTolerant<true, Real_t>(frame1.rangeR[1]))
        return false;
      // Phi embedding

      Vector3D<Real_t> SPhi{frame2.vecSPhi[0], frame2.vecSPhi[1], 0};
      Vector3D<Real_t> EPhi{frame2.vecEPhi[0], frame2.vecEPhi[1], 0};
      Vector3D<Real_t> trans_SPhi = trans.InverseTransformDirection(SPhi);
      Vector3D<Real_t> trans_EPhi = trans.InverseTransformDirection(EPhi);

      if (!frame1.InsidePhi(trans_SPhi[0], trans_SPhi[1]) || !frame1.InsidePhi(trans_EPhi[0], trans_EPhi[1]))
        return false;
      return true;
    }
    // General case: check if Safety inside for the center of the frame2 circle is large enough
    Vector3D<Real_t> center;
    Real_t safety     = frame1.SafetyInside(trans.InverseTransform(center));
    bool is_embedding = frame2.rangeR[1] < vecgeom::MakePlusTolerant<true, Real_t>(safety);
    if (!(is_embedding))
      VECGEOM_LOG(warning) << "Non-embedded Ring frame in Ring parent frame detected. This could be a general case "
                              "that is not treated or an extruding overlap.";
    return is_embedding;
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

    Vector3D<Real_t> SPhi{frame2.vecSPhi[0], frame2.vecSPhi[1], 0};
    Vector3D<Real_t> EPhi{frame2.vecEPhi[0], frame2.vecEPhi[1], 0};
    Vector3D<Real_t> trans_SPhi = trans.InverseTransform(SPhi);
    Vector3D<Real_t> trans_EPhi = trans.InverseTransform(EPhi);

    // Phi embedding (does not require transformation since transformations for
    // zphi masks on the same common surface are only allowed along z)
    if (!frame1.InsidePhi(trans_SPhi[0], trans_SPhi[1]) || !frame1.InsidePhi(trans_EPhi[0], trans_EPhi[1]))
      return false;

    return true;
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
};

/// More FrameChecker specializations follow

} // namespace vgbrep

#endif
