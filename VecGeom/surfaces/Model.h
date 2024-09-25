#ifndef VECGEOM_SURFACE_MODEL_H_
#define VECGEOM_SURFACE_MODEL_H_

#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/surfaces/base/CommonTypes.h>
#include <VecGeom/surfaces/base/Equations.h>
#include <VecGeom/surfaces/surf/SurfaceImpl.h>
#include <VecGeom/surfaces/mask/FrameMasks.h>
#include <VecGeom/surfaces/cuda/DeviceStorage.h>

namespace vgbrep {

///< Forward declaration of surface data structure used in navigation
template <typename Real_t>
struct SurfData;

struct Frame;
using Extent = Frame;

/// @brief Unplaced half-space surface type.
/// @details Unplaced surfaces are infinite half-spaces, having a normal side convention:
///   - All unplaced planes are (xOy), having the normal oriented on positive z
///   - Unplaced cylinders, cones and tori have the z axis as axis of symmetry. Normals pointing outwards.
///   - Unplaced spheres have the origin as center, normal pointing outwards.
/// The type does not store the surface data, but only an id to an external storage.
template <typename Real_t>
struct UnplacedSurface {
  SurfaceType type{SurfaceType::kPlanar}; ///< surface type
  int id{-1};                             ///< surface id
  Real_t fRadius{0.};
  Real_t fSlope{0.};

  VECCORE_ATT_HOST_DEVICE
  Real_t Radius() const { return std::abs(Real_t(fRadius)); }
  VECCORE_ATT_HOST_DEVICE
  Real_t RadiusZ(Real_t z) const { return std::abs(fRadius) + z * fSlope; }
  VECCORE_ATT_HOST_DEVICE
  bool IsFlipped() const { return fRadius < 0; }

  UnplacedSurface() = default;

  UnplacedSurface(SurfaceType stype, int sid = -1)
  {
    type    = stype;
    id      = sid;
    fRadius = static_cast<Real_t>(0);
    fSlope  = static_cast<Real_t>(0);
  }

  template <typename Real_i>
  UnplacedSurface(SurfaceType stype, int sid = -1, Real_i radius = 0, Real_i slope = 0, bool flip = false)
  {
    type    = stype;
    id      = sid;
    fRadius = flip ? static_cast<Real_t>(-radius) : static_cast<Real_t>(radius);
    fSlope  = static_cast<Real_t>(slope);
  }

  template <typename Real_i>
  UnplacedSurface(const UnplacedSurface<Real_i> &other)
      : type(other.type), id(other.id), fRadius(static_cast<Real_t>(other.fRadius)),
        fSlope(static_cast<Real_t>(other.fSlope))
  {
  }

  /// @brief A local point is inside if behind the normal within tolerance
  /// @tparam Real_t Floating-point precision type
  /// @param point Point in the local surface coordinates
  /// @param surfdata data container with the surface data
  /// @param flip flipping the tolerance for inside (needed for negated booleans)
  /// @return Inside half-space
  template <typename DataContainer>
  VECCORE_ATT_HOST_DEVICE bool Inside(Vector3D<Real_t> const &point, DataContainer const &data, bool flip,
                                      Real_t tol = vecgeom::kToleranceStrict<Real_t>) const
  {
    switch (type) {
    case SurfaceType::kPlanar:
      return SurfaceHelper<SurfaceType::kPlanar, Real_t>().Inside(point, flip, tol);
    case SurfaceType::kCylindrical:
      return SurfaceHelper<SurfaceType::kCylindrical, Real_t>().Inside(point, flip, fRadius, tol);
    case SurfaceType::kConical:
      return SurfaceHelper<SurfaceType::kConical, Real_t>().Inside(point, flip, fRadius, fSlope, tol);
    case SurfaceType::kElliptical:
      return SurfaceHelper<SurfaceType::kElliptical, Real_t>(data.GetEllipData(id)).Inside(point, flip, tol);
    case SurfaceType::kSpherical:
      return SurfaceHelper<SurfaceType::kSpherical, Real_t>().Inside(point, flip, fRadius, tol);
    case SurfaceType::kTorus:
      return SurfaceHelper<SurfaceType::kTorus, Real_t>(data.GetTorusData(id)).Inside(point, flip, tol);
    case SurfaceType::kArb4:
      return SurfaceHelper<SurfaceType::kArb4, Real_t>(data.GetArb4Data(id)).Inside(point, flip, tol);
    default:
      return false;
    };
  }

  /// @brief Find signed distance to next intersection from local point.
  /// @tparam Real_t Floating-point precision type
  /// @param point Point in the local surface coordinates
  /// @param dir Direction in the local surface coordinates
  /// @param left_side Flag specifying if the surface is intersected from the left-side that defines the normal
  /// @param surfdata Surface data storage.
  /// @param distance Computed distance to surface
  /// @return Validity of the intersection
  VECCORE_ATT_HOST_DEVICE bool Intersect(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, bool left_side,
                                         SurfData<Real_t> const &surfdata, Real_t &distance, bool &two_solutions,
                                         Real_t &safety) const
  {
    switch (type) {
    case SurfaceType::kPlanar:
      return SurfaceHelper<SurfaceType::kPlanar, Real_t>().Intersect(point, dir, left_side, distance, two_solutions,
                                                                     safety);
    case SurfaceType::kCylindrical:
      return SurfaceHelper<SurfaceType::kCylindrical, Real_t>().Intersect(point, dir, left_side, distance,
                                                                          two_solutions, safety, fRadius);
    case SurfaceType::kConical:
      return SurfaceHelper<SurfaceType::kConical, Real_t>().Intersect(point, dir, left_side, distance, two_solutions,
                                                                      safety, fRadius, fSlope);
    case SurfaceType::kElliptical:
      return SurfaceHelper<SurfaceType::kElliptical, Real_t>(surfdata.GetEllipData(id))
          .Intersect(point, dir, left_side, distance, two_solutions, safety);
    case SurfaceType::kSpherical:
      return SurfaceHelper<SurfaceType::kSpherical, Real_t>().Intersect(point, dir, left_side, distance, two_solutions,
                                                                        safety, fRadius);
    case SurfaceType::kTorus:
      return SurfaceHelper<SurfaceType::kTorus, Real_t>(surfdata.GetTorusData(id))
          .Intersect(point, dir, left_side, distance, two_solutions, safety);
    case SurfaceType::kArb4:
      return SurfaceHelper<SurfaceType::kArb4, Real_t>(surfdata.GetArb4Data(id))
          .Intersect(point, dir, left_side, distance, two_solutions, safety);
    default:
      return false;
    };
  }

  /// @brief Computes the isotropic safe distance to unplaced surfaces
  /// @tparam Real_t Precision type for parameters
  /// @param point Point in local surface coordinates
  /// @param left_side Flag specifying if the surface is intersected from the left-side that defines the normal
  /// @param surfdata Surface data storage
  /// @param distance Computed isotropic safety
  /// @param onsurf Projection of the point on surface
  /// @return
  VECCORE_ATT_HOST_DEVICE bool Safety(Vector3D<Real_t> const &point, bool left_side, SurfData<Real_t> const &surfdata,
                                      Real_t &distance, Vector3D<Real_t> &onsurf) const
  {
    switch (type) {
    case SurfaceType::kPlanar:
      return SurfaceHelper<SurfaceType::kPlanar, Real_t>().Safety(point, left_side, distance, onsurf);
    case SurfaceType::kCylindrical:
      return SurfaceHelper<SurfaceType::kCylindrical, Real_t>().Safety(point, left_side, distance, onsurf, fRadius);
    case SurfaceType::kConical:
      return SurfaceHelper<SurfaceType::kConical, Real_t>().Safety(point, left_side, distance, onsurf, fRadius, fSlope);
    case SurfaceType::kElliptical:
      return SurfaceHelper<SurfaceType::kElliptical, Real_t>(surfdata.GetEllipData(id))
          .Safety(point, left_side, distance, onsurf);
    case SurfaceType::kSpherical:
      return SurfaceHelper<SurfaceType::kSpherical, Real_t>().Safety(point, left_side, distance, onsurf, fRadius);
    case SurfaceType::kTorus:
      return SurfaceHelper<SurfaceType::kTorus, Real_t>(surfdata.GetTorusData(id))
          .Safety(point, left_side, distance, onsurf);
    case SurfaceType::kArb4:
      return SurfaceHelper<SurfaceType::kArb4, Real_t>(surfdata.GetArb4Data(id))
          .Safety(point, left_side, distance, onsurf);
    default:
      return false;
    };
  }
};

/// @brief A frame delimiting the real solid surface on an infinite half-space
struct Frame {
  FrameType type{FrameType::kWindow}; ///< frame type
  int id{-1};                         ///< frame mask id

  Frame() = default;
  Frame(FrameType mtype, int mid = -1) : type(mtype), id(mid) {}

  // A function to check if local point is within the Frame's mask.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE bool Inside(Vector3D<Real_t> const &local, SurfData<Real_t> const &surfdata) const
  {
    switch (type) {
    case FrameType::kRing:
      return surfdata.GetRingMask(id).Inside(local);
    case FrameType::kZPhi:
      return surfdata.GetZPhiMask(id).Inside(local);
    case FrameType::kWindow:
      return surfdata.GetWindowMask(id).Inside(local);
    // TODO: Support these
    case FrameType::kTriangle:
      return surfdata.GetTriangleMask(id).Inside(local);
    case FrameType::kQuadrilateral:
      return surfdata.GetQuadMask(id).Inside(local);
    case FrameType::kRangeZ:
      /*return (local[2] > vecgeom::MakeMinusTolerant<true>(u[0]) &&
              local[2] < vecgeom::MakePlusTolerant<true>(u[1]));*/
    case FrameType::kRangeSph:
      /*return (rsq > vecgeom::MakeMinusTolerantSquare<true>(u[0]) &&
              rsq < vecgeom::MakePlusTolerantSquare<true>(u[1]));*/
    default:
      // unhandled
      return false;
    };
    return false;
  }

  /// @brief A function dispatcher to compute the safety for the frame.
  /// @tparam Real_t
  /// @param local Point on surface in local coordinates
  /// @param safetySurf Safety to the support surface
  /// @param surfdata Surface data storage
  /// @return Safety to the framed surface.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE Real_t Safety(Vector3D<Real_t> const &local, Real_t safetySurf,
                                        SurfData<Real_t> const &surfdata, bool &valid) const
  {
    switch (type) {
    case FrameType::kRing:
      return surfdata.GetRingMask(id).Safety(local, safetySurf, valid);
    case FrameType::kZPhi:
      return surfdata.GetZPhiMask(id).Safety(local, safetySurf, valid);
    case FrameType::kWindow:
      return surfdata.GetWindowMask(id).Safety(local, safetySurf, valid);
    case FrameType::kTriangle:
      return surfdata.GetTriangleMask(id).Safety(local, safetySurf, valid);
    case FrameType::kQuadrilateral:
      return surfdata.GetQuadMask(id).Safety(local, safetySurf, valid);
    case FrameType::kRangeZ:
    case FrameType::kRangeSph:
    default:
      // unhandled
      valid = false;
    };
    return Real_t(0);
  }

  template <typename Real_t, typename DataContainer>
  VECCORE_ATT_HOST_DEVICE void Extent3D(Vector3D<Real_t> &aMin, Vector3D<Real_t> &aMax, DataContainer const &data) const
  {
    switch (type) {
    case FrameType::kRing:
      // printf("Frame type: Ring. ");
      data.GetRingMask(id).Extent3D(aMin, aMax);
      break;
    case FrameType::kZPhi:
      // printf("Frame type: ZPhi. ");
      data.GetZPhiMask(id).Extent3D(aMin, aMax);
      break;
    case FrameType::kWindow:
      // printf("Frame type: Window. ");
      data.GetWindowMask(id).Extent3D(aMin, aMax);
      break;
    case FrameType::kTriangle:
      // printf("Frame type: Triangle. ");
      data.GetTriangleMask(id).Extent3D(aMin, aMax);
      break;
    case FrameType::kQuadrilateral:
      // printf("Frame type: Quadrilateral. ");
      data.GetQuadMask(id).Extent3D(aMin, aMax);
      break;
    case FrameType::kRangeZ:;
    case FrameType::kRangeSph:;
    default:
      // unhandled
      return;
    };
  }
};

// This holds the transformation of the surface
// with respect to the frame of the ancestor volume onto which this surface is flattened.
//
// Example of a 4-level hierarchy flattened on 2 levels:
// A, B, C, D, E, F = logical volumes
// t1, t2, t3, t4, t5 = local volume transformations
//
//            A                     A + C(t2) + E(t2*t4)   (one scene given by volume A)
//      (t1) / \ (t2)                      |
//          B   C                     (t1) |
//    (t3) /     \ (t4)   -->              |
//        D       E                 B + D(t3) + F(t3*t5)   (another scene given by volume B)
//   (t5) |
//        F
//
// In the above example, say F has a surface S positioned with a local ransformation (ts).
// The global corresponding to S will have a total transformation (t1 * t3 * t5 * ts).
// However, it the tree is flattened on two scenes as above (one level is A and the other level is B)
// then the local transformation of S will be just (t3 * t5 * ts). Its global transformation
// will be just (t1). In this approach, the transformation for a global surface is always obtained
// by multiplying the full scene transformation with the local surface transformation.
//
// The advantage of this approach is that it gives full flexibility for chosing the flattened
// volumes, and a given local surface can be referenced by multiple portals (less memory)

/// @brief A placed surface on a scene having a frame and a navigation state associated to a touchable
template <typename Real_t, typename Transformation_t>
struct FramedSurface {
  using NavState_t = vecgeom::NavigationState::Value_t;
  UnplacedSurface<Real_t> fSurface; ///< Surface identifier
  Frame fFrame;                     ///< Frame
  Transformation_t fTrans;  ///< 3D Transformation of the surface in the compacted sub-hierarchy top volume frame or 2D
                            ///< transformation of the frame with respect to its common surface
  int fParent{-1};          ///< Index of the first parent frame on the common surface
  int fLogicId{0};          ///< Logic flag for surface:
                            ///<   0        = non-Bool
                            ///<   positive = true logic surface
                            ///<   negative = negated logic surface
  int fSceneCS{0};          ///< The frame may belong to a daughter scene common surface
  int fSceneCSind{0};       ///< Index of the corresponding frame on the scene CS
  unsigned fSurfIndex{0};   ///< Surface index in the volume shell (can be optimized by compacting with fLogicId)
  NavIndex_t fState{0};     ///< sub-path navigation state id in the parent scene
  bool fNeverCheck{false};  ///< The frame should never be checked
  bool fEmbedded{false};    ///< The surface is embedded in the parent surface if any
  bool fEmbedding{true};    ///< The frame always embeds daughter state frames if on the same CS
  bool fOverlapping{false}; ///< The frame is overlapping another frame and requires a relocation after crossing
  bool fVirtualParent{false}; ///< The parent frame is a virtual surface of a boolean
  bool fSkipConvexity{false}; ///< whether the convexity check for booleans can be skipped (only the case for end caps
                              ///< of elliptical tubes)

  FramedSurface() = default;
  FramedSurface(UnplacedSurface<Real_t> const &unplaced, Frame const &frame, Transformation_t trans,
                NavIndex_t index = 0, const bool never_check = 0)
      : fSurface(unplaced), fFrame(frame), fTrans(trans), fState(index), fNeverCheck(never_check)
  {
  }

  template <typename Real_i>
  FramedSurface(const FramedSurface<Real_i, TransformationMP<Real_i>> &other)
      : fSurface(other.fSurface), fFrame(other.fFrame), fTrans(other.fTrans), fParent(other.fParent),
        fLogicId(other.fLogicId), fSceneCS(other.fSceneCS), fSceneCSind(other.fSceneCSind),
        fSurfIndex(other.fSurfIndex), fState(other.fState), fNeverCheck(other.fNeverCheck), fEmbedded(other.fEmbedded),
        fEmbedding(other.fEmbedding), fOverlapping(other.fOverlapping), fVirtualParent(other.fVirtualParent),
        fSkipConvexity(other.fSkipConvexity)
  {
  }

  /// Sorting by decreasing state depth and increasing state index
  bool operator<(FramedSurface const &other) const
  {
    using vecgeom::NavigationState;
    auto level1 = NavigationState::GetLevelImpl(fState);
    auto level2 = NavigationState::GetLevelImpl(other.fState);
    if (level1 > level2) return true;
    if (level1 < level2) return false;
    if (!fEmbedding && other.fEmbedding) return true;
    if (fEmbedding && !other.fEmbedding) return false;
    if (fState < other.fState) return true;
    if (fState > other.fState) return false;
    // If states are identical, sort by fSurfIndex
    if (fSurfIndex == other.fSurfIndex) VECGEOM_LOG(critical) << "### Found frames with same state and surface index";
    if (fSurfIndex < other.fSurfIndex) return true;
    return false;
  }

  template <typename DataContainer>
  VECCORE_ATT_HOST_DEVICE void Extent3D(Vector3D<Real_t> &aMin, Vector3D<Real_t> &aMax, DataContainer const &data) const
  {
    aMin.Set(0, 0, 0);
    aMax.Set(0, 0, 0);
    if (fFrame.type == FrameType::kNoFrame) {
      VECGEOM_LOG(critical) << "Extent3D should not be called for kNoFrame frames";
      return;
    }
    if (fNeverCheck) {
      // Special cases where the bounding box is computed based on the UnplacedSurface
      if (fSurface.type == SurfaceType::kTorus)
        SurfaceHelper<SurfaceType::kTorus, Real_t>(data.GetTorusData(fSurface.id)).Extent3D(aMin, aMax);
      else if (fSurface.type == SurfaceType::kArb4)
        SurfaceHelper<SurfaceType::kArb4, Real_t>(data.GetArb4Data(fSurface.id)).Extent3D(aMin, aMax);
    } else
      fFrame.Extent3D(aMin, aMax, data);
  }

  /// @brief Get the parent state index for the framed surface
  /// @param parent_state Parent state variable filled with the return value
  /// @return True if there is a parent, false if this is a top scene surface
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool GetParentState(NavIndex_t &parent_state) const
  {
    using vecgeom::NavigationState;
    parent_state = fState;
    NavigationState::PopImpl(parent_state);
    return (fState > 0);
  }

  /// @brief Get logical volume id for this frame
  /// @return Volume id.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int VolumeId() const
  {
    // We may need to cache the logical volume id in the surface directly
    return vecgeom::NavigationState::GetLogicalIdImpl(fState);
  }

  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_NOINLINE
  void PrintState() const { vecgeom::NavigationState::PrintTopImpl(fState); }

  /// Transform point and direction to the local frame
  void Transform(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, Vector3D<Real_t> &localpoint,
                 Vector3D<Real_t> &localdir) const
  {
    localpoint = fTrans.Transform(point);
    localdir   = fTrans.TransformDirection(dir);
  }

  ///< Check if the propagated point on surface is within the frame
  VECCORE_ATT_HOST_DEVICE bool InsideFrame(Vector3D<Real_t> const &point, SurfData<Real_t> const &surfdata) const
  {
    Vector3D<Real_t> localpoint(point);
    // For single-frame surfaces, fTrans is zero, so it may be worth testing this.
    if (!fTrans.IsIdentity()) localpoint = fTrans.Transform(point);
    return fFrame.Inside(localpoint, surfdata);
  }

  /// @brief Calculate the shortest distance (or an underestimate) from a point on the surface
  ///  to the surface frame.
  /// @tparam Real_t Precision type for parameters
  /// @param point Point on the surface
  /// @param safetySurf Safety to the surface
  /// @param surfdata Surface data storage
  /// @return Combined safety surface+frame
  VECCORE_ATT_HOST_DEVICE Real_t SafetyFrame(Vector3D<Real_t> const &point, Real_t safetySurf,
                                             SurfData<Real_t> const &surfdata, bool &valid) const
  {
    Vector3D<Real_t> localpoint(point);
    // For single-frame surfaces, fTrans is zero, so it may be worth testing this.
    if (!fTrans.IsIdentity()) localpoint = fTrans.Transform(point);
    return fFrame.Safety(localpoint, safetySurf, surfdata, valid);
  }

  /// @brief Calculate the shortest distance (or an underestimate) from a point on the surface
  ///  to the surface frame.
  /// @tparam Real_t Precision type for parameters
  /// @param point Point on the surface, in the surface reference frame
  /// @param safetySurf Safety to the surface
  /// @param surfdata Surface data storage
  /// @return Combined safety surface+frame
  VECCORE_ATT_HOST_DEVICE Real_t LocalSafetyFrame(Vector3D<Real_t> const &point, Real_t safetySurf,
                                                  SurfData<Real_t> const &surfdata, bool &valid) const
  {
    return fFrame.Safety(point, safetySurf, surfdata, valid);
  }
};

/// @brief A side represents all common placed surfaces
struct Side {
  int fNumParents{0};      ///< number of different parent volumes contributing to this side
  int fNsurf{0};           ///< Number of placed surfaces on this side
  int fDivision{-1};       ///< Division helper for the side
  int *fSurfaces{nullptr}; ///< [fNsurf] Array of placed surfaces on this side

  Side() = default;

  // Add existing placed surface to this side
  int AddSurface(int isurf)
  {
    // Re-allocate policy to keep memory footprint low
    // Sides have to be relocated for GPU in contiguous memory
    int *surfaces = new int[fNsurf + 1];
    for (auto i = 0; i < fNsurf; ++i)
      surfaces[i] = fSurfaces[i];
    surfaces[fNsurf++] = isurf;
    delete[] fSurfaces;
    fSurfaces = surfaces;
    return fNsurf - 1;
  }

  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE int GetTopSurfaceIndex() const
  {
    return (fNsurf > 0) ? GetSurfaceIndex(fNsurf - 1) : -1;
  }
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE int GetSurfaceIndex(int isurf) const { return fSurfaces[isurf]; }
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE bool HasChildren() const { return fNsurf > fNumParents; }

  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE FramedSurface<Real_t, vecgeom::Transformation2DMP<Real_t>> &GetSurface(
      int index, SurfData<Real_t> const &surfdata) const
  {
    return surfdata.fFramedSurf[fSurfaces[index]];
  }

  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE FramedSurface<Real_t, vecgeom::Transformation2DMP<Real_t>> const &
  TopSurface(SurfData<Real_t> const &surfdata) const
  {
    return surfdata.fFramedSurf[fSurfaces[fNsurf - 1]];
  }

  size_t size() const { return sizeof(Side) + fNsurf * sizeof(int); }

  void CopyTo(char *buffer)
  {
    // to be implemented
  }
};

template <typename Real_t, typename Real_i>
struct SideIterator {
  SliceCand *fSlice{nullptr}; ///< Division slice to be iterated
  int fNsurf{0};              ///< number of surfaces to iterate
  int fCrt{0};                ///< Current index

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SideIterator(const Side &side, Vector3D<Real_i> const &onsurf, SurfData<Real_t> const &surfdata, int increment = 1,
               int istart = 0)
      : fNsurf(side.fNsurf), fCrt(istart)
  {
    if (side.fDivision >= 0) {
      auto const &div = surfdata.fSideDivisions[side.fDivision];
      auto ind        = div.GetSliceIndex(onsurf);
      if (ind >= 0) {
        fSlice    = &div.fSlices[ind];
        fNsurf    = fSlice->fNcand;
        int inc01 = (increment + 1) >> 1; // [-1/1 -> 0/1]
        int start = (1 - inc01) * (fSlice->fNcand - 1);
        int end   = inc01 * (fSlice->fNcand - 1);
        for (fCrt = start; fCrt * increment < end; fCrt += increment)
          if (increment * fSlice->fCandidates[fCrt] >= increment * istart) break;
      }
    }
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool Done() const { return (fCrt < 0) || (fCrt >= fNsurf); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetNextIndex(int ind)
  {
    if (fSlice) {
      int i;
      for (i = fCrt + 1; i < fSlice->fNcand; ++i)
        if (fSlice->fCandidates[i] >= ind) break;
      fCrt = i - 1;
    } else {
      fCrt = ind - 1;
    }
  }

  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE int operator()() { return fSlice ? fSlice->fCandidates[fCrt] : fCrt; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SideIterator &operator++()
  {
    fCrt++;
    return *this;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SideIterator &operator--()
  {
    fCrt--;
    return *this;
  }
};

/// @brief A common surface made of two sides, having a global transformation.
template <typename Real_t>
struct CommonSurface {
  SurfaceType fType{SurfaceType::kPlanar}; ///< Type of surface
  int fSceneId{0};                         ///< Scene id. if negative, it is a top scene id
  TransformationMP<Real_t> fTrans;         ///< Transformation of the first left frame
  NavIndex_t fDefaultState{0};             ///< The default state for this surface (deepest mother)
  Side fLeftSide;                          ///< Left-side (behind normal)
  Side fRightSide;                         ///< Right-side (alongside normal)
  bool fFlipped;                           ///< whether the common surface is flipped due to boolean negation

  CommonSurface() = default;

  CommonSurface(SurfaceType type, int global_surf, bool flipped) : fType(type)
  {
    // Add by default the first surface to the left side
    fLeftSide.AddSurface(global_surf);
    fFlipped = flipped;
  };

  template <typename Real_i>
  CommonSurface(const CommonSurface<Real_i> &other)
      : fType(other.fType), fSceneId(other.fSceneId), fTrans(other.fTrans), fDefaultState(other.fDefaultState),
        fLeftSide(other.fLeftSide), fRightSide(other.fRightSide), fFlipped(other.fFlipped)
  {
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetSceneId() const { return std::abs(fSceneId); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsSceneSurface() const { return (fSceneId < 0); }

  ///< Get the normal to the surface from a point on surface
  void GetNormal(Vector3D<Real_t> const &point, Vector3D<Real_t> &normal, SurfData<Real_t> const &surfdata) const
  {
    Vector3D<Real_t> localnorm;
    // point to local frame
    auto const &trans      = fTrans;
    auto localpoint        = trans.Transform(point);
    auto const &framedsurf = fLeftSide.GetSurface(0, surfdata);
    framedsurf.fSurface.GetNormal(localpoint, localnorm, surfdata);
    trans.InverseTransformDirection(localnorm, normal);
  }
};

/// @brief A volume shell holding indices for all placed surfaces belonging to a volume.
struct VolumeShell {
  LogicExpression fLogic;                   ///< Logic expression for local surfaces
  int fBVH{0};                              ///< The BVH index for this logical volume
  int fNsurf{0};                            ///< Number of local surfaces
  int fNExitingSurfaces{0};                 ///< Number of framed exiting surfaces
  int fNEnteringSurfaces{0};                ///< Number of framed entering surfaces
  int fNDaughterPvols{0};                   ///< Number of daughter volumes
  int *fSurfaces{nullptr};                  ///< Local surface id's
  int *fExitingSurfaces{nullptr};           ///< Indices of real surfaces in the fSurfaces array
  int *fEnteringSurfaces{nullptr};          ///< List of framed entering surfaces
  int *fEnteringSurfacesPvol{nullptr};      ///< Pvol id per framed surface
  int *fEnteringSurfacesPvolTrans{nullptr}; ///< Array of ids to daughter placed volume transformations
  int *fEnteringSurfacesLvolIds{nullptr};   ///< Array of ids to daughter logical volumes
  int *fDaughterPvolIds{nullptr};           ///< Global PV Ids of the daughter PVs of this Volume
  int *fDaughterPvolTrans{nullptr};         ///< Transformations of the daughter PVs of this Volume
};
} // namespace vgbrep

#endif
