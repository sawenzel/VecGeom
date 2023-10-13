#ifndef VECGEOM_SURFACE_MODEL_H_
#define VECGEOM_SURFACE_MODEL_H_

#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/surfaces/base/Equations.h>
#include <VecGeom/surfaces/surf/SurfaceImpl.h>
#include <VecGeom/surfaces/mask/FrameMasks.h>
#include <VecGeom/surfaces/cuda/DeviceStorage.h>

// #include <VecGeom/base/Vector3D.h>

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
struct UnplacedSurface {
  SurfaceType type{kPlanar}; ///< surface type
  int id{-1};                ///< surface id

  UnplacedSurface() = default;
  UnplacedSurface(SurfaceType stype, int sid = -1)
  {
    type = stype;
    id   = sid;
  }

  /// @brief A local point is inside if behind the normal within tolerance
  /// @tparam Real_t Floating-point precision type
  /// @param point Point in the local surface coordinates
  /// @return Inside half-space
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE bool Inside(Vector3D<Real_t> const &point, SurfData<Real_t> const &surfdata) const
  {
    switch (type) {
    case kPlanar:
      return SurfaceHelper<kPlanar, Real_t>().Inside(point);
    case kCylindrical:
      return SurfaceHelper<kCylindrical, Real_t>(surfdata.GetCylData(id)).Inside(point);
    case kConical:
      return SurfaceHelper<kConical, Real_t>(surfdata.GetConeData(id)).Inside(point);
    case kSpherical:
      return SurfaceHelper<kSpherical, Real_t>(surfdata.GetSphData(id)).Inside(point);
    case kTorus:
    case kGenSecondOrder:
      // unhandled
      return false;
    };
    return false;
  }

  /// @brief Find signed distance to next intersection from local point.
  /// @tparam Real_t Floating-point precision type
  /// @param point Point in the local surface coordinates
  /// @param dir Direction in the local surface coordinates
  /// @param left_side Flag specifying if the surface is intersected from the left-side that defines the normal
  /// @param surfdata Surface data storage.
  /// @param distance Computed distance to surface
  /// @return Validity of the intersection
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE bool Intersect(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, bool left_side,
                                         SurfData<Real_t> const &surfdata, Real_t &distance) const
  {
    switch (type) {
    case kPlanar:
      return SurfaceHelper<kPlanar, Real_t>().Intersect(point, dir, left_side, distance);
    case kCylindrical:
      return SurfaceHelper<kCylindrical, Real_t>(surfdata.GetCylData(id)).Intersect(point, dir, left_side, distance);
    case kConical:
      return SurfaceHelper<kConical, Real_t>(surfdata.GetConeData(id)).Intersect(point, dir, left_side, distance);
    case kSpherical:
      return SurfaceHelper<kSpherical, Real_t>(surfdata.GetSphData(id)).Intersect(point, dir, left_side, distance);
    case kTorus:
    case kGenSecondOrder:
      // unhandled
      return false;
    };
    return false;
  }

  /// @brief Computes the isotropic safe distance to unplaced surfaces
  /// @tparam Real_t Precision type for parameters
  /// @param point Point in local surface coordinates
  /// @param left_side Flag specifying if the surface is intersected from the left-side that defines the normal
  /// @param surfdata Surface data storage
  /// @param distance Computed isotropic safety
  /// @param compute_onsurf Instructs to compute the projection of the point on surface
  /// @param onsurf Projection of the point on surface
  /// @return
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE bool Safety(Vector3D<Real_t> const &point, bool left_side, SurfData<Real_t> const &surfdata,
                                      Real_t &distance, bool compute_onsurf, Vector3D<Real_t> &onsurf) const
  {
    switch (type) {
    case kPlanar:
      return SurfaceHelper<kPlanar, Real_t>().Safety(point, left_side, distance, compute_onsurf, onsurf);
    case kCylindrical:
      return SurfaceHelper<kCylindrical, Real_t>(surfdata.GetCylData(id))
          .Safety(point, left_side, distance, compute_onsurf, onsurf);
    case kConical:
      return SurfaceHelper<kConical, Real_t>(surfdata.GetConeData(id))
          .Safety(point, left_side, distance, compute_onsurf, onsurf);
    case kSpherical:
      return SurfaceHelper<kSpherical, Real_t>(surfdata.GetSphData(id))
          .Safety(point, left_side, distance, compute_onsurf, onsurf);
    case kTorus:
    case kGenSecondOrder:
      // unhandled
      return false;
    };
    return false;
  }
};

/// @brief A frame delimiting the real solid surface on an infinite half-space
struct Frame {
  FrameType type{kWindow}; ///< frame type
  int id{-1};              ///< frame mask id

  Frame() = default;
  Frame(FrameType mtype, int mid = -1) : type(mtype), id(mid) {}

  // A function to check if local point is within the Frame's mask.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE bool Inside(Vector3D<Real_t> const &local, SurfData<Real_t> const &surfdata) const
  {
    switch (type) {
    case kRing:
      return surfdata.GetRingMask(id).Inside(local);
    case kZPhi:
      return surfdata.GetZPhiMask(id).Inside(local);
    case kWindow:
      return surfdata.GetWindowMask(id).Inside(local);
    // TODO: Support these
    case kTriangle:
      return surfdata.GetTriangleMask(id).Inside(local);
    case kQuadrilateral:
      return surfdata.GetQuadMask(id).Inside(local);
    case kRangeZ:
      /*return (local[2] > vecgeom::MakeMinusTolerant<true>(u[0]) &&
              local[2] < vecgeom::MakePlusTolerant<true>(u[1]));*/
    case kRangeSph:
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
    case kRing:
      return surfdata.GetRingMask(id).Safety(local, safetySurf, valid);
    case kZPhi:
      return surfdata.GetZPhiMask(id).Safety(local, safetySurf, valid);
    case kWindow:
      return surfdata.GetWindowMask(id).Safety(local, safetySurf, valid);
    case kTriangle:
      return surfdata.GetTriangleMask(id).Safety(local, safetySurf, valid);
    case kQuadrilateral:
      return surfdata.GetQuadMask(id).Safety(local, safetySurf, valid);
    case kRangeZ:
    case kRangeSph:
    default:
      // unhandled
      valid = false;
    };
    return Real_t(0);
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
struct FramedSurface {
  UnplacedSurface fSurface;   ///< Surface identifier
  Frame fFrame;               ///< Frame
  int fTrans{-1};             ///< Transformation of the surface in the compacted sub-hierarchy top volume frame
  int fParent{-1};            ///< Topmost parent frame index on the common surface
  int fLogicId{0};            ///< Logic flag for surface:
                              ///<   0        = non-Bool
                              ///<   positive = true logic surface
                              ///<   negative = negated logic surface
  NavIndex_t fState{0};       ///< sub-path navigation state id in the parent scene
  bool fUseSurfSafety{false}; ///< The surface has virtual intersections with the 3D shape. Use just the surface safety
                              ///< to outside in the minimization procedure

  FramedSurface() = default;
  FramedSurface(UnplacedSurface const &unplaced, Frame const &frame, int trans, bool surfsafety, NavIndex_t index = 0)
      : fSurface(unplaced), fFrame(frame), fTrans(trans), fState(index), fUseSurfSafety(surfsafety)
  {
  }

  /// Sorting by decreasing state depth and increasing state index
  bool operator<(FramedSurface const &other) const
  {
    using vecgeom::NavStateIndex;
    auto level1 = NavStateIndex::GetLevelImpl(fState);
    auto level2 = NavStateIndex::GetLevelImpl(other.fState);
    if (level1 > level2)
      return true;
    else if (level1 < level2)
      return false;
    if (fState < other.fState) return true;
    return false;
  }

  /// @brief Get logical volume id for this frame
  /// @return Volume id.
  VECCORE_ATT_HOST_DEVICE inline int VolumeId() const
  {
    // We may need to cache the logical volume id in the surface directly
    return vecgeom::NavStateIndex::TopImpl(fState)->GetLogicalVolume()->id();
  }

  /// Transform point and direction to the local frame
  template <typename Real_t>
  void Transform(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, Vector3D<Real_t> &localpoint,
                 Vector3D<Real_t> &localdir, SurfData<Real_t> const &surfdata) const
  {
    auto &localRef = surfdata.LocalT(fTrans);
    localpoint     = localRef.Transform(point);
    localdir       = localRef.TransformDirection(dir);
  }

  ///< Check if the propagated point on surface is within the frame
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE bool InsideFrame(Vector3D<Real_t> const &point, SurfData<Real_t> const &surfdata) const
  {
    Vector3D<Real_t> localpoint(point);
    // For single-frame surfaces, fTrans is zero, so it may be worth testing this.
    if (fTrans) localpoint = surfdata.fGlobalTrans[fTrans].Transform(point);
    return fFrame.Inside(localpoint, surfdata);
  }

  /// @brief Calculate the shortest distance (or an underestimate) from a point on the surface
  ///  to the surface frame.
  /// @tparam Real_t Precision type for parameters
  /// @param point Point on the surface
  /// @param safetySurf Safety to the surface
  /// @param surfdata Surface data storage
  /// @return Combined safety surface+frame
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE Real_t SafetyFrame(Vector3D<Real_t> const &point, Real_t safetySurf,
                                             SurfData<Real_t> const &surfdata, bool &valid) const
  {
    Vector3D<Real_t> localpoint(point);
    // For single-frame surfaces, fTrans is zero, so it may be worth testing this.
    if (fTrans) localpoint = surfdata.fGlobalTrans[fTrans].Transform(point);
    return fFrame.Safety(localpoint, safetySurf, surfdata, valid);
  }
};

/// @brief A list of candidate surfaces
struct Candidates {
  int fNcand{0};             ///< Number of candidate surfaces
  int fNEntering{0};          ///< Number of Entering candidate surfaces. fNcand = NEntering + NExiting
  int *fCandidates{nullptr}; ///< [fNcand] Array of candidates
  int *fFrameInd{nullptr};   ///< [fNcand] Start index of the frame contributed by the touchable on the common surface

  VECCORE_ATT_HOST_DEVICE
  int operator[](int i) const { return fCandidates[i]; }
  VECCORE_ATT_HOST_DEVICE
  int operator[](int i) { return fCandidates[i]; }

  Candidates() = default;
};

/// @brief A side represents all common placed surfaces
struct Side {
  Extent fExtent;          ///< Extent on a side.
  int fNumParents{0};      ///< number of different parent volumes contributing to this side
  int fNsurf{0};           ///< Number of placed surfaces on this side
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

  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE inline FramedSurface const &GetSurface(int index, SurfData<Real_t> const &surfdata) const
  {
    return surfdata.fFramedSurf[fSurfaces[index]];
  }

  size_t size() const { return sizeof(Side) + fNsurf * sizeof(int); }

  void CopyTo(char *buffer)
  {
    // to be implemented
  }
};

/// @brief A common surface made of two sides, having a global transformation.
struct CommonSurface {
  SurfaceType fType{kPlanar};  ///< Type of surface
  int fTrans{-1};              ///< Transformation of the first left frame
  NavIndex_t fDefaultState{0}; ///< The default state for this surface (deepest mother)
  Side fLeftSide;              ///< Left-side portal side id (behind normal)
  Side fRightSide;             ///< Right-side portal side id (alongside normal)

  CommonSurface() = default;

  CommonSurface(SurfaceType type, int global_surf) : fType(type)
  {
    // Add by default the first surface to the left side
    fLeftSide.AddSurface(global_surf);
  };

  ///< Get the normal to the surface from a point on surface
  template <typename Real_t>
  void GetNormal(Vector3D<Real_t> const &point, Vector3D<Real_t> &normal, SurfData<Real_t> const &surfdata) const
  {
    Vector3D<Real_t> localnorm;
    // point to local frame
    auto const &trans      = surfdata.fGlobalTrans[fTrans];
    auto localpoint        = trans.Transform(point);
    auto const &framedsurf = fLeftSide.GetSurface(0, surfdata);
    framedsurf.fSurface.GetNormal(localpoint, localnorm, surfdata);
    trans.InverseTransformDirection(localnorm, normal);
  }
};

/// @brief A volume shell holding indices for all placed surfaces belonging to a volume.
struct VolumeShell {
  int fNsurf;              ///< Number of local surfaces
  int *fSurfaces{nullptr}; ///< Local surface id's
  LogicExpression fLogic;  ///< Logic expression for local surfaces

  /// @brief Check if a point is inside the volume defined by surfaces
  /// @tparam Real_t Floating-point precision type
  /// @param point Point in the local volume coordinates
  /// @return Inside volume
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE bool Inside(Vector3D<Real_t> const &point, SurfData<Real_t> const &surfdata)
  {
    /*** IMPORTANT ***/
    // The current implementation works only if a volume is a Boolean intersection (logical AND)
    // of the half-spaces represented by its surfaces. In future we need a proper logical evaluator
    //****************/
    Vector3D<Real_t> local;
    // This loop is less efficient than the specialized shape treatment, but this is
    // not important since the Inside function is called only once per track in the surface model
    for (int isurf = 0; isurf < fNsurf; ++isurf) {
      local                = surfdata.fLocalTrans[fSurfaces[isurf]].Transform(point);
      auto const &unplaced = surfdata.fLocalSurf[fSurfaces[isurf]].fSurface;
      if (!unplaced.Inside(local, surfdata)) return false;
    }
    return true;
  }
};

class BVH;

// A level of the geometry setup with a coordinate system and multiple volumes
// Currently called 'Universe' in Orange (name taken from MCNP, other codes)
// A detector or setup will be composed of one or two levels of Scene
struct Scene {
  CommonSurface *fSurfaces{nullptr}; // Doors to other scenes
  BVH *fNavigator{nullptr};
};
// How we decompose scene in hierarchical Scenes
//

/*
 * The main surface storage utility, providing access by index to:
 *    * global and local transformations applied to surfaces
 *    * surface data per surface type
 *    * mask data per mask type
 *    * imprint data (list of masks)
 */
template <typename Real_t>
struct SurfData {

  using CylData_t      = CylData<Real_t>;
  using ConeData_t     = ConeData<Real_t>;
  using SphData_t      = SphData<Real_t>;
  using WindowMask_t   = WindowMask<Real_t>;
  using RingMask_t     = RingMask<Real_t>;
  using ZPhiMask_t     = ZPhiMask<Real_t>;
  using TriangleMask_t = TriangleMask<Real_t>;
  using QuadMask_t     = QuadrilateralMask<Real_t>;

  int fNlocalTrans{0};
  int fNglobalTrans{0};
  int fNlocalSurf{0};
  int fNglobalSurf{0};
  int fNcommonSurf{0};
  int fNsides{0};
  int fNStates{0};
  int fSizeCandList{0};
  int fNcylsph{0};
  int fNcone{0};
  int fNshells{0};
  int fNlogic{0};
  int fNrange{0};
  int fNwindows{0};
  int fNrings{0};
  int fNzphis{0};
  int fNtriangs{0};
  int fNquads{0};

  /// Transformations. A portal transformation is a tuple global + local
  Transformation *fLocalTrans{nullptr};  ///< Local surface transformations per logical volume
  Transformation *fGlobalTrans{nullptr}; ///< Touchable global transformations

  /// Cylindrical surface data (radius)
  CylData_t *fCylSphData{nullptr}; ///< Cyl and sphere data
  ConeData_t *fConeData{nullptr};  ///< Cone data

  /// Volume shells, indexed by the logical volume id
  VolumeShell *fShells{nullptr}; ///< volume shells

  FramedSurface *fLocalSurf{nullptr};      ///< local surfaces
  FramedSurface *fFramedSurf{nullptr};     ///< global surfaces
  WindowMask_t *fWindowMasks{nullptr};     ///< rectangular masks
  RingMask_t *fRingMasks{nullptr};         ///< ring masks
  ZPhiMask_t *fZPhiMasks{nullptr};         ///< cylindrical masks
  TriangleMask_t *fTriangleMasks{nullptr}; ///< triangular masks
  QuadMask_t *fQuadMasks{nullptr};         ///< quadrilateral masks
  CommonSurface *fCommonSurfaces{nullptr}; ///< common surfaces
  Candidates *fCandidates;                 ///< candidate surfaces per navigation state
  int *fSides{nullptr};                    ///< side surface indices
  int *fSurfShellList{nullptr};            ///< indices of local surfaces used in shells
  logic_int *fLogicList{nullptr};          ///< list of logic expressions per volume
  int *fCandList{nullptr};                 ///< global list of candidate indices

  VECCORE_ATT_HOST_DEVICE
  static inline SurfData<Real_t> &Instance()
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    return *globaldevicesurfdata::gSurfDataDevice<Real_t>;
#else
    static SurfData<Real_t> gSurfData;
    return gSurfData;
#endif
  }

  /// Surface data accessors by component id
  VECCORE_ATT_HOST_DEVICE
  CylData_t const &GetCylData(int id) const { return fCylSphData[id]; }
  VECCORE_ATT_HOST_DEVICE
  SphData_t const &GetSphData(int id) const { return fCylSphData[id]; }
  VECCORE_ATT_HOST_DEVICE
  ConeData_t const &GetConeData(int id) const { return fConeData[id]; }
  VECCORE_ATT_HOST_DEVICE
  WindowMask_t const &GetWindowMask(int id) const { return fWindowMasks[id]; }
  VECCORE_ATT_HOST_DEVICE
  RingMask_t const &GetRingMask(int id) const { return fRingMasks[id]; }
  VECCORE_ATT_HOST_DEVICE
  ZPhiMask_t const &GetZPhiMask(int id) const { return fZPhiMasks[id]; }
  VECCORE_ATT_HOST_DEVICE
  TriangleMask_t const &GetTriangleMask(int id) const { return fTriangleMasks[id]; }
  VECCORE_ATT_HOST_DEVICE
  QuadMask_t const &GetQuadMask(int id) const { return fQuadMasks[id]; }

  // Accessors by common surface id
  VECCORE_ATT_HOST_DEVICE
  UnplacedSurface const &GetUnplaced(int isurf, bool &flipped) const
  {
    FramedSurface const &surf_frame = fFramedSurf[fCommonSurfaces[isurf].fLeftSide.fSurfaces[0]];
    flipped                         = surf_frame.fLogicId < 0;
    return surf_frame.fSurface;
  }

  // private:
  SurfData() = default;
};

} // namespace vgbrep

#endif
