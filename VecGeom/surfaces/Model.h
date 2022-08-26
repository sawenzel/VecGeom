#ifndef VECGEOM_SURFACE_MODEL_H_
#define VECGEOM_SURFACE_MODEL_H_

#include <iostream>

#include <VecGeom/surfaces/Equations.h>
#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/surfaces/FrameMasks.h>
//#include <VecGeom/base/Vector3D.h>

namespace vgbrep {
///< Supported surface types
enum SurfaceType { kPlanar, kCylindrical, kConical, kSpherical, kTorus, kGenSecondOrder };

///< Supported frame types
///> kRangeZ      <- range along z-axis
///> kRing        <- a "ring" range on a plane
///> kZPhi        <- z and phi range on a cylinder
///> kRangeSph    <- theta and phi range on a sphere
///> kWindow      <- rectangular range in xy-plane
///> kTriangle    <- triangular range in xy-plane
enum FrameType { kRangeZ, kRing, kZPhi, kRangeSph, kWindow, kTriangle, kQuadrilateral };

///< VecGeom type aliases
template <typename Real_t>
using Vector3D = vecgeom::Vector3D<Real_t>;

using Transformation = vecgeom::Transformation3D;

///< Forward declaration of surface data structure used in navigation
template <typename Real_t>
struct SurfData;

struct Frame;
using Extent = Frame;

///<
/*
   Unplaced half-space surface type. The actual surface data pointed by the surface id
   is stored in a separate SurfData structure
   - All unplaced planes are (xOy), having the oriented normal on positive z
   - Unplaced cylinders, cones and tori have the z axis as axis of symmetry. Normal pointing outwards
   - Unplaced spheres have the origin as center

   To get the distance from a global point/direction to the surface one needs to first convert
   those to the local frame
 */
struct UnplacedSurface {
  SurfaceType type{kPlanar}; ///< surface type
  int id{-1};                ///< surface id

  UnplacedSurface() = default;
  UnplacedSurface(SurfaceType stype, int sid = -1)
  {
    type = stype;
    id   = sid;
  }

  /// Find positive distance to next intersection from local point
  template <typename Real_t>
  bool Intersect(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, bool exiting, bool flip_normal,
                 SurfData<Real_t> const &surfdata, Real_t &distance) const
  {
    QuadraticCoef<Real_t> coef;
    Vector3D<Real_t> normal;
    Real_t roots[2];
    int numroots = 0;
    bool flipped = flip_normal;

    switch (type) {
    case kPlanar:
      // Just need to propagate to (xOy) plane
      distance = -point[2] / vecgeom::NonZero(dir[2]); // Division by zero?
      if (flipped)
        return exiting ^ (dir[2] > 0);
      else
        return exiting ^ (dir[2] < 0);
    case kCylindrical:
      // Intersect with the cylindrical surface having Z as axis of symmetry
      flipped ^= surfdata.GetCylData(id).IsFlipped();
      CylinderEq<Real_t>(point, dir, surfdata.GetCylData(id).Radius(), coef);
      QuadraticSolver(coef, roots, numroots);
      for (auto i = 0; i < numroots; ++i) {
        distance                = roots[i];
        Vector3D<Real_t> onsurf = point + distance * dir;
        normal.Set(onsurf[0], onsurf[1], 0);
        if (flipped) normal *= -1;
        bool hit = exiting ^ (dir.Dot(normal) < 0);
        // First solution giving a valid hit wins
        if (hit) return true;
      }
      return false;
    case kConical:
      // Intersect with the conical surface having Z as axis of symmetry
      flipped ^= surfdata.GetConeData(id).IsFlipped();
      ConeEq<Real_t>(point, dir, surfdata.GetConeData(id).Radius(), surfdata.GetConeData(id).Slope(), coef);
      QuadraticSolver(coef, roots, numroots);
      for (auto i = 0; i < numroots; ++i) {
        distance                = roots[i];
        Vector3D<Real_t> onsurf = point + distance * dir;
        normal.Set(onsurf[0], onsurf[1],
                   -std::sqrt(onsurf[0] * onsurf[0] + onsurf[1] * onsurf[1]) * surfdata.GetConeData(id).Slope());
        if (flipped) normal *= -1;
        bool hit = exiting ^ (dir.Dot(normal) < 0);
        // First solution giving a valid hit wins
        if (hit) return true;
      }
      return false;
    case kSpherical:
      // Intersect with the sphere having the center in the origin
      flipped ^= surfdata.GetSphData(id).IsFlipped();
      SphereEq<Real_t>(point, dir, surfdata.GetSphData(id).Radius(), coef);
      QuadraticSolver(coef, roots, numroots);
      for (auto i = 0; i < numroots; ++i) {
        distance                = roots[i];
        Vector3D<Real_t> onsurf = point + distance * dir;
        normal.Set(onsurf[0], onsurf[1], onsurf[2]);
        if (flipped) normal *= -1;
        bool hit = exiting ^ (dir.Dot(normal) < 0);
        // First solution giving a valid hit wins
        if (hit) return true;
      }
      return false;
    case kTorus:
    case kGenSecondOrder:
      std::cout << "kTorus, kGenSecondOrder unhandled\n";
      return false;
    };
    return false;
  }

  /// @brief Computes the isotropic safe distance to unplaced surfaces
  /// @tparam Real_t Precision type for parameters
  /// @param point Point in local surface coordinates
  /// @param surfdata Surface data storage
  /// @param distance Computed isotropic safety
  /// @param onsurf Projection of the point on surface
  /// @return Success of the calculation
  template <typename Real_t>
  bool Safety(Vector3D<Real_t> const &point, bool exiting, bool flip_normal, SurfData<Real_t> const &surfdata,
              Real_t &distance, bool compute_onsurf, Vector3D<Real_t> &onsurf) const
  {
    Real_t rho;
    switch (type) {
    case kPlanar:
      distance = exiting ? -point[2] : point[2];
      // Computing onsurf is cheap
      onsurf.Set(point[0], point[1], 0);
      return true;
    case kCylindrical: {
      Real_t cylR = surfdata.GetCylData(id).Radius();
      rho         = point.Perp();
      distance    = exiting ^ flip_normal ? cylR - rho : rho - cylR;
      // Cannot project if the point is on the center of the cylinder
      if (distance > -vecgeom::kTolerance && compute_onsurf) {
        onsurf.Set(0, 0, 0);
        if (rho > vecgeom::kTolerance) {
          auto invrho = 1. / rho;
          onsurf.Set(point[0] * invrho, point[1] * invrho, point[2]);
        }
      }
      return true;
    }
    case kConical: {
      Real_t t       = surfdata.GetConeData(id).Slope();
      Real_t coneR   = std::abs(surfdata.GetConeData(id).Radius() + point[2] * t);
      rho            = point.Perp();
      auto distanceR = exiting ^ flip_normal ? coneR - rho : rho - coneR;
      Real_t calf    = Real_t(1) / std::sqrt(Real_t(1) + t * t);
      distance       = distanceR * calf;
      // the onsurf computation code is missing below

      return true;
    }
    case kSpherical: {
      Real_t sphR = surfdata.GetSphData(id).Radius();
      rho         = point.Mag();
      distance    = exiting ^ flip_normal ? sphR - rho : rho - sphR;
      // the onsurf computation code is missing below
      return true;
    }
    case kTorus:
    case kGenSecondOrder:
      std::cout << "kTorus, kGenSecondOrder unhandled\n";
      return false;
    };
    return false;
  }

  /// Get normal direction to the surface in a point on surface
  template <typename Real_t>
  void GetNormal(Vector3D<Real_t> const &point, Vector3D<Real_t> &normal, SurfData<Real_t> const &surfdata) const
  {
    bool flipped = false;
    switch (type) {
    case kPlanar:
      // Just return (0, 0, 1);
      normal.Set(0, 0, 1);
      // This cannot be flipped and already normalized, so return
      return;
    case kCylindrical:
      // Return normal direction outwards.
      normal.Set(point[0], point[1], 0);
      flipped = surfdata.GetCylData(id).IsFlipped();
      break;
    case kConical:
      // Return normal direction outwards.
      normal.Set(point[0], point[1],
                 -std::sqrt(point[0] * point[0] + point[1] * point[1]) * surfdata.GetConeData(id).Slope());
      flipped = surfdata.GetConeData(id).IsFlipped();
      break;
    case kSpherical:
      // Return normal direction outwards.
      normal.Set(point[0], point[1], point[2]);
      flipped = surfdata.GetSphData(id).IsFlipped();
      break;
    case kTorus:
    case kGenSecondOrder:
      normal.Set(0, 0, 1);
      std::cout << "kTorus, kGenSecondOrder unhandled\n";
    };
    if (flipped) normal *= -1;
    // For dot product checks we need un-normalized normals. Add a flag to the interface?
    normal.Normalize();
  }
};

/* An frame delimiting the real surface on an infinite half-space */
struct Frame {
  FrameType type{kWindow}; ///< frame type
  int id{-1};              ///< frame mask id

  Frame() = default;
  Frame(FrameType mtype, int mid) : type(mtype), id(mid) {}

  // Mask getters for various mask types

  template <typename Real_t>
  void GetMask(WindowMask<Real_t> &mask, SurfData<Real_t> const &surfdata) const
  {
    surfdata.fWindowMasks[id].GetMask(mask);
  }

  template <typename Real_t>
  void GetMask(RingMask<Real_t> &mask, SurfData<Real_t> const &surfdata)
  {
    surfdata.fRingMasks[id].GetMask(mask);
  }

  template <typename Real_t>
  void GetMask(ZPhiMask<Real_t> &mask, SurfData<Real_t> const &surfdata)
  {
    surfdata.fZPhiMasks[id].GetMask(mask);
  }

  template <typename Real_t>
  void GetMask(TriangleMask<Real_t> &mask, SurfData<Real_t> const &surfdata)
  {
    surfdata.fTriangleMasks[id].GetMask(mask);
  }

  template <typename Real_t>
  void GetMask(QuadrilateralMask<Real_t> &mask, SurfData<Real_t> const &surfdata)
  {
    surfdata.fQuadMasks[id].GetMask(mask);
  }

  // A function to check if local point is within the Frame's mask.
  template <typename Real_t>
  bool Inside(Vector3D<Real_t> const &local, SurfData<Real_t> const &surfdata) const
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
      std::cout << "Frame type not supported." << std::endl;
      break;
    };
    return false;
  }

  // A function dispatcher to compute the safety for the frame.
  template <typename Real_t>
  Real_t Safety(Vector3D<Real_t> const &local, SurfData<Real_t> const &surfdata) const
  {
    switch (type) {
    case kRing:
      return surfdata.GetRingMask(id).Safety(local);
    case kZPhi:
      return surfdata.GetZPhiMask(id).Safety(local);
    case kWindow:
      return surfdata.GetWindowMask(id).Safety(local);
    // TODO: Support these
    case kTriangle:
      return surfdata.GetTriangleMask(id).Safety(local);
    case kQuadrilateral:
      return surfdata.GetQuadMask(id).Safety(local);
    case kRangeZ:
    case kRangeSph:
    default:
      std::cout << "Frame type not supported." << std::endl;
      break;
    };
    return false;
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

/* A placed surface on a scene having a frame and a navigation state associated to a touchable */
struct FramedSurface {
  UnplacedSurface fSurface;   ///< Surface identifier
  Frame fFrame;               ///< Frame
  int fTrans{-1};             ///< Transformation of the surface in the compacted sub-hierarchy top volume frame
  int fParent{-1};            ///< Topmost parent frame index on the common surface
  NavIndex_t fState{0};       ///< sub-path navigation state id in the parent scene
  bool fUseSurfSafety{false}; ///< Use just the surface safety to outside in the minimization procedure

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
  bool InsideFrame(Vector3D<Real_t> const &point, SurfData<Real_t> const &surfdata) const
  {
    Vector3D<Real_t> localpoint(point);
    // For single-frame surfaces, fTrans is zero, so it may be worth testing this.
    if (fTrans) localpoint = surfdata.fGlobalTrans[fTrans].Transform(point);
    return fFrame.Inside(localpoint, surfdata);
  }

  ///< Check if the propagated point on surface is within the frame
  template <typename Real_t>
  Real_t SafetyFrame(Vector3D<Real_t> const &point, SurfData<Real_t> const &surfdata) const
  {
    Vector3D<Real_t> localpoint(point);
    // For single-frame surfaces, fTrans is zero, so it may be worth testing this.
    if (fTrans) localpoint = surfdata.fGlobalTrans[fTrans].Transform(point);
    return fFrame.Safety(localpoint, surfdata);
  }
};

///< A list of candidate surfaces
struct Candidates {
  int fNcand{0};             ///< Number of candidate surfaces
  int *fCandidates{nullptr}; ///< [fNcand] Array of candidates
  int *fFrameInd{nullptr};   ///< [fNcand] Framed surface indices for each candidate

  int operator[](int i) const { return fCandidates[i]; }
  int operator[](int i) { return fCandidates[i]; }

  Candidates() = default;
};

///< A side represents all common placed surfaces
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
  inline FramedSurface const &GetSurface(int index, SurfData<Real_t> const &surfdata) const
  {
    return surfdata.fFramedSurf[fSurfaces[index]];
  }

  size_t size() const { return sizeof(Side) + fNsurf * sizeof(int); }

  void CopyTo(char *buffer)
  {
    // to be implemented
  }
};

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

  using CylData_t           = CylData<Real_t>;
  using ConeData_t          = ConeData<Real_t>;
  using SphData_t           = SphData<Real_t>;
  using WindowMask_t        = WindowMask<Real_t>;
  using RingMask_t          = RingMask<Real_t>;
  using ZPhiMask_t          = ZPhiMask<Real_t>;
  using TriangleMask_t      = TriangleMask<Real_t>;
  using QuadrilateralMask_t = QuadrilateralMask<Real_t>;

  int fNglobalTrans{0};
  int fNglobalSurf{0};
  int fNcommonSurf{0};
  int fNcylsph{0};
  int fNcone{0};
  int fNrange{0};
  int fNwindows{0};
  int fNrings{0};
  int fNzphis{0};
  int fNtriangs{0};
  int fNquads{0};

  /// Transformations. A portal transformation is a tuple global + local
  Transformation *fGlobalTrans{nullptr}; ///< Touchable global transformations

  /// Cylindrical surface data (radius)
  CylData_t *fCylSphData{nullptr}; ///< Cyl and sphere data
  ConeData_t *fConeData{nullptr};  ///< Cone data

  FramedSurface *fFramedSurf{nullptr};      ///< global surfaces
  WindowMask_t *fWindowMasks{nullptr};      ///< rectangular masks
  RingMask_t *fRingMasks{nullptr};          ///< ring masks
  ZPhiMask_t *fZPhiMasks{nullptr};          ///< cylindrical masks
  TriangleMask_t *fTriangleMasks{nullptr};  ///< triangular masks
  QuadrilateralMask_t *fQuadMasks{nullptr}; ///< quadrilateral masks
  CommonSurface *fCommonSurfaces{nullptr};  ///< common surfaces
  Candidates *fCandidates;                  ///< candidate surfaces per navigation state
  int *fSides{nullptr};                     ///< side surface indices
  int *fCandList{nullptr};                  ///< global list of candidate indices

  SurfData() = default;

  /// Surface data accessors by component id
  CylData_t const &GetCylData(int id) const { return fCylSphData[id]; }
  SphData_t const &GetSphData(int id) const { return fCylSphData[id]; }
  ConeData_t const &GetConeData(int id) const { return fConeData[id]; }
  WindowMask_t const &GetWindowMask(int id) const { return fWindowMasks[id]; }
  RingMask_t const &GetRingMask(int id) const { return fRingMasks[id]; }
  ZPhiMask_t const &GetZPhiMask(int id) const { return fZPhiMasks[id]; }
  TriangleMask_t const &GetTriangleMask(int id) const { return fTriangleMasks[id]; }
  QuadrilateralMask_t const &GetQuadMask(int id) const { return fQuadMasks[id]; }

  // Accessors by common surface id
  UnplacedSurface const GetUnplaced(int isurf) const
  {
    FramedSurface surf_frame = fFramedSurf[fCommonSurfaces[isurf].fLeftSide.fSurfaces[0]];
    return surf_frame.fSurface;
  }
};

} // namespace vgbrep

#endif
