#ifndef VECGEOM_SURFACE_BREPHELPER_H_
#define VECGEOM_SURFACE_BREPHELPER_H_

#include <cassert>
#include <functional>
#include <VecGeom/surfaces/Model.h>

// Check if math necessary
#include "VecGeom/base/Math.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeom/management/GeoManager.h"

namespace vgbrep {

template <typename Real_t>
constexpr Real_t Tolerance()
{
  return 0;
}

template <>
constexpr double Tolerance()
{
  return 1.e-9;
}

template <>
constexpr float Tolerance()
{
  return 1.e-4;
}

template <typename Real_t>
bool ApproxEqual(Real_t t1, Real_t t2)
{
  return std::abs(t1 - t2) <= Tolerance<Real_t>();
}

template <typename Real_t>
bool ApproxEqualVector(Vector3D<Real_t> const &v1, Vector3D<Real_t> const &v2)
{
  return ApproxEqual(v1[0], v2[0]) && ApproxEqual(v1[1], v2[1]) && ApproxEqual(v1[2], v2[2]);
}

bool ApproxEqualTransformation(Transformation const &t1, Transformation const &t2)
{
  if (!ApproxEqualVector(t1.Translation(), t2.Translation())) return false;
  for (int i = 0; i < 9; ++i)
    if (!ApproxEqual(t1.Rotation(i), t2.Rotation(i))) return false;
  return true;
}

// Placeholder (on host) for all surfaces belonging to a volume. An array of those will be indexed
// by the logical volume id. Also an intermediate helper for building portals.
// Note: the local surfaces defined by solids will have local references that will be changed by
// the flattening process, depending on the scene on which the parent volume will be flattened
struct VolumeShell {
  std::vector<int> fSurfaces; ///< Local surface id's for this volume
};

// We would need masks, imprints, portals created contiguously, by some factory
// Host helper for filling the SurfData store
template <typename Real_t>
class BrepHelper {
  using SurfData_t   = SurfData<Real_t>;
  using CylData_t    = CylData<Real_t>;
  using ConeData_t   = ConeData<Real_t>;
  using SphData_t    = SphData<Real_t>;
  using WindowMask_t = WindowMask<Real_t>;
  using RingMask_t   = RingMask<Real_t>;
  using ZPhiMask_t   = ZPhiMask<Real_t>;
  //  using Vector         = vecgeom::Vector3D<Real_t>;

private:
  int fVerbose{0};                   ///< verbosity level
  SurfData_t *fSurfData{nullptr};    ///< Surface data
  SurfData_t *fSurfDataGPU{nullptr}; ///< Surface data on device

  std::vector<WindowMask_t> fWindowMasks;     ///< rectangular masks
  std::vector<RingMask_t> fRingMasks;         ///< ring masks
  std::vector<ZPhiMask_t> fZPhiMasks;         ///< cylindrical masks
  std::vector<CylData_t> fCylSphData;         ///< data for cyl surfaces
  std::vector<ConeData_t> fConeData;          ///< data for conical surfaces
  std::vector<Transformation> fLocalTrans;    ///< local transformations
  std::vector<Transformation> fGlobalTrans;   ///< global transformations for surfaces in the scene
  std::vector<FramedSurface> fLocalSurfaces;  ///< local surfaces
  std::vector<FramedSurface> fFramedSurf;     ///< global surfaces
  std::vector<CommonSurface> fCommonSurfaces; ///< common surfaces
  std::vector<VolumeShell> fShells;           ///< vector of local volume surfaces
  std::vector<std::vector<int>> fCandidates;  ///< candidate lists for each state
  std::vector<std::vector<int>> fFrameInd;    ///< frame index per candidate

  BrepHelper() : fSurfData(new SurfData_t()) {}

public:
  /// Returns the singleton instance (CPU only)
  static BrepHelper &Instance()
  {
    static BrepHelper instance;
    return instance;
  }

  SurfData_t const &GetSurfData() const { return *fSurfData; }

  void ClearData()
  {
    // Dispose of surface data and shrink the container
    fWindowMasks.clear();
    std::vector<WindowMask_t>().swap(fWindowMasks);
    fRingMasks.clear();
    std::vector<RingMask_t>().swap(fRingMasks);
    fZPhiMasks.clear();
    std::vector<ZPhiMask_t>().swap(fZPhiMasks);
    fCylSphData.clear();
    std::vector<CylData_t>().swap(fCylSphData);
    fConeData.clear();
    std::vector<ConeData_t>().swap(fConeData);
    fLocalTrans.clear();
    std::vector<Transformation>().swap(fLocalTrans);
    fGlobalTrans.clear();
    std::vector<Transformation>().swap(fGlobalTrans);
    fLocalSurfaces.clear();
    std::vector<FramedSurface>().swap(fLocalSurfaces);
    fFramedSurf.clear();
    std::vector<FramedSurface>().swap(fFramedSurf);
    fCommonSurfaces.clear();
    std::vector<CommonSurface>().swap(fCommonSurfaces);
    for (size_t i = 0; i < fShells.size(); ++i) {
      fShells[i].fSurfaces.clear();
      std::vector<int>().swap(fShells[i].fSurfaces);
    }
    fShells.clear();
    std::vector<VolumeShell>().swap(fShells);

    delete fSurfData;
    fSurfData = nullptr;
    // cudaDelete(fSurfDataGPU);
    fSurfDataGPU = nullptr;
  }

  ~BrepHelper() { delete fSurfData; }

  void SetVerbosity(int verbose) { fVerbose = verbose; }

  void SortSides(int common_id)
  {
    // lambda to remove a surface from a side
    auto removeSurface = [&](Side &side, int ind) {
      for (int i = ind + 1; i < side.fNsurf; ++i)
        side.fSurfaces[i - 1] = side.fSurfaces[i];
      if (ind < side.fNsurf) side.fNsurf--;
    };

    // lambda to detect surfaces on a side that have identical frame
    auto sortAndRemoveCommonFrames = [&](Side &side) {
      if (!side.fNsurf) return;
      std::sort(side.fSurfaces, side.fSurfaces + side.fNsurf,
                [&](int i, int j) { return fFramedSurf[i] < fFramedSurf[j]; });
      for (int i = 0; i < side.fNsurf - 1; ++i) {
        for (int j = side.fNsurf - 1; j > i; --j) {
          if (EqualFrames(side, i, j)) removeSurface(side, j);
        }
      }
    };

    // lambda to find if the surfaces on one side have a common parent
    auto findParentFramedSurf = [&](Side &side) {
      if (!side.fNsurf) return;
      // if there is a parent, it can only be at the last position after sorting
      int parent_ind     = side.fNsurf - 1;
      auto parent_navind = fFramedSurf[side.fSurfaces[parent_ind]].fState;
      for (int i = 0; i < parent_ind; ++i) {
        auto navind = fFramedSurf[side.fSurfaces[i]].fState;
        if (!vecgeom::NavStateIndex::IsDescendentImpl(navind, parent_navind)) {
          parent_ind = -1;
          break;
        }
      }
      side.fParentSurf = parent_ind;
    };

    sortAndRemoveCommonFrames(fCommonSurfaces[common_id].fLeftSide);
    sortAndRemoveCommonFrames(fCommonSurfaces[common_id].fRightSide);
    findParentFramedSurf(fCommonSurfaces[common_id].fLeftSide);
    findParentFramedSurf(fCommonSurfaces[common_id].fRightSide);
  }

  void ComputeDefaultStates(int common_id)
  {
    using vecgeom::NavStateIndex;
    // Computes the default states for each side of a common surface
    Side &left  = fCommonSurfaces[common_id].fLeftSide;
    Side &right = fCommonSurfaces[common_id].fRightSide;
    assert(left.fNsurf > 0 || right.fNsurf > 0);

    NavIndex_t default_ind = 0;

    // A lambda that finds the deepest common ancestor between 2 states
    auto getCommonState = [&](NavIndex_t const &s1, NavIndex_t const &s2) {
      NavIndex_t a1(s1), a2(s2);
      // Bring both states at the same level
      while (NavStateIndex::GetLevelImpl(a1) > NavStateIndex::GetLevelImpl(a2))
        a1 = NavStateIndex::PopImpl(a1);
      while (NavStateIndex::GetLevelImpl(a2) > NavStateIndex::GetLevelImpl(a1))
        a2 = NavStateIndex::PopImpl(a2);

      // Pop until we reach the same state
      while (a1 != a2) {
        a1 = NavStateIndex::PopImpl(a1);
        a2 = NavStateIndex::PopImpl(a2);
      }
      return a1;
    };

    int minlevel = 10000; // this is a big-enough number as level
    for (int isurf = 0; isurf < left.fNsurf; ++isurf) {
      auto navind = fFramedSurf[left.fSurfaces[isurf]].fState;
      minlevel    = std::min(minlevel, (int)NavStateIndex::GetLevelImpl(navind));
    }
    for (int isurf = 0; isurf < right.fNsurf; ++isurf) {
      auto navind = fFramedSurf[right.fSurfaces[isurf]].fState;
      minlevel    = std::min(minlevel, (int)NavStateIndex::GetLevelImpl(navind));
    }

    // initialize the default state
    if (left.fNsurf > 0)
      default_ind = fFramedSurf[left.fSurfaces[0]].fState;
    else if (right.fNsurf > 0)
      default_ind = fFramedSurf[right.fSurfaces[0]].fState;

    for (int isurf = 0; isurf < left.fNsurf; ++isurf)
      default_ind = getCommonState(default_ind, fFramedSurf[left.fSurfaces[isurf]].fState);
    for (int isurf = 0; isurf < right.fNsurf; ++isurf)
      default_ind = getCommonState(default_ind, fFramedSurf[right.fSurfaces[isurf]].fState);

    if (NavStateIndex::GetLevelImpl(default_ind) == minlevel) default_ind = NavStateIndex::PopImpl(default_ind);
    fCommonSurfaces[common_id].fDefaultState = default_ind;
  }

  // Computes the bounding extent on a planar side.
  void ComputePlaneExtent(Side &side)
  {
    // This is a helper-lambda that updates extents
    // for all sides of common plane surfaces
    auto updatePlaneExtent = [](WindowMask_t &e, Vector3D<Real_t> const &pt) {
      e.rangeU[0] = std::min(e.rangeU[0], pt[0]);
      e.rangeU[1] = std::max(e.rangeU[1], pt[0]);
      e.rangeV[0] = std::min(e.rangeV[0], pt[1]);
      e.rangeV[1] = std::max(e.rangeV[1], pt[1]);
    };

    // Setting initial mask for an extent.
    constexpr Real_t kBig = 1.e30;
    WindowMask_t ext{kBig, -kBig, kBig, -kBig};

    // loop through all extents on a side:
    for (int i = 0; i < side.fNsurf; ++i) {
      // convert surface frame to local coordinates
      auto framed_surf     = fSurfData->fFramedSurf[side.fSurfaces[i]];
      FrameType frame_type = framed_surf.fFrame.type;
      Vector3D<Real_t> local;
      Real_t xmax, ymax, ymin, xmin;
      // Calculating the limits
      switch (frame_type) {
      case kWindow: {
        WindowMask_t extLocal;
        framed_surf.fFrame.GetMask(extLocal, *fSurfData);
        xmin = extLocal.rangeU[0];
        xmax = extLocal.rangeU[1];
        ymin = extLocal.rangeV[0];
        ymax = extLocal.rangeV[1];
        break;
      }
      case kRing: {
        // There is probably a more clever way to do this.
        RingMask_t extLocal;
        framed_surf.fFrame.GetMask(extLocal, *fSurfData);

        auto Rmax = extLocal.rangeR[1];
        // The axis vector has to be between Rmin and Rmax and cannot be unit vector anymore
        auto Rmean = (extLocal.rangeR[0] + Rmax) * 0.5;
        Vector3D<Real_t> axis{Rmean, 0, 0};
        if (!extLocal.isFullCirc) {
          // Projections of points that delimit vertices of the phi-cut ring
          Real_t x1, x2, x3, x4, y1, y2, y3, y4;

          auto Rmin = extLocal.rangeR[0];
          Vector3D<Real_t> vecSPhi{extLocal.vecSPhi[0], extLocal.vecSPhi[1], 0};
          Vector3D<Real_t> vecEPhi{extLocal.vecEPhi[0], extLocal.vecEPhi[1], 0};

          x1 = Rmax * axis.Dot(vecSPhi); //< (sphi, Rmax)_x
          x2 = Rmax * axis.Dot(vecEPhi); //< (ephi, Rmax)_x
          x3 = Rmin * axis.Dot(vecSPhi); //< (sphi, Rmin)_x
          x4 = Rmin * axis.Dot(vecEPhi); //< (ephi, Rmin)_x
          axis.Set(0, Rmean, 0);
          y1 = Rmax * axis.Dot(vecSPhi); //< (sphi, Rmax)_x
          y2 = Rmax * axis.Dot(vecEPhi); //< (ephi, Rmax)_x
          y3 = Rmin * axis.Dot(vecSPhi); //< (sphi, Rmin)_x
          y4 = Rmin * axis.Dot(vecEPhi); //< (ephi, Rmin)_x

          xmax = vecgeom::Max(vecgeom::Max(x1, x2), vecgeom::Max(x3, x4));
          ymax = vecgeom::Max(vecgeom::Max(y1, y2), vecgeom::Max(y3, y4));
          xmin = vecgeom::Min(vecgeom::Min(x1, x2), vecgeom::Min(x3, x4));
          ymin = vecgeom::Min(vecgeom::Min(y1, y2), vecgeom::Min(y3, y4));
        }
        // If the axes lie within the circle
        axis.Set(Rmean, 0, 0);
        if (ext.Inside(axis)) xmax = Rmax;
        axis.Set(0, Rmean, 0);
        if (ext.Inside(axis)) ymax = Rmax;
        axis.Set(-Rmean, 0, 0);
        if (ext.Inside(axis)) xmin = -Rmax;
        axis.Set(0, -Rmean, 0);
        if (ext.Inside(axis)) ymin = -Rmax;
      }
      default:
        break;
      } // case

      // This part updates extent
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(Vector3D<Real_t>{xmin, ymin, 0});
      updatePlaneExtent(ext, local);
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(Vector3D<Real_t>{xmin, ymax, 0});
      updatePlaneExtent(ext, local);
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(Vector3D<Real_t>{xmax, ymax, 0});
      updatePlaneExtent(ext, local);
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(Vector3D<Real_t>{xmax, ymin, 0});
      updatePlaneExtent(ext, local);
    } // for

    // Add new extent mask to vector
    int id = fWindowMasks.size();
    fWindowMasks.push_back(ext);
    side.fExtent.id = id;
  }

  // Computes bounding extent on a side of cylindrical surface
  void ComputeCylinderExtent(Side &side)
  {
    // Setting initial extent mask
    constexpr Real_t kBig = 1.e30;
    ZPhiMask_t sideext{kBig, -kBig, false};
    side.fExtent.type = kZPhi;

    bool first = true;
    for (int i = 0; i < side.fNsurf; ++i) {
      // convert surface frame to local coordinates
      auto framed_surf = fSurfData->fFramedSurf[side.fSurfaces[i]];
      ZPhiMask_t extLocal;
      framed_surf.fFrame.GetMask(extLocal, *fSurfData);
      Vector3D<Real_t> local;

      // The z-axis is shared and all surfaces are on the same side, so
      // there is no flipping.
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(Vector3D<Real_t>{0, 0, extLocal.rangeZ[0]});
      sideext.rangeZ[0] = std::min(sideext.rangeZ[0], local[2]);
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(Vector3D<Real_t>{0, 0, extLocal.rangeZ[1]});
      sideext.rangeZ[1] = std::max(sideext.rangeZ[1], local[2]);

      // If we already had a mask that is full circle
      if (sideext.isFullCirc) continue;

      // If current frame is wider than extent:
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(
          Vector3D<Real_t>{extLocal.vecSPhi[0], extLocal.vecSPhi[1], 0});
      if (!sideext.Inside(local)) sideext.vecSPhi.Set(extLocal.vecSPhi[0], extLocal.vecSPhi[1]);
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(
          Vector3D<Real_t>{extLocal.vecEPhi[0], extLocal.vecEPhi[1], 0});
      if (!sideext.Inside(local)) sideext.vecEPhi.Set(extLocal.vecEPhi[0], extLocal.vecEPhi[1]);

      if (first) {
        sideext.vecSPhi.Set(extLocal.vecSPhi[0], extLocal.vecSPhi[1]);
        sideext.vecEPhi.Set(extLocal.vecEPhi[0], extLocal.vecEPhi[1]);
        first = !first;
      }
    } // for

    // Add new extent mask to the vector
    int id = fZPhiMasks.size();
    fZPhiMasks.push_back(sideext);
    side.fExtent.id = id;
  }

  void ComputeExtents()
  {
    // Lambda for computing the extent of a single side
    auto computeSingleSideExtent = [&](SurfaceType type, Side &side) {
      switch (type) {
      case kPlanar:
        ComputePlaneExtent(side);
        break;
      case kCylindrical:
        ComputeCylinderExtent(side);
        break;
      default:
        std::cout << "Computing side extents dropped to default." << std::endl;
        break;
      }
    };

    // Compute extents for all sides on all surfaces
    for (int common_id = 1; common_id < fSurfData->fNcommonSurf; ++common_id) {
      if (fSurfData->fCommonSurfaces[common_id].fLeftSide.fNsurf) {
        computeSingleSideExtent(fSurfData->fCommonSurfaces[common_id].fType,
                                fSurfData->fCommonSurfaces[common_id].fLeftSide);
      }
      if (fSurfData->fCommonSurfaces[common_id].fRightSide.fNsurf) {
        computeSingleSideExtent(fSurfData->fCommonSurfaces[common_id].fType,
                                fSurfData->fCommonSurfaces[common_id].fRightSide);
      }
    }

    // We created new masks, update them.
    UpdateMaskData();
  }

  // Printing is ugly currently and scales badly with the new data structure.
  // Perhaps each mask should have its own print() method that returns a string.
  void PrintCommonSurface(int common_id)
  {
    auto const &surf = fSurfData->fCommonSurfaces[common_id];
    printf("== common surface %d: type: %d, default state: ", common_id, surf.fType);
    vecgeom::NavStateIndex default_state(surf.fDefaultState);
    default_state.Print();
    printf(" transformation %d: ", surf.fTrans);
    fSurfData->fGlobalTrans[surf.fTrans].Print();
    switch (surf.fType) {
    case kPlanar: {
      WindowMask_t const &extL = fSurfData->fWindowMasks[surf.fLeftSide.fExtent.id];
      printf("\n   left: %d surfaces, parent=%d, extent %d: {{%g, %g}, {%g, %g}}\n", surf.fLeftSide.fNsurf,
             surf.fLeftSide.fParentSurf, surf.fLeftSide.fExtent.id, extL.rangeU[0], extL.rangeU[1], extL.rangeV[0],
             extL.rangeV[1]);
      break;
    }
    case kCylindrical: {
      ZPhiMask_t const &extL = fSurfData->fZPhiMasks[surf.fLeftSide.fExtent.id];
      printf("\n   left: %d surfaces, parent=%d, extent %d: {{%g, %g}, {%g, %g}, {%g, %g}}\n", surf.fRightSide.fNsurf,
             surf.fRightSide.fParentSurf, surf.fRightSide.fExtent.id, extL.rangeZ[0], extL.rangeZ[1], extL.vecSPhi[0],
             extL.vecSPhi[1], extL.vecEPhi[0], extL.vecEPhi[1]);
      break;
    }
    case kConical:
    case kSpherical:
    case kTorus:
    case kGenSecondOrder:
    default:
      std::cout << "Case not implemented. " << std::endl;
    }
    for (int i = 0; i < surf.fLeftSide.fNsurf; ++i) {
      int idglob         = surf.fLeftSide.fSurfaces[i];
      auto const &placed = fSurfData->fFramedSurf[idglob];
      printf("    surf %d: trans: ", idglob);
      fSurfData->fGlobalTrans[placed.fTrans].Print();
      printf(", ");
      vecgeom::NavStateIndex state(placed.fState);
      state.Print();
    }
    if (surf.fRightSide.fNsurf > 0) switch (surf.fType) {
      case kPlanar: {
        WindowMask_t const &extR = fSurfData->fWindowMasks[surf.fRightSide.fExtent.id];
        printf("\n   left: %d surfaces, parent=%d, extent %d: {{%g, %g}, {%g, %g}}\n", surf.fRightSide.fNsurf,
               surf.fRightSide.fParentSurf, surf.fRightSide.fExtent.id, extR.rangeU[0], extR.rangeU[1], extR.rangeV[0],
               extR.rangeV[1]);
        break;
      }
      case kCylindrical: {
        ZPhiMask_t const &extR = fSurfData->fZPhiMasks[surf.fRightSide.fExtent.id];
        printf("\n   left: %d surfaces, parent=%d, extent %d: {{%g, %g}, {%g, %g}, {%g, %g}}\n", surf.fRightSide.fNsurf,
               surf.fRightSide.fParentSurf, surf.fRightSide.fExtent.id, extR.rangeZ[0], extR.rangeZ[1], extR.vecSPhi[0],
               extR.vecSPhi[1], extR.vecEPhi[0], extR.vecEPhi[1]);
        break;
      }
      case kConical:
      case kSpherical:
      case kTorus:
      case kGenSecondOrder:
      default:
        std::cout << "Case not implemented. " << std::endl;
      }
    else
      printf("   right: 0 surfaces\n");

    for (int i = 0; i < surf.fRightSide.fNsurf; ++i) {
      int idglob         = surf.fRightSide.fSurfaces[i];
      auto const &placed = fSurfData->fFramedSurf[idglob];
      printf("    surf %d: trans: ", idglob);
      fSurfData->fGlobalTrans[placed.fTrans].Print();
      vecgeom::NavStateIndex state(placed.fState);
      state.Print();
    }
  }

  void SetNvolumes(int nvolumes)
  {
    if (fShells.size() > 0) {
      std::cout << "BrepHelper::SetNvolumes already called for this instance.\n";
      return;
    }
    fShells.resize(nvolumes);
  }

  bool CreateLocalSurfaces()
  {
    // Iterate logical volumes and create local surfaces
    std::vector<vecgeom::LogicalVolume *> volumes;
    vecgeom::GeoManager::Instance().GetAllLogicalVolumes(volumes);
    SetNvolumes(volumes.size());
    // TODO: Implement a VUnplacedVolume::CreateSurfaces interface for surface creation
    // create a placeholder for surface data
    for (auto volume : volumes) {
      vecgeom::VUnplacedVolume const *solid = volume->GetUnplacedVolume();
      vecgeom::UnplacedBox const *box       = dynamic_cast<vecgeom::UnplacedBox const *>(solid);
      if (box) {
        CreateBoxSurfaces(*box, volume->id());
        continue;
      }
      vecgeom::UnplacedTube const *tube = dynamic_cast<vecgeom::UnplacedTube const *>(solid);
      if (tube) {
        CreateTubeSurfaces(*tube, volume->id());
        continue;
      }
      std::cout << "testEm3: solid type not supported for volume: " << volume->GetName() << "\n";
      return false;
    }
    return true;
  }

  bool CreateCommonSurfacesFlatTop()
  {
    // Iterate the geometry tree and flatten surfaces at top level
    int nphysical = 0;
    vecgeom::NavStateIndex state;

    // recursive geometry visitor lambda creating the common surfaces for the current placed volume
    typedef std::function<void(vecgeom::VPlacedVolume const *)> func_t;
    func_t createCommonSurfaces = [&](vecgeom::VPlacedVolume const *pvol) {
      state.Push(pvol);
      const auto vol = pvol->GetLogicalVolume();
      auto daughters = vol->GetDaughters();
      int nd         = daughters.size();
      nphysical++;
      Transformation trans;
      state.TopMatrix(trans);
      VolumeShell const &shell = fShells[vol->id()];
      for (int lsurf_id : shell.fSurfaces) {
        FramedSurface const &lsurf = fLocalSurfaces[lsurf_id];
        Transformation global(trans);
        global.MultiplyFromRight(fLocalTrans[lsurf.fTrans]);
        int trans_id = fGlobalTrans.size();
        fGlobalTrans.push_back(global);
        // Create the global surface
        int id_glob = fFramedSurf.size();
        fFramedSurf.push_back({lsurf.fSurface, lsurf.fFrame, trans_id, lsurf.fFlip, state.GetNavIndex()});
        CreateCommonSurface(id_glob);
      }

      // Now do the daughters
      for (int id = 0; id < nd; ++id) {
        createCommonSurfaces(daughters[id]);
      }
      state.Pop();
    };

    // add identity first in the list of global transformations
    Transformation identity;
    fGlobalTrans.push_back(identity);
    // add a dummy common surface since index 0 is not allowed for correctly handling sides
    fCommonSurfaces.push_back({});

    createCommonSurfaces(vecgeom::GeoManager::Instance().GetWorld());

    for (size_t isurf = 1; isurf < fCommonSurfaces.size(); ++isurf) {
      // Compute the default states in case no frame on the surface is hit
      ComputeDefaultStates(isurf);
      // Sort placed surfaces on sides by geometry depth (bigger depth comes first)
      SortSides(isurf);
      // Convert transformations of placed surfaces in the local frame of the common surface
      ConvertTransformations(isurf);
    }

    // Create the full surface candidate list for each navigation state
    CreateCandidateLists();

    // Now update the surface data structure used for navigation
    UpdateSurfData();

    // Compute extents for all sides of common surfaces
    ComputeExtents();

    if (fVerbose > 0) {
      for (size_t isurf = 1; isurf < fCommonSurfaces.size(); ++isurf)
        PrintCommonSurface(isurf);
    }

    if (fVerbose > 1) {
      PrintCandidateLists();
      std::cout << "Visited " << nphysical << " physical volumes, created " << fCommonSurfaces.size() - 1
                << " common surfaces\n";
    }

    return true;
  }

  ///< This method uses the transformation T1 of the first placed surface on the left side (which always exists)
  ///< as transformation for the common surface, then recalculates the transformations of all placed
  ///< surfaces as T' = T1.Inverse() * T. If identity this will get the index 0.
  void ConvertTransformations(int idsurf)
  {
    auto &surf = fCommonSurfaces[idsurf];
    // Adopt the transformation of the first surface on left for the common surface
    surf.fTrans = fFramedSurf[surf.fLeftSide.fSurfaces[0]].fTrans;
    // Set transformation of first surface on left to identity
    fFramedSurf[surf.fLeftSide.fSurfaces[0]].fTrans = 0;

    Transformation tsurfinv;
    fGlobalTrans[surf.fTrans].Inverse(tsurfinv);

    // Skip first surface on left side
    for (int i = 1; i < surf.fLeftSide.fNsurf; ++i) {
      int idglob = surf.fLeftSide.fSurfaces[i];
      auto &surf = fFramedSurf[idglob];
      Transformation tnew(tsurfinv);
      tnew.MultiplyFromRight(fGlobalTrans[surf.fTrans]);
      if (ApproxEqualTransformation(tnew, fGlobalTrans[0])) {
        surf.fTrans = 0;
      } else {
        fGlobalTrans[surf.fTrans] = tnew;
      }
    }

    // Convert right-side surfaces
    for (int i = 0; i < surf.fRightSide.fNsurf; ++i) {
      int idglob = surf.fRightSide.fSurfaces[i];
      auto &surf = fFramedSurf[idglob];
      Transformation tnew(tsurfinv);
      tnew.MultiplyFromRight(fGlobalTrans[surf.fTrans]);
      if (ApproxEqualTransformation(tnew, fGlobalTrans[0])) {
        surf.fTrans = 0;
      } else {
        fGlobalTrans[surf.fTrans] = tnew;
      }
    }
  }

  ///< This method creates helper lists of candidate surfaces for each navigation state
  void CreateCandidateLists()
  {
    int numNodes = vecgeom::GeoManager::Instance().GetTotalNodeCount() + 1; // count also outside state
    fCandidates.reserve(numNodes);
    fFrameInd.reserve(numNodes);

    // Lambda adding the surface id as candidate to all states from a side
    auto addSurfToSideStates = [&](int isurf, int iside) {
      Side const &side = (iside > 0) ? fCommonSurfaces[isurf].fLeftSide : fCommonSurfaces[isurf].fRightSide;
      for (int i = 0; i < side.fNsurf; ++i) {
        int idglob             = side.fSurfaces[i];
        auto const &framedsurf = fFramedSurf[idglob];
        vecgeom::NavStateIndex state(framedsurf.fState);
        int state_id = state.GetId();
        fCandidates[state_id].push_back(isurf * iside);
        fFrameInd[state_id].push_back(i);
      }
    };

    // prepare all lists
    for (int i = 0; i < numNodes; ++i) {
      fCandidates.push_back({});
      fFrameInd.push_back({});
    }

    // loop over all common surfaces and add their index in the appropriate list
    for (size_t isurf = 1; isurf < fCommonSurfaces.size(); ++isurf) {
      auto const &surf = fCommonSurfaces[isurf];
      // Add to default surface state
      vecgeom::NavStateIndex state(surf.fDefaultState);
      fCandidates[state.GetId()].push_back(-isurf);
      fFrameInd[state.GetId()].push_back(-1); // means this state is the default for isurf
      // Add to side states
      addSurfToSideStates(isurf, 1);
      addSurfToSideStates(isurf, -1);
    }
  }

  void PrintCandidateLists()
  {
    vecgeom::NavStateIndex state;

    // recursive geometry visitor lambda printing the candidates lists
    // We have no direct access from a state (contiguous) id to the actual state index
    typedef std::function<void(vecgeom::VPlacedVolume const *)> func_t;
    func_t printCandidates = [&](vecgeom::VPlacedVolume const *pvol) {
      state.Push(pvol);
      const auto vol = pvol->GetLogicalVolume();
      auto daughters = vol->GetDaughters();
      int nd         = daughters.size();
      state.Print();
      auto const &cand = fSurfData->fCandidates[state.GetId()];
      printf(" %d candidates: ", cand.fNcand);
      for (int i = 0; i < cand.fNcand; ++i)
        printf("%d (ind %d) ", cand.fCandidates[i], cand.fFrameInd[i]);
      printf("\n");

      // do daughters
      for (int id = 0; id < nd; ++id) {
        printCandidates(daughters[id]);
      }
      state.Pop();
    };

    printf("\nCandidate surfaces per state:");
    state.Print();
    auto const &cand = fSurfData->fCandidates[state.GetId()];
    printf(" %d candidates: ", cand.fNcand);
    for (int i = 0; i < cand.fNcand; ++i)
      printf("%d (ind %d) ", cand.fCandidates[i], cand.fFrameInd[i]);
    printf("\n");

    printCandidates(vecgeom::GeoManager::Instance().GetWorld());
  }

private:
  int AddSurfaceToShell(int logical_id, int isurf)
  {
    if (fShells.size() == 0) {
      std::cout << "BrepHelper::AddSurfaceToShell: need to call SetNvolumes first\n";
      return -1;
    }
    assert(logical_id < (int)fShells.size() && "surface shell id exceeding number of volumes");
    int id = fShells[logical_id].fSurfaces.size();
    fShells[logical_id].fSurfaces.push_back(isurf);
    return id;
  }

  UnplacedSurface CreateUnplacedSurface(SurfaceType type, Real_t *data = nullptr)
  {
    switch (type) {
    case kPlanar:
      return UnplacedSurface(type, -1);
    case kCylindrical:
    case kSpherical:
      fCylSphData.push_back({data[0]});
      return UnplacedSurface(type, fCylSphData.size() - 1);
    case kConical:
      fConeData.push_back({data[0], data[1]});
      return UnplacedSurface(type, fConeData.size() - 1);
    case kTorus:
    case kGenSecondOrder:
      std::cout << "kTorus, kGenSecondOrder unhandled\n";
      return UnplacedSurface(type, -1);
    };
    return UnplacedSurface(type, -1);
  }

  // There could be a more elegant solution, with a function that takes a
  // pointer to mask parameters and uses switch structure to select appropriate
  // constructor and mask, but this is OK for now.
  // Creators for different types of frames.
  Frame CreateFrame(FrameType type, WindowMask_t const &mask)
  {
    int id = fWindowMasks.size();
    fWindowMasks.push_back(mask);
    return Frame(type, id);
  }

  Frame CreateFrame(FrameType type, RingMask_t const &mask)
  {
    int id = fRingMasks.size();
    fRingMasks.push_back(mask);
    return Frame(type, id);
  }

  Frame CreateFrame(FrameType type, ZPhiMask_t const &mask)
  {
    int id = fZPhiMasks.size();
    fZPhiMasks.push_back(mask);
    return Frame(type, id);
  }

  int CreateLocalTransformation(Transformation const &trans)
  {
    int id = fLocalTrans.size();
    fLocalTrans.push_back(trans);
    return id;
  }

  int CreateLocalSurface(UnplacedSurface const &unplaced, Frame const &frame, int trans, int flip = 1)
  {
    int id = fLocalSurfaces.size();
    fLocalSurfaces.push_back({unplaced, frame, trans, flip});
    return id;
  }

  int CreateCommonSurface(int idglob)
  {
    bool flip;

    auto approxEqual = [&](int idglob1, int idglob2) {
      flip                    = false;
      FramedSurface const &s1 = fFramedSurf[idglob1];
      FramedSurface const &s2 = fFramedSurf[idglob2];
      // Surfaces may be in future "compatible" even if they are not the same, for now enforce equality
      if (s1.fSurface.type != s2.fSurface.type) return false;

      // Check if the 2 surfaces are parallel
      Transformation const &t1 = fGlobalTrans[s1.fTrans];
      Transformation const &t2 = fGlobalTrans[s2.fTrans];
      // Calculate normalized connection vector between the two transformations
      // Use double precision explicitly
      vecgeom::Vector3D<double> tdiff = t1.Translation() - t2.Translation();
      bool same_tr                    = ApproxEqualVector(tdiff, {0, 0, 0});
      vecgeom::Vector3D<double> ldir;
      switch (s1.fSurface.type) {
      case kPlanar:
        if (same_tr) break;
        // For planes to match, the connecting vector must be along the planes
        tdiff.Normalize();
        t1.TransformDirection(tdiff, ldir);
        if (std::abs(ldir[2]) > vecgeom::kTolerance) return false;
        break;
      case kCylindrical:
        if (std::abs(fCylSphData[s1.fSurface.id].radius - fCylSphData[s2.fSurface.id].radius) > vecgeom::kTolerance)
          return false;
        if (same_tr) break;
        tdiff.Normalize();
        t1.TransformDirection(tdiff, ldir);
        // For connected cylinders, the connecting vector must be along the Z axis
        if (!ApproxEqualVector(ldir, {0, 0, ldir[2]})) return false;
        break;
      case kConical:
      case kSpherical:
      case kTorus:
      case kGenSecondOrder:
      default:
        printf("CreateCommonSurface: case not implemented\n");
        return false;
      };

      // Now check if the rotations are matching. The z axis transformed
      // with the two rotations should end up in aligned vectors. This is
      // true for planes (Z is the normal) but also for tubes/cones where
      // Z is the axis of symmetry
      vecgeom::Vector3D<double> zaxis(0, 0, 1);
      auto z1 = t1.InverseTransformDirection(zaxis);
      auto z2 = t2.InverseTransformDirection(zaxis);
      if (!ApproxEqualVector(z1.Cross(z2), {0, 0, 0})) return false;

      flip = z1.Dot(z2) < 0;
      return true;
    };

    // this may be slow
    auto it = std::find_if(std::begin(fCommonSurfaces), std::end(fCommonSurfaces), [&](const CommonSurface &t) {
      return (t.fLeftSide.fNsurf > 0) ? approxEqual(t.fLeftSide.fSurfaces[0], idglob) : false;
    });
    int id  = -1;
    if (it != std::end(fCommonSurfaces)) {
      id = int(it - std::begin(fCommonSurfaces));
      // Add the global surface to the appropriate side
      if (flip)
        (*it).fRightSide.AddSurface(idglob);
      else
        (*it).fLeftSide.AddSurface(idglob);

    } else {
      // Construct a new common surface from the current placed global surface
      id = fCommonSurfaces.size();
      fCommonSurfaces.push_back({fFramedSurf[idglob].fSurface.type, idglob});
    }
    return id;
  }

  // The code for creating solid-specific surfaces should sit in the specific solid struct type
  void CreateBoxSurfaces(vecgeom::UnplacedBox const &box, int logical_id)
  {
    int isurf;
    // surface at -dx:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.y(), box.z()}),
                               CreateLocalTransformation({-box.x(), 0, 0, -90, 90, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // surface at +dx:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.y(), box.z()}),
                               CreateLocalTransformation({box.x(), 0, 0, 90, 90, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // surface at -dy:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.x(), box.z()}),
                               CreateLocalTransformation({0, -box.y(), 0, 0, 90, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // surface at +dy:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.x(), box.z()}),
                               CreateLocalTransformation({0, box.y(), 0, 0, -90, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // surface at -dz:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.x(), box.y()}),
                               CreateLocalTransformation({0, 0, -box.z(), 0, 180, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // surface at +dz:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.x(), box.y()}),
                               CreateLocalTransformation({0, 0, box.z(), 0, 0, 0}));
    AddSurfaceToShell(logical_id, isurf);
  }

  void CreateTubeSurfaces(vecgeom::UnplacedTube const &tube, int logical_id)
  {
    auto sphi = tube.sphi();
    auto dphi = tube.dphi();
    auto ephi = tube.sphi() + tube.dphi();

    assert(dphi > vecgeom::kTolerance);

    auto Rmean = (tube.rmin() + tube.rmax()) / 2;
    auto Rdiff = (tube.rmax() - tube.rmin()) / 2;

    assert(Rdiff > 0);

    bool fullCirc = ApproxEqual(dphi, vecgeom::kTwoPi);

    int isurf;

    // We need angles in degrees for transformations
    auto sphid = vecgeom::kRadToDeg * sphi;
    auto ephid = vecgeom::kRadToDeg * ephi;

    // surface at +dz
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar),
                               CreateFrame(kRing, RingMask_t{tube.rmin(), tube.rmax(), fullCirc, sphi, ephi}),
                               CreateLocalTransformation({0, 0, tube.z(), 0, 0, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // surface at -dz
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar),
                               CreateFrame(kRing, RingMask_t{tube.rmin(), tube.rmax(), fullCirc, sphi, ephi}),
                               CreateLocalTransformation({0, 0, -tube.z(), 0, 180, -sphid - ephid}));
    AddSurfaceToShell(logical_id, isurf);
    // inner cylinder
    if (tube.rmin() > vecgeom::kTolerance) {
      Real_t *rmin_ptr = new Real_t(tube.rmin());
      isurf            = CreateLocalSurface(CreateUnplacedSurface(kCylindrical, rmin_ptr),
                                            CreateFrame(kZPhi, ZPhiMask_t{-tube.z(), tube.z(), fullCirc, sphi, ephi}),
                                            CreateLocalTransformation({0, 0, 0, 0, 0, 0}), -1);
      AddSurfaceToShell(logical_id, isurf);
    }
    // outer cylinder
    Real_t *rmax_ptr = new Real_t(tube.rmax());
    isurf            = CreateLocalSurface(CreateUnplacedSurface(kCylindrical, rmax_ptr),
                                          CreateFrame(kZPhi, ZPhiMask_t{-tube.z(), tube.z(), fullCirc, sphi, ephi}),
                                          CreateLocalTransformation({0, 0, 0, 0, 0, 0}));
    AddSurfaceToShell(logical_id, isurf);

    if (ApproxEqual(dphi, vecgeom::kTwoPi)) return;
    // plane cap at Sphi
    isurf = CreateLocalSurface(
        CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{Rdiff, tube.z()}),
        CreateLocalTransformation({Rmean * std::cos(sphi), Rmean * std::sin(sphi), 0, sphid, 90, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // plane cap at Sphi+Dphi
    isurf = CreateLocalSurface(
        CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{Rdiff, tube.z()}),
        CreateLocalTransformation({Rmean * std::cos(ephi), Rmean * std::sin(ephi), 0, ephid, -90, 0}));
    AddSurfaceToShell(logical_id, isurf);
  }

  ///< This function evaluates if the frames of two placed surfaces on the same side
  ///< of a common surface are matching
  bool EqualFrames(Side const &side, int i1, int i2)
  {
    using Vector3D = vecgeom::Vector3D<Real_t>;

    FramedSurface const &s1 = fFramedSurf[side.fSurfaces[i1]];
    FramedSurface const &s2 = fFramedSurf[side.fSurfaces[i2]];
    if (s1.fFrame.type != s2.fFrame.type) return false;
    // Get displacement vector between the 2 frame centers and check if it has null length
    Transformation const &t1 = fGlobalTrans[s1.fTrans];
    Transformation const &t2 = fGlobalTrans[s2.fTrans];
    Vector3D tdiff           = t1.Translation() - t2.Translation();
    if (!ApproxEqualVector(tdiff, {0, 0, 0})) return false;

    // Different treatment of different frame types
    switch (s1.fFrame.type) {
    case kRangeZ:
      break;
    case kRing: {
      auto mask1 = fRingMasks[s1.fFrame.id];
      auto mask2 = fRingMasks[s2.fFrame.id];

      // Inner radius must be the same
      if (!ApproxEqual(mask1.rangeR[0], mask2.rangeR[0]) || !ApproxEqual(mask1.rangeR[1], mask2.rangeR[1]) ||
          mask1.isFullCirc != mask2.isFullCirc)
        return false;
      if (mask1.isFullCirc) return true;
      // Unit-vectors used for checking sphi
      auto phimin1 = t1.InverseTransformDirection(Vector3D{mask1.vecSPhi[0], mask1.vecSPhi[1], 0});
      auto phimin2 = t2.InverseTransformDirection(Vector3D{mask2.vecSPhi[0], mask2.vecSPhi[1], 0});
      // Rmax vectors used for checking dphi and outer radius
      auto phimax1 = t1.InverseTransformDirection(Vector3D{mask1.vecEPhi[0], mask1.vecEPhi[1], 0});
      auto phimax2 = t2.InverseTransformDirection(Vector3D{mask2.vecEPhi[0], mask2.vecEPhi[1], 0});

      if (ApproxEqualVector(phimin1, phimin2) && ApproxEqualVector(phimax1, phimax2)) return true;
      break;
    }
    case kZPhi: {
      auto mask1 = fZPhiMasks[s1.fFrame.id];
      auto mask2 = fZPhiMasks[s2.fFrame.id];

      // They are on the same side, so there is no flipping,
      // and z extents must be equal
      if (!ApproxEqual(mask1.rangeZ[0], mask2.rangeZ[0]) || !ApproxEqual(mask1.rangeZ[1], mask2.rangeZ[1]))
        return false;
      if (mask1.isFullCirc) return true;
      // Unit-vectors used for checking sphi
      auto phimin1 = t1.InverseTransformDirection(Vector3D{mask1.vecSPhi[0], mask1.vecSPhi[1], 0});
      auto phimin2 = t2.InverseTransformDirection(Vector3D{mask2.vecSPhi[0], mask2.vecSPhi[1], 0});
      // Rmax vectors used for checking dphi and outer radius
      auto phimax1 = t1.InverseTransformDirection(Vector3D{mask1.vecEPhi[0], mask1.vecEPhi[1], 0});
      auto phimax2 = t2.InverseTransformDirection(Vector3D{mask2.vecEPhi[0], mask2.vecEPhi[1], 0});

      if (ApproxEqualVector(phimin1, phimin2) && ApproxEqualVector(phimax1, phimax2)) return true;
      break;
    }
    case kRangeSph:
      // if (ApproxEqualVector(v1, v2)) return true; //Must be changed.
      break;
    case kWindow: {
      auto frameData1 = fWindowMasks[s1.fFrame.id];
      auto frameData2 = fWindowMasks[s2.fFrame.id];

      // Vertices
      Vector3D v11 =
          t1.InverseTransformDirection(Vector3D{frameData1.rangeU[0], frameData1.rangeV[0], 0}); // 1 down left
      Vector3D v12 =
          t1.InverseTransformDirection(Vector3D{frameData1.rangeU[1], frameData1.rangeV[1], 0}); // 1 up right
      Vector3D v21 =
          t2.InverseTransformDirection(Vector3D{frameData2.rangeU[0], frameData2.rangeV[0], 0}); // 2 down left
      Vector3D v22 =
          t2.InverseTransformDirection(Vector3D{frameData2.rangeU[1], frameData2.rangeV[1], 0}); // 2 up right

      // Diagonals
      auto diag1 = v12 - v11;
      auto diag2 = v22 - v21;

      return ApproxEqualVector(diag1.Abs(), diag2.Abs());
    }
    case kTriangle:
      // to be implemented
      break;
    };
    return false;
  }

  // A function to update all mask containers. Needs to be called
  // when updating masks after creating both frames and extents.
  void UpdateMaskData()
  {
    fSurfData->fNwindows    = fWindowMasks.size();
    fSurfData->fWindowMasks = new WindowMask_t[fWindowMasks.size()];
    for (size_t i = 0; i < fWindowMasks.size(); ++i)
      fSurfData->fWindowMasks[i] = fWindowMasks[i];

    fSurfData->fNrings    = fRingMasks.size();
    fSurfData->fRingMasks = new RingMask_t[fRingMasks.size()];
    for (size_t i = 0; i < fRingMasks.size(); ++i)
      fSurfData->fRingMasks[i] = fRingMasks[i];

    fSurfData->fNzphis    = fZPhiMasks.size();
    fSurfData->fZPhiMasks = new ZPhiMask_t[fZPhiMasks.size()];
    for (size_t i = 0; i < fZPhiMasks.size(); ++i)
      fSurfData->fZPhiMasks[i] = fZPhiMasks[i];
  }

  ///< The method updates the SurfData storage
  void UpdateSurfData()
  {
    // Create and copy surface data
    fSurfData->fNcylsph    = fCylSphData.size();
    fSurfData->fCylSphData = new CylData_t[fCylSphData.size()];
    for (size_t i = 0; i < fCylSphData.size(); ++i)
      fSurfData->fCylSphData[i] = fCylSphData[i];

    fSurfData->fNcone    = fConeData.size();
    fSurfData->fConeData = new ConeData_t[fConeData.size()];
    for (size_t i = 0; i < fConeData.size(); ++i)
      fSurfData->fConeData[i] = fConeData[i];

    // Create transformations
    fSurfData->fGlobalTrans = new Transformation[fGlobalTrans.size()];
    for (size_t i = 0; i < fGlobalTrans.size(); ++i)
      fSurfData->fGlobalTrans[i] = fGlobalTrans[i];

    // Create placed surfaces
    fSurfData->fNglobalSurf = fFramedSurf.size();
    fSurfData->fFramedSurf  = new FramedSurface[fFramedSurf.size()];
    for (size_t i = 0; i < fFramedSurf.size(); ++i)
      fSurfData->fFramedSurf[i] = fFramedSurf[i];

    // Create common surfaces
    size_t size_sides = 0;
    for (auto const &surf : fCommonSurfaces)
      size_sides += surf.fLeftSide.fNsurf + surf.fRightSide.fNsurf;

    // Create Masks
    UpdateMaskData();

    fSurfData->fSides          = new int[size_sides];
    int *current_side          = fSurfData->fSides;
    fSurfData->fNcommonSurf    = fCommonSurfaces.size();
    fSurfData->fCommonSurfaces = new CommonSurface[fCommonSurfaces.size()];
    for (size_t i = 0; i < fCommonSurfaces.size(); ++i) {
      // Raw copy of surface (wrong pointers in sides)
      fSurfData->fCommonSurfaces[i] = fCommonSurfaces[i];
      // Copy left sides content in buffer
      for (auto isurf = 0; isurf < fCommonSurfaces[i].fLeftSide.fNsurf; ++isurf)
        current_side[isurf] = fCommonSurfaces[i].fLeftSide.fSurfaces[isurf];
      // Make left sides arrays point to the buffer
      fSurfData->fCommonSurfaces[i].fLeftSide.fSurfaces = current_side;
      current_side += fCommonSurfaces[i].fLeftSide.fNsurf;

      // Copy right sides content in buffer
      for (auto isurf = 0; isurf < fCommonSurfaces[i].fRightSide.fNsurf; ++isurf)
        current_side[isurf] = fCommonSurfaces[i].fRightSide.fSurfaces[isurf];
      // Make right sides arrays point to the buffer
      fSurfData->fCommonSurfaces[i].fRightSide.fSurfaces = current_side;
      current_side += fCommonSurfaces[i].fRightSide.fNsurf;
      // Copy parent surface indices
      fSurfData->fCommonSurfaces[i].fLeftSide.fParentSurf  = fCommonSurfaces[i].fLeftSide.fParentSurf;
      fSurfData->fCommonSurfaces[i].fRightSide.fParentSurf = fCommonSurfaces[i].fRightSide.fParentSurf;
    }

    // Create candidates lists
    size_t size_candidates = 0;
    for (auto const &list : fCandidates)
      size_candidates += list.size();

    fSurfData->fCandList    = new int[2 * size_candidates];
    int *current_candidates = fSurfData->fCandList;
    fSurfData->fCandidates  = new Candidates[fCandidates.size()];
    for (size_t i = 0; i < fCandidates.size(); ++i) {
      auto ncand                       = fCandidates[i].size();
      fSurfData->fCandidates[i].fNcand = ncand;
      for (size_t icand = 0; icand < ncand; icand++) {
        current_candidates[icand]         = (fCandidates[i])[icand];
        current_candidates[ncand + icand] = (fFrameInd[i])[icand];
      }
      fSurfData->fCandidates[i].fCandidates = current_candidates;
      current_candidates += fCandidates[i].size();
      fSurfData->fCandidates[i].fFrameInd = current_candidates;
      current_candidates += fCandidates[i].size();
    }
  }
};

} // namespace vgbrep
#endif
