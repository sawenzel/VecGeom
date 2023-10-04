#ifndef VECGEOM_SURFACE_BREPHELPER_H_
#define VECGEOM_SURFACE_BREPHELPER_H_

#include <cassert>
#include <functional>
#include <map>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/base/CpuTypes.h>
#include <VecGeom/surfaces/conv/SolidConverter.h>

// Check if math necessary
#include <VecGeom/base/Math.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/Box.h>
#include <VecGeom/volumes/Trd.h>
#include <VecGeom/volumes/Tube.h>
#include <VecGeom/volumes/BooleanVolume.h>
#include <VecGeom/management/GeoManager.h>

namespace vgbrep {

template <typename Real_t>
class BrepHelper {
  using SurfData_t     = SurfData<Real_t>;
  using CylData_t      = CylData<Real_t>;
  using ConeData_t     = ConeData<Real_t>;
  using SphData_t      = SphData<Real_t>;
  using WindowMask_t   = WindowMask<Real_t>;
  using RingMask_t     = RingMask<Real_t>;
  using ZPhiMask_t     = ZPhiMask<Real_t>;
  using TriangleMask_t = TriangleMask<Real_t>;
  using QuadMask_t     = QuadrilateralMask<Real_t>;
  using CPUsurfData_t  = CPUsurfData<Real_t>;

private:
  int fVerbose{0};                ///< verbosity level
  SurfData_t *fSurfData{nullptr}; ///< Surface data
  CPUsurfData_t &fCPUdata;        ///< Transient CPU surface data used during conversion

  BrepHelper() : fSurfData(&SurfData_t::Instance()), fCPUdata(CPUsurfData_t::Instance()) {}

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
    fCPUdata.Clear();

    delete[] fSurfData->fWindowMasks;
    fSurfData->fWindowMasks = nullptr;
    delete[] fSurfData->fRingMasks;
    fSurfData->fRingMasks = nullptr;
    delete[] fSurfData->fZPhiMasks;
    fSurfData->fZPhiMasks = nullptr;
    delete[] fSurfData->fQuadMasks;
    fSurfData->fQuadMasks = nullptr;
    delete[] fSurfData->fCylSphData;
    fSurfData->fCylSphData = nullptr;
    delete[] fSurfData->fConeData;
    fSurfData->fConeData = nullptr;
    delete[] fSurfData->fGlobalTrans;
    fSurfData->fGlobalTrans = nullptr;
    delete[] fSurfData->fLocalTrans;
    fSurfData->fLocalTrans = nullptr;
    delete[] fSurfData->fFramedSurf;
    fSurfData->fFramedSurf = nullptr;
    delete[] fSurfData->fSides;
    fSurfData->fSides = nullptr;
    delete[] fSurfData->fCommonSurfaces;
    fSurfData->fCommonSurfaces = nullptr;
    delete[] fSurfData->fCandList;
    fSurfData->fCandList = nullptr;
    delete[] fSurfData->fCandidates;
    fSurfData->fCandidates = nullptr;
    delete[] fSurfData->fLocalSurf;
    fSurfData->fLocalSurf = nullptr;
    delete[] fSurfData->fShells;
    fSurfData->fShells = nullptr;
    delete[] fSurfData->fSurfShellList;
    fSurfData->fSurfShellList = nullptr;
    delete[] fSurfData->fLogicList;
    fSurfData->fLogicList = nullptr;
    // delete fSurfData;
    fSurfData = nullptr;
  }

  ~BrepHelper()
  {
    if (fSurfData) ClearData();
  }

  bool ApproxEqualTransformation(Transformation const &t1, Transformation const &t2)
  {
    if (!ApproxEqualVector(t1.Translation(), t2.Translation())) return false;
    for (int i = 0; i < 9; ++i)
      if (!ApproxEqual(t1.Rotation(i), t2.Rotation(i))) return false;
    return true;
  }

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
                [&](int i, int j) { return fCPUdata.fFramedSurf[i] < fCPUdata.fFramedSurf[j]; });
      for (int i = 0; i < side.fNsurf - 1; ++i) {
        for (int j = side.fNsurf - 1; j > i; --j) {
          if (EqualFrames(side, i, j)) removeSurface(side, j);
        }
      }
    };

    // lambda to find all parent frames on one side
    auto findParentFramedSurf = [&](Side &side) {
      if (!side.fNsurf) return;
      int top_parent  = side.fNsurf - 1;
      int num_parents = 0;
      // if there is a parent, it can only be at the last position after sorting
      for (auto parent_ind = top_parent; parent_ind >= 0; --parent_ind) {
        auto &parent_frame = fCPUdata.fFramedSurf[side.fSurfaces[parent_ind]];
        if (parent_frame.fParent < 0) num_parents++;
        auto parent_navind = parent_frame.fState;
        // loop remaining frames
        for (int i = 0; i < parent_ind; ++i) {
          auto &child_frame = fCPUdata.fFramedSurf[side.fSurfaces[i]];
          auto navind       = child_frame.fState;
          if (vecgeom::NavStateIndex::IsDescendentImpl(navind, parent_navind)) child_frame.fParent = parent_ind;
        }
      }
      side.fNumParents = num_parents;
    };

    sortAndRemoveCommonFrames(fCPUdata.fCommonSurfaces[common_id].fLeftSide);
    sortAndRemoveCommonFrames(fCPUdata.fCommonSurfaces[common_id].fRightSide);
    findParentFramedSurf(fCPUdata.fCommonSurfaces[common_id].fLeftSide);
    findParentFramedSurf(fCPUdata.fCommonSurfaces[common_id].fRightSide);
  }

  void ComputeDefaultStates(int common_id)
  {
    using vecgeom::NavStateIndex;
    // Computes the default states for each side of a common surface
    Side &left  = fCPUdata.fCommonSurfaces[common_id].fLeftSide;
    Side &right = fCPUdata.fCommonSurfaces[common_id].fRightSide;
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
      auto navind = fCPUdata.fFramedSurf[left.fSurfaces[isurf]].fState;
      minlevel    = std::min(minlevel, (int)NavStateIndex::GetLevelImpl(navind));
    }
    for (int isurf = 0; isurf < right.fNsurf; ++isurf) {
      auto navind = fCPUdata.fFramedSurf[right.fSurfaces[isurf]].fState;
      minlevel    = std::min(minlevel, (int)NavStateIndex::GetLevelImpl(navind));
    }

    // initialize the default state
    if (left.fNsurf > 0)
      default_ind = fCPUdata.fFramedSurf[left.fSurfaces[0]].fState;
    else if (right.fNsurf > 0)
      default_ind = fCPUdata.fFramedSurf[right.fSurfaces[0]].fState;

    for (int isurf = 0; isurf < left.fNsurf; ++isurf)
      default_ind = getCommonState(default_ind, fCPUdata.fFramedSurf[left.fSurfaces[isurf]].fState);
    for (int isurf = 0; isurf < right.fNsurf; ++isurf)
      default_ind = getCommonState(default_ind, fCPUdata.fFramedSurf[right.fSurfaces[isurf]].fState);

    if (NavStateIndex::GetLevelImpl(default_ind) == minlevel) default_ind = NavStateIndex::PopImpl(default_ind);
    fCPUdata.fCommonSurfaces[common_id].fDefaultState = default_ind;
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
      auto &framed_surf    = fSurfData->fFramedSurf[side.fSurfaces[i]];
      FrameType frame_type = framed_surf.fFrame.type;
      Vector3D<Real_t> local;

      WindowMask_t extentL;
      // Calculating the limits
      switch (frame_type) {
      case kWindow: {
        auto const &maskLocal = fSurfData->fWindowMasks[framed_surf.fFrame.id];
        maskLocal.GetExtent(extentL);
        break;
      }
      case kRing: {
        auto const &maskLocal = fSurfData->fRingMasks[framed_surf.fFrame.id];
        maskLocal.GetExtent(extentL);
        break;
      }
      case kQuadrilateral: {
        WindowMask_t extLocal;
        auto const &quad = fSurfData->fQuadMasks[framed_surf.fFrame.id];
        quad.GetExtent(extentL);
        break;
      }
      default:
        assert(0 && "Not implemented");
      } // case

      // This part updates extent
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(
          Vector3D<Real_t>{extentL.rangeU[0], extentL.rangeV[0], 0});
      updatePlaneExtent(ext, local);
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(
          Vector3D<Real_t>{extentL.rangeU[0], extentL.rangeV[1], 0});
      updatePlaneExtent(ext, local);
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(
          Vector3D<Real_t>{extentL.rangeU[1], extentL.rangeV[1], 0});
      updatePlaneExtent(ext, local);
      local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(
          Vector3D<Real_t>{extentL.rangeU[1], extentL.rangeV[0], 0});
      updatePlaneExtent(ext, local);
    } // for

    // Add new extent mask to vector
    int id = fCPUdata.fWindowMasks.size();
    fCPUdata.fWindowMasks.push_back(ext);
    side.fExtent.id = id;
  }

  // Computes bounding extent on a side of cylindrical surface
  void ComputeCylinderExtent(Side &side)
  {
    // Setting initial extent mask
    side.fExtent.type  = kZPhi;
    ZPhiMask_t sideext = fSurfData->GetZPhiMask(fSurfData->fFramedSurf[side.fSurfaces[0]].fFrame.id);

    // loop over remaining frames on the side
    for (int i = 1; i < side.fNsurf; ++i) {
      // convert extent of current frame to local coordinates
      auto &framed_surf = fSurfData->fFramedSurf[side.fSurfaces[i]];
      // Transform the ZPhi mask to the local system
      ZPhiMask_t const &extLocal = fSurfData->GetZPhiMask(framed_surf.fFrame.id);
      auto extFrame              = extLocal.InverseTransform(fSurfData->fGlobalTrans[framed_surf.fTrans]);
      // Combine with current extent
      sideext.CombineWith(extFrame);
    }

    // Add new extent mask to the vector
    int id = fCPUdata.fZPhiMasks.size();
    fCPUdata.fZPhiMasks.push_back(sideext);
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
    const char *types[] = {"planar", "cylindrical", "Conical", "Spherical", "Torus", "GenSecondOrder"};
    auto round0         = [](Real_t x) { return (std::abs(x) < vecgeom::kTolerance) ? Real_t(0) : x; };
    vecgeom::Vector3D<Real_t> normal;
    const vecgeom::Vector3D<Real_t> lnorm(0, 0, 1);
    auto const &surf = fSurfData->fCommonSurfaces[common_id];
    fSurfData->fGlobalTrans[surf.fTrans].InverseTransformDirection(lnorm, normal);
    printf("\n== common surface %d: type: %s, default state: ", common_id, types[int(surf.fType)]);
    vecgeom::NavStateIndex default_state(surf.fDefaultState);
    default_state.Print();
    printf(" transformation %d: ", surf.fTrans);
    fSurfData->fGlobalTrans[surf.fTrans].Print();
    switch (surf.fType) {
    case kPlanar: {
      WindowMask_t const &extL = fSurfData->fWindowMasks[surf.fLeftSide.fExtent.id];
      printf(
          "\n   \x1B[34mleft:\x1B[0m %d surfaces, num_parents=%d, extent %d: {u{%g, %g}, v{%g, %g}}, normal: (%g, %g, "
          "%g)\n",
          surf.fLeftSide.fNsurf, surf.fLeftSide.fNumParents, surf.fLeftSide.fExtent.id, extL.rangeU[0], extL.rangeU[1],
          extL.rangeV[0], extL.rangeV[1], round0(normal[0]), round0(normal[1]), round0(normal[2]));
      break;
    }
    case kCylindrical: {
      ZPhiMask_t const &extL = fSurfData->fZPhiMasks[surf.fLeftSide.fExtent.id];
      printf(
          "\n   \x1B[34mleft\x1B[0m: %d surfaces, num_parents=%d, extent %d: {z{%g, %g}, sphi{%g, %g}, ephi{%g, %g}}\n",
          surf.fLeftSide.fNsurf, surf.fLeftSide.fNumParents, surf.fLeftSide.fExtent.id, extL.rangeZ[0], extL.rangeZ[1],
          extL.vecSPhi[0], extL.vecSPhi[1], extL.vecEPhi[0], extL.vecEPhi[1]);
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
      printf("    surf %d: logic_id: %d parent: %d trans: ", idglob, placed.fLogicId, placed.fParent);
      fSurfData->fGlobalTrans[placed.fTrans].Print();
      printf("\n    ");
      vecgeom::NavStateIndex state(placed.fState);
      state.Print();
    }
    if (surf.fRightSide.fNsurf > 0) {
      switch (surf.fType) {
      case kPlanar: {
        WindowMask_t const &extR = fSurfData->fWindowMasks[surf.fRightSide.fExtent.id];
        printf("   \x1B[31mright:\x1B[0m %d surfaces, num_parents=%d, extent %d: {u{%g, %g}, v{%g, %g}}, normal: (%g, "
               "%g, %g)\n",
               surf.fRightSide.fNsurf, surf.fRightSide.fNumParents, surf.fRightSide.fExtent.id, extR.rangeU[0],
               extR.rangeU[1], extR.rangeV[0], extR.rangeV[1], round0(-normal[0]), round0(-normal[1]),
               round0(-normal[2]));
        break;
      }
      case kCylindrical: {
        ZPhiMask_t const &extR = fSurfData->fZPhiMasks[surf.fRightSide.fExtent.id];
        printf("   \x1B[31mright:\x1B[0m %d surfaces, num_parents=%d, extent %d: {z{%g, %g}, sphi{%g, %g}, ephi{%g, "
               "%g}}\n",
               surf.fRightSide.fNsurf, surf.fRightSide.fNumParents, surf.fRightSide.fExtent.id, extR.rangeZ[0],
               extR.rangeZ[1], extR.vecSPhi[0], extR.vecSPhi[1], extR.vecEPhi[0], extR.vecEPhi[1]);
        break;
      }
      case kConical:
      case kSpherical:
      case kTorus:
      case kGenSecondOrder:
      default:
        std::cout << "Case not implemented. " << std::endl;
      }
    } else {
      printf("   \x1B[31mright:\x1B[0m 0 surfaces\n");
    }

    for (int i = 0; i < surf.fRightSide.fNsurf; ++i) {
      int idglob         = surf.fRightSide.fSurfaces[i];
      auto const &placed = fSurfData->fFramedSurf[idglob];
      printf("    surf %d: logic_id: %d parent: %d trans: ", idglob, placed.fLogicId, placed.fParent);
      fSurfData->fGlobalTrans[placed.fTrans].Print();
      printf("\n    ");
      vecgeom::NavStateIndex state(placed.fState);
      state.Print();
    }
  }

  void SetNvolumes(int nvolumes)
  {
    if (fCPUdata.fShells.size() > 0) {
      std::cout << "BrepHelper::SetNvolumes already called for this instance.\n";
      return;
    }
    fCPUdata.fShells.resize(nvolumes);
  }

  bool CreateLocalSurfaces()
  {
    // Iterate logical volumes and create local surfaces
    std::vector<vecgeom::LogicalVolume *> volumes;
    auto n_registered_volumes = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
    vecgeom::GeoManager::Instance().GetAllLogicalVolumes(volumes);
    SetNvolumes(n_registered_volumes);
    // TODO: Implement a VUnplacedVolume::CreateSurfaces interface for surface creation
    // create a placeholder for surface data
    for (auto volume : volumes) {
      vecgeom::VUnplacedVolume const *solid = volume->GetUnplacedVolume();
      bool result                           = conv::CreateSolidSurfaces<Real_t>(solid, volume->id());
      if (!result) {
        std::cout << "BrepHelper::CreateLocalSurfaces: solid type not supported for volume: " << volume->GetName()
                  << "\n";
        solid->Print();
      }
      assert(result);
      // Finalize logic expression
      auto &crtlogic = fCPUdata.fShells[volume->id()].fLogic;
      logichelper::simplify_logic(crtlogic);
    }

    if (fVerbose > 0) {
      for (auto volume : volumes) {
        VolumeShellCPU const &shell = fCPUdata.fShells[volume->id()];
        printf("shell %d for volume %s:\n", volume->id(), volume->GetName());
        logichelper::print_logic(shell.fLogic);
        for (int lsurf_id : shell.fSurfaces) {
          FramedSurface const &lsurf = fCPUdata.fLocalSurfaces[lsurf_id];
          printf(" local surf %d (logic_id=%d): ", lsurf_id, lsurf.fLogicId);
          fCPUdata.fLocalTrans[lsurf.fTrans].Print();
          printf("\n");
        }
      }
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
      VolumeShellCPU const &shell = fCPUdata.fShells[vol->id()];
      for (int lsurf_id : shell.fSurfaces) {
        FramedSurface const &lsurf = fCPUdata.fLocalSurfaces[lsurf_id];
        // Ignore 'inside' helper surfaces having no frame
        if (lsurf.fFrame.type == kNoFrame) continue;
        Transformation global(trans);
        global.MultiplyFromRight(fCPUdata.fLocalTrans[lsurf.fTrans]);
        int trans_id = fCPUdata.fGlobalTrans.size();
        fCPUdata.fGlobalTrans.push_back(global);
        // Create the global surface
        int id_glob = fCPUdata.fFramedSurf.size();
        fCPUdata.fFramedSurf.push_back(
            {lsurf.fSurface, lsurf.fFrame, trans_id, lsurf.fUseSurfSafety, state.GetNavIndex()});
        fCPUdata.fFramedSurf.back().fLogicId = lsurf.fLogicId;
        if (fVerbose > 0) {
          std::cout << "framed surface " << id_glob << " for state: ";
          state.Print();
          std::cout << "  " << global << "\n";
        }
        CreateCommonSurface(id_glob, vol->id());
      }

      // Now do the daughters
      for (int id = 0; id < nd; ++id) {
        createCommonSurfaces(daughters[id]);
      }
      state.Pop();
    };

    // add identity first in the list of global transformations
    Transformation identity;
    fCPUdata.fGlobalTrans.push_back(identity);
    fCPUdata.fLocalTrans.push_back(identity);
    // add a dummy common surface since index 0 is not allowed for correctly handling sides
    fCPUdata.fCommonSurfaces.push_back({});

    createCommonSurfaces(vecgeom::GeoManager::Instance().GetWorld());

    for (size_t isurf = 1; isurf < fCPUdata.fCommonSurfaces.size(); ++isurf) {
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
      for (size_t isurf = 1; isurf < fCPUdata.fCommonSurfaces.size(); ++isurf)
        PrintCommonSurface(isurf);
    }

    if (fVerbose > 1) {
      PrintCandidateLists();
      std::cout << "Visited " << nphysical << " physical volumes, created " << fCPUdata.fCommonSurfaces.size() - 1
                << " common surfaces\n";
    }

    return true;
  }

  /// @brief Top-level conversion from a closed GeoManager to the surface model
  /// @return Successful conversion
  bool Convert()
  {
    bool success = CreateLocalSurfaces();
    if (!success) return false;
    success = CreateCommonSurfacesFlatTop();
    return success;
  }

  ///< This method uses the transformation T1 of the first placed surface on the left side (which always exists)
  ///< as transformation for the common surface, then recalculates the transformations of all placed
  ///< surfaces as T' = T1.Inverse() * T. If identity this will get the index 0.
  void ConvertTransformations(int idsurf)
  {
    auto &surf = fCPUdata.fCommonSurfaces[idsurf];
    // Adopt the transformation of the first surface on left for the common surface
    surf.fTrans = fCPUdata.fFramedSurf[surf.fLeftSide.fSurfaces[0]].fTrans;
    // Set transformation of first surface on left to identity
    fCPUdata.fFramedSurf[surf.fLeftSide.fSurfaces[0]].fTrans = 0;

    Transformation tsurfinv = fCPUdata.fGlobalTrans[surf.fTrans].Inverse();

    // Skip first surface on left side
    for (int i = 1; i < surf.fLeftSide.fNsurf; ++i) {
      int idglob = surf.fLeftSide.fSurfaces[i];
      auto &surf = fCPUdata.fFramedSurf[idglob];
      Transformation tnew(tsurfinv);
      tnew.MultiplyFromRight(fCPUdata.fGlobalTrans[surf.fTrans]);
      if (ApproxEqualTransformation(tnew, fCPUdata.fGlobalTrans[0])) {
        surf.fTrans = 0;
      } else {
        fCPUdata.fGlobalTrans[surf.fTrans] = tnew;
      }
    }

    // Convert right-side surfaces
    for (int i = 0; i < surf.fRightSide.fNsurf; ++i) {
      int idglob = surf.fRightSide.fSurfaces[i];
      auto &surf = fCPUdata.fFramedSurf[idglob];
      Transformation tnew(tsurfinv);
      tnew.MultiplyFromRight(fCPUdata.fGlobalTrans[surf.fTrans]);
      if (ApproxEqualTransformation(tnew, fCPUdata.fGlobalTrans[0])) {
        surf.fTrans = 0;
      } else {
        fCPUdata.fGlobalTrans[surf.fTrans] = tnew;
      }
    }
  }

  ///< This method creates helper lists of candidate surfaces for each navigation state
  void CreateCandidateLists()
  {
    int numNodes = vecgeom::GeoManager::Instance().GetTotalNodeCount() + 1; // count also outside state
    fCPUdata.fCandidates.reserve(numNodes);
    fCPUdata.fFrameInd.reserve(numNodes);

    // Lambda adding the surface id as candidate to all states from a side
    auto addSurfToSideStates = [&](int isurf, int iside) {
      Side const &side =
          (iside > 0) ? fCPUdata.fCommonSurfaces[isurf].fLeftSide : fCPUdata.fCommonSurfaces[isurf].fRightSide;
      for (int i = 0; i < side.fNsurf; ++i) {
        int idglob             = side.fSurfaces[i];
        auto const &framedsurf = fCPUdata.fFramedSurf[idglob];
        vecgeom::NavStateIndex state(framedsurf.fState);
        int state_id     = state.GetId();
        auto isignedsurf = iside * isurf;
        if (!fCPUdata.fCandidates[state_id].size() || fCPUdata.fCandidates[state_id].back() != isignedsurf) {
          fCPUdata.fCandidates[state_id].push_back(isignedsurf);
          fCPUdata.fFrameInd[state_id].push_back(i);
        }
      }
    };

    // prepare all lists
    for (int i = 0; i < numNodes; ++i) {
      fCPUdata.fCandidates.push_back({});
      fCPUdata.fFrameInd.push_back({});
    }

    // loop over all common surfaces and add their index in the appropriate list
    for (size_t isurf = 1; isurf < fCPUdata.fCommonSurfaces.size(); ++isurf) {
      auto const &surf = fCPUdata.fCommonSurfaces[isurf];
      // Add to default surface state
      vecgeom::NavStateIndex state(surf.fDefaultState);
      fCPUdata.fCandidates[state.GetId()].push_back(-isurf);
      fCPUdata.fFrameInd[state.GetId()].push_back(-1); // means this state is the default for isurf
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

  void PrintSurfData()
  {
    constexpr int megabyte = 1024 * 1024;
    float total = 0, size = 0;
    std::cout << "___________________________________________________________________________________\n";
    std::cout << " Surface model info:  " << vecgeom::GeoManager::Instance().GetTotalNodeCount() + 1 << " touchables\n";
    size = float(fSurfData->fNshells * sizeof(VolumeShell) + fSurfData->fNlocalSurf * sizeof(int)) / megabyte;
    total += size;
    std::cout << "    volume shells          = " << fSurfData->fNshells << " [" << size << " MB]\n";
    size = float(fSurfData->fNlocalTrans * sizeof(Transformation)) / megabyte;
    total += size;
    std::cout << "    local transformations  = " << fSurfData->fNlocalTrans << " [" << size << " MB]\n";
    size = float(fSurfData->fNglobalTrans * sizeof(Transformation)) / megabyte;
    total += size;
    std::cout << "    global transformations = " << fSurfData->fNglobalTrans << " [" << size << " MB]\n";
    size = float(fSurfData->fNlocalSurf * sizeof(FramedSurface)) / megabyte;
    total += size;
    std::cout << "    local surfaces         = " << fSurfData->fNlocalSurf << " [" << size << " MB]\n";
    size = float(fSurfData->fNglobalSurf * sizeof(FramedSurface)) / megabyte;
    total += size;
    std::cout << "    global surfaces        = " << fSurfData->fNglobalSurf << " [" << size << " MB]\n";
    size = float(fSurfData->fNcommonSurf * sizeof(CommonSurface) + fSurfData->fNsides * sizeof(int)) / megabyte;
    total += size;
    std::cout << "    common surfaces        = " << fSurfData->fNcommonSurf << " [" << size << " MB]\n";
    size = float(fSurfData->fNcandList * sizeof(int)) / megabyte;
    total += size;
    std::cout << "    candidates             = " << fSurfData->fNcandList << " [" << size << " MB]\n";
    size = float(fSurfData->fNwindows * sizeof(WindowMask_t)) / megabyte;
    total += size;
    std::cout << "    window masks           = " << fSurfData->fNwindows << " [" << size << " MB]\n";
    size = float(fSurfData->fNcylsph * sizeof(FramedSurface)) / megabyte;
    total += size;
    std::cout << "    cyl/sph masks          = " << fSurfData->fNcylsph << " [" << size << " MB]\n";
    size = float(fSurfData->fNrings * sizeof(RingMask_t)) / megabyte;
    total += size;
    std::cout << "    ring masks             = " << fSurfData->fNrings << " [" << size << " MB]\n";
    size = float(fSurfData->fNzphis * sizeof(ZPhiMask_t)) / megabyte;
    total += size;
    std::cout << "    Z/phi masks            = " << fSurfData->fNzphis << " [" << size << " MB]\n";
    size = float(fSurfData->fNtriangs * sizeof(TriangleMask_t)) / megabyte;
    total += size;
    std::cout << "    triangle masks         = " << fSurfData->fNtriangs << " [" << size << " MB]\n";
    size = float(fSurfData->fNquads * sizeof(QuadMask_t)) / megabyte;
    total += size;
    std::cout << "    quad masks             = " << fSurfData->fNquads << " [" << size << " MB]\n";
    std::cout << " Total: " << total << "[MB]\n";
    std::cout << "___________________________________________________________________________________\n";
  }

private:
  int CreateCommonSurface(int idglob, int volId)
  {
    bool flip, flip_bool;
    auto approxEqual = [&](int idglob1, int idglob2) {
      flip                    = false;
      flip_bool               = false;
      FramedSurface const &s1 = fCPUdata.fFramedSurf[idglob1];
      FramedSurface const &s2 = fCPUdata.fFramedSurf[idglob2];
      // Surfaces may be in future "compatible" even if they are not the same, for now enforce equality
      if (s1.fSurface.type != s2.fSurface.type) return false;

      // Check if the surfaces may be flipped because of Boolean negation
      flip_bool = s1.fLogicId * s2.fLogicId < 0;

      // Check if the 2 surfaces are parallel
      Transformation const &t1 = fCPUdata.fGlobalTrans[s1.fTrans];
      Transformation const &t2 = fCPUdata.fGlobalTrans[s2.fTrans];
      // Check if the rotations are matching. The z axis inverse-transformed
      // with the two rotations should end up as aligned vectors. This is
      // true for planes (Z is the normal) but also for tubes/cones where
      // Z is the axis of symmetry
      vecgeom::Vector3D<double> const zaxis(0, 0, 1);
      auto z1 = t1.InverseTransformDirection(zaxis);
      auto z2 = t2.InverseTransformDirection(zaxis);
      if (!ApproxEqualVector(z1.Cross(z2), {0, 0, 0})) return false;
      // Calculate normalized connection vector between the two transformations
      // Use double precision explicitly
      vecgeom::Vector3D<double> tdiff = t1.Translation() - t2.Translation();
      bool same_tr                    = ApproxEqualVector(tdiff, {0, 0, 0});
      vecgeom::Vector3D<double> ldir;
      switch (s1.fSurface.type) {
      case kPlanar:
        flip = z1.Dot(z2) < 0;
        if (same_tr) break;
        // For planes to match, the connecting vector must be along the planes
        tdiff.Normalize();
        t1.TransformDirection(tdiff, ldir);
        if (std::abs(ldir[2]) > vecgeom::kTolerance) return false;
        break;
      case kCylindrical:
        if (std::abs(fCPUdata.fCylSphData[s1.fSurface.id].Radius() - fCPUdata.fCylSphData[s2.fSurface.id].Radius()) >
            vecgeom::kTolerance)
          return false;
        // Check if the cylynders are flipped with respect to each other
        flip = fCPUdata.fCylSphData[s1.fSurface.id].IsFlipped() ^ fCPUdata.fCylSphData[s2.fSurface.id].IsFlipped();
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
      return true;
    };

    auto surfHashUgly = [&](int idglobal) {
      // Compute hash for the surface rotation
      constexpr int nth           = 1000;
      constexpr int nph           = 1000;
      FramedSurface const &surf   = fCPUdata.fFramedSurf[idglobal];
      Transformation const &trans = fCPUdata.fGlobalTrans[surf.fTrans];
      // convert local Z axis to the global frame
      vecgeom::Vector3D<double> const zaxis(0, 0, 1);
      auto vzglob = trans.InverseTransformDirection(zaxis);
      int ith     = nth * vecCore::math::Abs(vzglob.z());
      // backward vectors should generate the same hash
      Real_t phi = vzglob.Phi() + vecgeom::kPi * int(vzglob.z() < 0);
      phi        = fmod(fmod(phi, vecgeom::kTwoPi) + vecgeom::kTwoPi, vecgeom::kTwoPi); // [0, 2pi]
      if (ith == 0) phi = fmod(phi, vecgeom::kPi);
      int iph = nph * phi * vecgeom::kTwoPiInv + 0.5;
      if (ith == nth) {
        ith--;
        iph = 0;
      }
      auto hash = ith * nph + iph;
      // std::cout << "framed_surf " << idglob << ": vzglob=" << vzglob << " ith=" << ith << " iph=" << iph << " hash= "
      // << hash << std::endl;
      return hash;
    };

#if (1)
    FramedSurface const &surf = fCPUdata.fFramedSurf[idglob];
    auto hash                 = surfHashUgly(idglob);
    // Get the compatible surfaces
    auto range          = fCPUdata.fSurfHash.equal_range(hash);
    bool found_dup_surf = false;
    int id              = -1;
    flip ^= flip_bool;
    for (auto it = range.first; it != range.second; ++it) {
      const auto &other_id = fCPUdata.fCommonSurfaces[it->second].fLeftSide.fSurfaces[0];
      // Do not de-duplicate surfaces if they do not belong to the same Boolean volume.
      // This is needed because safety for Booleans must be evaluated only once based on the volume logic expression.
      // Safety evaluation is triggered by the first Boolean surface found closest. All candidate surfaces to be checked
      // for the same Boolean volume must be consecutive, to allow caching the result.
      FramedSurface const &othersurf = fCPUdata.fFramedSurf[other_id];
      if (surf.fLogicId && surf.fLogicId != othersurf.fLogicId) continue;

      if (approxEqual(other_id, idglob)) {
        // Do not allow surfaces of the same volume on different sides of the same common surface, otherwise the surface
        // will be missed when coming from the entering side.
        if (flip && othersurf.VolumeId() == volId) continue;
        found_dup_surf = true;
        id             = it->second;
        auto &crt_side = flip ? fCPUdata.fCommonSurfaces[id].fRightSide : fCPUdata.fCommonSurfaces[id].fLeftSide;
        // The common surface is compatible only if the parent state for the current framed surface
        // has a frame on the same side or it is already the default state.
        auto parent_state_index = vecgeom::NavStateIndex::PopImpl(fCPUdata.fFramedSurf[idglob].fState);
        if (fCPUdata.fCommonSurfaces[id].fDefaultState != parent_state_index) {
          // To be compatible, a surface of the parent state MUST exist on the same side
          bool has_parent = false;
          for (auto isurf = 0; isurf < crt_side.fNsurf; ++isurf) {
            has_parent = fCPUdata.fFramedSurf[crt_side.fSurfaces[isurf]].fState == parent_state_index;
            if (has_parent) break;
          }
          if (!has_parent) {
            found_dup_surf = false;
            continue;
          }
        }
        // Add the global surface to the appropriate side
        crt_side.AddSurface(idglob);
        break;
      }
    }
    if (!found_dup_surf) {
      // Construct a new common surface from the current placed global surface
      // Set the common state to be the parent of the idglob surface state
      id = fCPUdata.fCommonSurfaces.size();
      fCPUdata.fCommonSurfaces.push_back({fCPUdata.fFramedSurf[idglob].fSurface.type, idglob});
      fCPUdata.fCommonSurfaces[id].fDefaultState = vecgeom::NavStateIndex::PopImpl(fCPUdata.fFramedSurf[idglob].fState);
      fCPUdata.fSurfHash.insert(std::make_pair(hash, id));
    }
#else
    // this may be slow
    auto it = std::find_if(std::begin(fCPUdata.fCommonSurfaces), std::end(fCPUdata.fCommonSurfaces),
                           [&](const CommonSurface &t) {
                             return (t.fLeftSide.fNsurf > 0) ? approxEqual(t.fLeftSide.fSurfaces[0], idglob) : false;
                           });
    int id  = -1;
    if (it != std::end(fCPUdata.fCommonSurfaces)) {
      id = int(it - std::begin(fCPUdata.fCommonSurfaces));
      // Add the global surface to the appropriate side
      if (flip)
        (*it).fRightSide.AddSurface(idglob);
      else
        (*it).fLeftSide.AddSurface(idglob);

    } else {
      // Construct a new common surface from the current placed global surface
      id = fCPUdata.fCommonSurfaces.size();
      fCPUdata.fCommonSurfaces.push_back({fCPUdata.fFramedSurf[idglob].fSurface.type, idglob});
    }
#endif
    return id;
  }

  ///< This function evaluates if the frames of two placed surfaces on the same side
  ///< of a common surface are matching
  bool EqualFrames(Side const &side, int i1, int i2)
  {
    using Vector3D = vecgeom::Vector3D<Real_t>;

    FramedSurface const &s1 = fCPUdata.fFramedSurf[side.fSurfaces[i1]];
    FramedSurface const &s2 = fCPUdata.fFramedSurf[side.fSurfaces[i2]];
    if (s1.fFrame.type != s2.fFrame.type) return false;
    // Get displacement vector between the 2 frame centers and check if it has null length
    Transformation const &t1 = fCPUdata.fGlobalTrans[s1.fTrans];
    Transformation const &t2 = fCPUdata.fGlobalTrans[s2.fTrans];
    Vector3D tdiff           = t1.Translation() - t2.Translation();
    // TODO: Check if this has to always hold with the new mask types!!
    if (!ApproxEqualVector(tdiff, {0, 0, 0})) return false;

    // Different treatment of different frame types
    switch (s1.fFrame.type) {
    case kRangeZ:
      break;
    case kRing: {
      auto mask1 = fCPUdata.fRingMasks[s1.fFrame.id];
      auto mask2 = fCPUdata.fRingMasks[s2.fFrame.id];

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
      auto mask1 = fCPUdata.fZPhiMasks[s1.fFrame.id];
      auto mask2 = fCPUdata.fZPhiMasks[s2.fFrame.id];

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
      auto frameData1 = fCPUdata.fWindowMasks[s1.fFrame.id];
      auto frameData2 = fCPUdata.fWindowMasks[s2.fFrame.id];

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
    case kQuadrilateral: {
      // to be implemented
      break;
    }
    default:
      break;
    };
    return false;
  }

  // A function to update all mask containers. Needs to be called
  // when updating masks after creating both frames and extents.
  void UpdateMaskData()
  {
    fSurfData->fNwindows    = fCPUdata.fWindowMasks.size();
    fSurfData->fWindowMasks = new WindowMask_t[fCPUdata.fWindowMasks.size()];
    for (size_t i = 0; i < fCPUdata.fWindowMasks.size(); ++i)
      fSurfData->fWindowMasks[i] = fCPUdata.fWindowMasks[i];

    fSurfData->fNrings    = fCPUdata.fRingMasks.size();
    fSurfData->fRingMasks = new RingMask_t[fCPUdata.fRingMasks.size()];
    for (size_t i = 0; i < fCPUdata.fRingMasks.size(); ++i)
      fSurfData->fRingMasks[i] = fCPUdata.fRingMasks[i];

    fSurfData->fNzphis    = fCPUdata.fZPhiMasks.size();
    fSurfData->fZPhiMasks = new ZPhiMask_t[fCPUdata.fZPhiMasks.size()];
    for (size_t i = 0; i < fCPUdata.fZPhiMasks.size(); ++i)
      fSurfData->fZPhiMasks[i] = fCPUdata.fZPhiMasks[i];

    fSurfData->fNquads    = fCPUdata.fQuadMasks.size();
    fSurfData->fQuadMasks = new QuadMask_t[fCPUdata.fQuadMasks.size()];
    for (size_t i = 0; i < fCPUdata.fQuadMasks.size(); ++i)
      fSurfData->fQuadMasks[i] = fCPUdata.fQuadMasks[i];
  }

  ///< The method updates the SurfData storage
  void UpdateSurfData()
  {
    // Create and copy surface data
    fSurfData->fNcylsph    = fCPUdata.fCylSphData.size();
    fSurfData->fCylSphData = new CylData_t[fCPUdata.fCylSphData.size()];
    for (size_t i = 0; i < fCPUdata.fCylSphData.size(); ++i)
      fSurfData->fCylSphData[i] = fCPUdata.fCylSphData[i];

    fSurfData->fNcone    = fCPUdata.fConeData.size();
    fSurfData->fConeData = new ConeData_t[fCPUdata.fConeData.size()];
    for (size_t i = 0; i < fCPUdata.fConeData.size(); ++i)
      fSurfData->fConeData[i] = fCPUdata.fConeData[i];

    // Copy transformations
    fSurfData->fNglobalTrans = fCPUdata.fGlobalTrans.size();
    fSurfData->fGlobalTrans  = new Transformation[fCPUdata.fGlobalTrans.size()];
    for (size_t i = 0; i < fCPUdata.fGlobalTrans.size(); ++i)
      fSurfData->fGlobalTrans[i] = fCPUdata.fGlobalTrans[i];
    fSurfData->fNlocalTrans = fCPUdata.fLocalTrans.size();
    fSurfData->fLocalTrans  = new Transformation[fCPUdata.fLocalTrans.size()];
    for (size_t i = 0; i < fCPUdata.fLocalTrans.size(); ++i)
      fSurfData->fLocalTrans[i] = fCPUdata.fLocalTrans[i];

    // Copy global surfaces
    auto numGlobalSurf      = fCPUdata.fFramedSurf.size();
    fSurfData->fNglobalSurf = numGlobalSurf;
    fSurfData->fFramedSurf  = new FramedSurface[numGlobalSurf];
    for (size_t i = 0; i < numGlobalSurf; ++i)
      fSurfData->fFramedSurf[i] = fCPUdata.fFramedSurf[i];

    // Copy common surfaces
    size_t size_sides = 0;
    for (auto const &surf : fCPUdata.fCommonSurfaces)
      size_sides += surf.fLeftSide.fNsurf + surf.fRightSide.fNsurf;

    // Create Masks
    UpdateMaskData();

    fSurfData->fNsides         = size_sides;
    fSurfData->fSides          = new int[size_sides];
    int *current_side          = fSurfData->fSides;
    fSurfData->fNcommonSurf    = fCPUdata.fCommonSurfaces.size();
    fSurfData->fCommonSurfaces = new CommonSurface[fCPUdata.fCommonSurfaces.size()];
    for (size_t i = 0; i < fCPUdata.fCommonSurfaces.size(); ++i) {
      // Raw copy of surface (wrong pointers in sides)
      fSurfData->fCommonSurfaces[i] = fCPUdata.fCommonSurfaces[i];
      // Copy left sides content in buffer
      for (auto isurf = 0; isurf < fCPUdata.fCommonSurfaces[i].fLeftSide.fNsurf; ++isurf)
        current_side[isurf] = fCPUdata.fCommonSurfaces[i].fLeftSide.fSurfaces[isurf];
      // Make left sides arrays point to the buffer
      fSurfData->fCommonSurfaces[i].fLeftSide.fSurfaces = current_side;
      current_side += fCPUdata.fCommonSurfaces[i].fLeftSide.fNsurf;

      // Copy right sides content in buffer
      for (auto isurf = 0; isurf < fCPUdata.fCommonSurfaces[i].fRightSide.fNsurf; ++isurf)
        current_side[isurf] = fCPUdata.fCommonSurfaces[i].fRightSide.fSurfaces[isurf];
      // Make right sides arrays point to the buffer
      fSurfData->fCommonSurfaces[i].fRightSide.fSurfaces = current_side;
      current_side += fCPUdata.fCommonSurfaces[i].fRightSide.fNsurf;
      // Copy parent surface indices
      fSurfData->fCommonSurfaces[i].fLeftSide.fNumParents  = fCPUdata.fCommonSurfaces[i].fLeftSide.fNumParents;
      fSurfData->fCommonSurfaces[i].fRightSide.fNumParents = fCPUdata.fCommonSurfaces[i].fRightSide.fNumParents;
    }

    // Copy candidates lists
    auto size_candidates = 0;
    for (auto const &list : fCPUdata.fCandidates)
      size_candidates += list.size();

    fSurfData->fNcandList   = 2 * size_candidates;
    fSurfData->fCandList    = new int[2 * size_candidates];
    int *current_candidates = fSurfData->fCandList;
    fSurfData->fNcandidates = fCPUdata.fCandidates.size();
    fSurfData->fCandidates  = new Candidates[fCPUdata.fCandidates.size()];
    for (size_t i = 0; i < fCPUdata.fCandidates.size(); ++i) {
      auto ncand                       = fCPUdata.fCandidates[i].size();
      fSurfData->fCandidates[i].fNcand = ncand;
      for (size_t icand = 0; icand < ncand; icand++) {
        current_candidates[icand]         = (fCPUdata.fCandidates[i])[icand];
        current_candidates[ncand + icand] = (fCPUdata.fFrameInd[i])[icand];
      }
      fSurfData->fCandidates[i].fCandidates = current_candidates;
      current_candidates += fCPUdata.fCandidates[i].size();
      fSurfData->fCandidates[i].fFrameInd = current_candidates;
      current_candidates += fCPUdata.fCandidates[i].size();
    }

    // Copy local surfaces
    auto numLocalSurf      = fCPUdata.fLocalSurfaces.size();
    fSurfData->fNlocalSurf = numLocalSurf;
    fSurfData->fLocalSurf  = new FramedSurface[numLocalSurf];
    for (size_t i = 0; i < numLocalSurf; ++i)
      fSurfData->fLocalSurf[i] = fCPUdata.fLocalSurfaces[i];

    // Copy volume shells
    auto numShells            = fCPUdata.fShells.size();
    fSurfData->fNshells       = numShells;
    fSurfData->fShells        = new VolumeShell[numShells];
    fSurfData->fSurfShellList = new int[numLocalSurf];
    int *current_surf         = fSurfData->fSurfShellList;
    size_t sizeLogic          = 0;
    for (size_t i = 0; i < numShells; ++i) {
      sizeLogic += fCPUdata.fShells[i].fLogic.size();
    }
    fSurfData->fLogicList    = new logic_int[sizeLogic];
    fSurfData->fNlogic       = sizeLogic;
    logic_int *current_logic = fSurfData->fLogicList;

    for (size_t i = 0; i < numShells; ++i) {
      auto const &surfaces         = fCPUdata.fShells[i].fSurfaces;
      auto nsurf                   = surfaces.size();
      fSurfData->fShells[i].fNsurf = nsurf;
      for (size_t isurf = 0; isurf < nsurf; isurf++)
        current_surf[isurf] = surfaces[isurf];
      fSurfData->fShells[i].fSurfaces = current_surf;
      current_surf += nsurf;
      auto nlogic                        = fCPUdata.fShells[i].fLogic.size();
      fSurfData->fShells[i].fLogic.size_ = nlogic;
      fSurfData->fShells[i].fLogic.data_ = current_logic;
      int iitem                          = 0;
      for (auto item : fCPUdata.fShells[i].fLogic)
        fSurfData->fShells[i].fLogic.data_[iitem++] = item;
      current_logic += nlogic;
    }
  }
};

} // namespace vgbrep
#endif
