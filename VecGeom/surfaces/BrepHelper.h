#ifndef VECGEOM_SURFACE_BREPHELPER_H_
#define VECGEOM_SURFACE_BREPHELPER_H_

#include <cassert>
#include <functional>
#include <VecGeom/surfaces/Model.h>

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
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
  using SurfData_t  = SurfData<Real_t>;
  using CylData_t   = CylData<Real_t>;
  using ConeData_t  = ConeData<Real_t>;
  using SphData_t   = SphData<Real_t>;
  using RangeMask_t = RangeMask<Real_t>;
  using Extent_t    = Extent<Real_t>;
  //  using Vector         = vecgeom::Vector3D<Real_t>;

private:
  int fVerbose{0};                   ///< verbosity level
  SurfData_t *fSurfData{nullptr};    ///< Surface data
  SurfData_t *fSurfDataGPU{nullptr}; ///< Surface data on device

  std::vector<Extent_t> fExtents;             ///< List of extents
  std::vector<RangeMask_t> fRanges;           ///< List of ranges
  std::vector<Frame> fFrames;                 ///< vector of masks
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
    fRanges.clear();
    std::vector<RangeMask<Real_t>>().swap(fRanges);
    fExtents.clear();
    std::vector<Extent_t>().swap(fExtents);
    fFrames.clear();
    std::vector<Frame>().swap(fFrames);
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
      int parent_ind = side.fNsurf - 1;
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

  void ComputeExtents()
  {
    // This computes extents for all sides of common surfaces
    auto updateExtent = [](Extent_t &e, Vector3D<Real_t> const &pt) {
      e.rangeU[0] = std::min(e.rangeU[0], pt[0]);
      e.rangeU[1] = std::max(e.rangeU[1], pt[0]);
      e.rangeV[0] = std::min(e.rangeV[0], pt[1]);
      e.rangeV[1] = std::max(e.rangeV[1], pt[1]);
    };

    auto computeSingleSideExtent = [&](SurfaceType type, Side &side, Extent_t &ext) {
      constexpr Real_t kBig = 1.e30;
      ext.rangeU.Set(kBig, -kBig);
      ext.rangeV.Set(kBig, -kBig);

      for (int i = 0; i < side.fNsurf; ++i) {
        // convert surface frame to local coordinates
        Extent_t extLocal;
        auto framed_surf = fSurfData->fFramedSurf[side.fSurfaces[i]];
        Vector3D<Real_t> local;
        framed_surf.fFrame.GetExtent(extLocal, *fSurfData);
        switch (type) {
        case kPlanar:
          local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(
              Vector3D<Real_t>{extLocal.rangeU[0], extLocal.rangeV[0], 0});
          updateExtent(ext, local);
          local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(
              Vector3D<Real_t>{extLocal.rangeU[0], extLocal.rangeV[1], 0});
          updateExtent(ext, local);
          local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(
              Vector3D<Real_t>{extLocal.rangeU[1], extLocal.rangeV[1], 0});
          updateExtent(ext, local);
          local = fSurfData->fGlobalTrans[framed_surf.fTrans].InverseTransform(
              Vector3D<Real_t>{extLocal.rangeU[1], extLocal.rangeV[0], 0});
          updateExtent(ext, local);
          break;
        case kCylindrical:
        case kConical:
        case kSpherical:
        case kTorus:
        case kGenSecondOrder:
          break;
        };
      }
    };

    for (int common_id = 1; common_id < fSurfData->fNcommonSurf; ++common_id) {
      Extent_t common_extent;
      if (fSurfData->fCommonSurfaces[common_id].fLeftSide.fNsurf) {
        computeSingleSideExtent(fSurfData->fCommonSurfaces[common_id].fType,
                                fSurfData->fCommonSurfaces[common_id].fLeftSide, common_extent);
        int iextent = fExtents.size();
        fExtents.push_back(common_extent);
        fSurfData->fCommonSurfaces[common_id].fLeftSide.fExtent = iextent;
      }
      if (fSurfData->fCommonSurfaces[common_id].fRightSide.fNsurf) {
        computeSingleSideExtent(fSurfData->fCommonSurfaces[common_id].fType,
                                fSurfData->fCommonSurfaces[common_id].fRightSide, common_extent);
        int iextent = fExtents.size();
        fExtents.push_back(common_extent);
        fSurfData->fCommonSurfaces[common_id].fRightSide.fExtent = iextent;
      }
    }
    // Create extents in the surface data structure
    fSurfData->fNextents = fExtents.size();
    fSurfData->fExtents  = new Extent_t[fExtents.size()];
    for (size_t i = 0; i < fExtents.size(); ++i)
      fSurfData->fExtents[i] = fExtents[i];
  }

  void PrintCommonSurface(int common_id)
  {
    auto const &surf = fSurfData->fCommonSurfaces[common_id];
    printf("== common surface %d: default state: ", common_id);
    vecgeom::NavStateIndex default_state(surf.fDefaultState);
    default_state.Print();
    printf(" transformation %d: ", surf.fTrans);
    fSurfData->fGlobalTrans[surf.fTrans].Print();
    Extent_t const &extL = fSurfData->fExtents[surf.fLeftSide.fExtent];
    printf("\n   left: %d surfaces, parent=%d, extent %d: {{%g, %g}, {%g, %g}}\n", surf.fLeftSide.fNsurf,
           surf.fLeftSide.fParentSurf, surf.fLeftSide.fExtent,
           extL.rangeU[0], extL.rangeU[1], extL.rangeV[0], extL.rangeV[1]);
    for (int i = 0; i < surf.fLeftSide.fNsurf; ++i) {
      int idglob         = surf.fLeftSide.fSurfaces[i];
      auto const &placed = fSurfData->fFramedSurf[idglob];
      printf("    surf %d: trans: ", idglob);
      fSurfData->fGlobalTrans[placed.fTrans].Print();
      printf(", ");
      vecgeom::NavStateIndex state(placed.fState);
      state.Print();
    }
    Extent_t const &extR = fSurfData->fExtents[surf.fRightSide.fExtent];
    if (surf.fRightSide.fNsurf > 0)
      printf("   right: %d surfaces, parent=%d, extent %d: {{%g, %g}, {%g, %g}}\n", surf.fRightSide.fNsurf,
             surf.fRightSide.fParentSurf, surf.fRightSide.fExtent,
             extR.rangeU[0], extR.rangeU[1], extR.rangeV[0], extR.rangeV[1]);
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
    // creat a placeholder for surface data
    for (auto volume : volumes) {
      vecgeom::VUnplacedVolume const *solid = volume->GetUnplacedVolume();
      vecgeom::UnplacedBox const *box       = dynamic_cast<vecgeom::UnplacedBox const *>(solid);
      if (!box) {
        std::cout << "testEm3: solid type not supported for volume: " << volume->GetName() << "\n";
        return false;
      }
      CreateBoxSurfaces(*box, volume->id());
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
        fFramedSurf.push_back({lsurf.fSurface, lsurf.fFrame, trans_id, state.GetNavIndex()});
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

    if (fVerbose > 1) {
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
        int idglob         = side.fSurfaces[i];
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

  Frame CreateFrame(FrameType type, RangeMask<Real_t> const &mask)
  {
    int id = fRanges.size();
    fRanges.push_back(mask);
    return Frame(type, id);
  }

  int CreateLocalTransformation(Transformation const &trans)
  {
    int id = fLocalTrans.size();
    fLocalTrans.push_back(trans);
    return id;
  }

  int CreateLocalSurface(UnplacedSurface const &unplaced, Frame const &frame, int trans)
  {
    int id = fLocalSurfaces.size();
    fLocalSurfaces.push_back({unplaced, frame, trans});
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
        if (std::abs(fCylSphData[s1.fSurface.id].radius - fCylSphData[s1.fSurface.id].radius) > vecgeom::kTolerance)
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
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, {box.z(), box.y()}),
                               CreateLocalTransformation({-box.x(), 0, 0, 90, -90, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // surface at +dx:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, {box.z(), box.y()}),
                               CreateLocalTransformation({box.x(), 0, 0, 90, 90, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // surface at -dy:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, {box.x(), box.z()}),
                               CreateLocalTransformation({0, -box.y(), 0, 0, 90, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // surface at +dy:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, {box.x(), box.z()}),
                               CreateLocalTransformation({0, box.y(), 0, 0, -90, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // surface at -dz:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, {box.x(), box.y()}),
                               CreateLocalTransformation({0, 0, -box.z(), 0, 180, 0}));
    AddSurfaceToShell(logical_id, isurf);
    // surface at +dz:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, {box.x(), box.y()}),
                               CreateLocalTransformation({0, 0, box.z(), 0, 0, 0}));
    AddSurfaceToShell(logical_id, isurf);
  }

  ///< This function evaluates if the frames of two placed surfaces on the same side
  ///< of a common surface are matching
  bool EqualFrames(Side const &side, int i1, int i2)
  {
    FramedSurface const &s1 = fFramedSurf[side.fSurfaces[i1]];
    FramedSurface const &s2 = fFramedSurf[side.fSurfaces[i2]];
    if (s1.fFrame.type != s2.fFrame.type) return false;
    // Get displacement vector between the 2 frame centers and check if it has null length
    Transformation const &t1        = fGlobalTrans[s1.fTrans];
    Transformation const &t2        = fGlobalTrans[s2.fTrans];
    vecgeom::Vector3D<double> tdiff = t1.Translation() - t2.Translation();
    if (!ApproxEqualVector(tdiff, {0, 0, 0})) return false;

    auto frameData1 = fRanges[s1.fFrame.id];
    auto frameData2 = fRanges[s2.fFrame.id];
    vecgeom::Vector3D<double> v1(frameData1.range[0], frameData1.range[1], 0);
    vecgeom::Vector3D<double> v1_transposed(frameData1.range[1], frameData1.range[0], 0);
    vecgeom::Vector3D<double> v2(frameData2.range[0], frameData2.range[1], 0);

    // Different treatment of different frame types
    switch (s1.fFrame.type) {
    case kRangeZ:
    case kRangeCyl:
    case kRangeSph:
      if (ApproxEqualVector(v1, v2)) return true;
      return false;
    case kWindow:
      // Here we can have matching also if the window is rotated
      if (!ApproxEqualVector(v1, v2) && !ApproxEqualVector(v1_transposed, v2)) return false;
      {
        // The vectors connecting the local window center to the closest edge
        // must be aligned in the transformed frames
        vecgeom::Vector3D<double> vshort;
        if (v1[0] > v1[1])
          vshort.Set(0, v1[1], 0);
        else
          vshort.Set(v1[0], 0, 0);
        auto v1glob = t1.InverseTransform(vshort);
        if (v2[0] > v2[1])
          vshort.Set(0, v2[1], 0);
        else
          vshort.Set(v2[0], 0, 0);
        auto v2glob = t2.InverseTransform(vshort);
        if (ApproxEqualVector(v1glob.Cross(v2glob), {0, 0, 0})) return true;
      }
      break;
    case kTriangle:
      // to be implemented
      break;
    };
    return false;
  }

  ///< The method updates the SurfData storage
  void UpdateSurfData()
  {
    // Create and copy surface data
    fSurfData->fCylSphData = new CylData_t[fCylSphData.size()];
    fSurfData->fNcylsph    = fCylSphData.size();
    for (size_t i = 0; i < fCylSphData.size(); ++i)
      fSurfData->fCylSphData[i] = fCylSphData[i];

    fSurfData->fNcone    = fConeData.size();
    fSurfData->fConeData = new ConeData_t[fConeData.size()];
    for (size_t i = 0; i < fConeData.size(); ++i)
      fSurfData->fConeData[i] = fConeData[i];

    fSurfData->fNrange    = fRanges.size();
    fSurfData->fRangeData = new RangeMask_t[fRanges.size()];
    for (size_t i = 0; i < fRanges.size(); ++i)
      fSurfData->fRangeData[i] = fRanges[i];

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
      fSurfData->fCommonSurfaces[i].fLeftSide.fParentSurf = fCommonSurfaces[i].fLeftSide.fParentSurf;
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
      auto ncand = fCandidates[i].size();
      fSurfData->fCandidates[i].fNcand = ncand;
      for (size_t icand = 0; icand < ncand; icand++) {
        current_candidates[icand] = (fCandidates[i])[icand];
        current_candidates[ncand + icand] = (fFrameInd[i])[icand];
      }
      fSurfData->fCandidates[i].fCandidates = current_candidates;
      current_candidates += fCandidates[i].size();
      fSurfData->fCandidates[i].fFrameInd   = current_candidates;
      current_candidates += fCandidates[i].size();
    }
  }
};

} // namespace vgbrep
#endif
