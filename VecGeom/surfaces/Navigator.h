#ifndef VECGEOM_SURFACE_NAVIGATOR_H_
#define VECGEOM_SURFACE_NAVIGATOR_H_

#include <VecGeom/surfaces/Model.h>
//#include <VecGeom/surfaces/orange/LogicStack.hh>
#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/base/Algorithms.h>

namespace vgbrep {
namespace protonav {

/// @brief Locate a point in a volume sub-hierarchy
/// @tparam Real_t Floating point type
/// @param vol Volume to start checking from
/// @param point Point in local volume coordinates
/// @param path Path pointing to the top volume to check
/// @param top The top volume must be checked also
/// @param exclude Placed volume to exclude from checking
/// @return
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE
vecgeom::VPlacedVolume const *LocatePointIn(vecgeom::VPlacedVolume const *vol, vecgeom::Vector3D<Real_t> const &point,
                                            vecgeom::NavStateIndex &path, SurfData<Real_t> const &surfdata, bool top,
                                            vecgeom::VPlacedVolume *exclude = nullptr)
{
  using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const *;
  using Vector3D           = vecgeom::Vector3D<Real_t>;

  if (top) {
    assert(vol != nullptr);
    auto volShell = surfdata.fShells[vol->GetLogicalVolume()->id()];
    if (!volShell.Inside(point, surfdata)) return nullptr;
  }

  VPlacedVolumePtr_t currentvolume = vol;
  Vector3D currentpoint(point);
  path.Push(currentvolume);

  bool godeeper;
  do {
    godeeper = false;
    for (auto *daughter : currentvolume->GetDaughters()) {
      if (daughter == exclude) {
        continue;
      }
      Vector3D localpoint;
      Transformation trans;
      path.Push(daughter);
      path.TopMatrix(trans);
      trans.Transform(currentpoint, localpoint);
      auto volShell = surfdata.fShells[daughter->GetLogicalVolume()->id()];
      if (volShell.Inside(localpoint, surfdata)) {
        currentvolume = daughter;
        godeeper      = true;
        break;
      } else {
        path.Pop();
      }
    }

    // Only exclude the placed volume once since we could enter it again via a
    // different volume history.
    exclude = nullptr;
  } while (godeeper);

  return currentvolume;
}

/// @brief Method computing the distance to the next surface and state after crossing it
/// @tparam Real_t Floating point type for the interface and data storage
/// @param point Global point
/// @param direction Global direction
/// @param in_state Input navigation state before crossing
/// @param out_state Output navigation state after crossing
/// @param surfdata Surface data storage
/// @param exit_surf Input: surface to be skipped, output: crossed surface index
/// @return Distance to next surface
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE
Real_t ComputeStepAndHit(vecgeom::Vector3D<Real_t> const &point, vecgeom::Vector3D<Real_t> const &direction,
                         vecgeom::NavStateIndex const &in_state, vecgeom::NavStateIndex &out_state,
                         SurfData<Real_t> const &surfdata, int &exit_surf)
{
  // Get the list of candidate surfaces for in_state
  out_state         = in_state;
  int current_level = in_state.GetLevel();
  auto skip_surf    = exit_surf;
  exit_surf         = 0;
  Real_t distance   = vecgeom::InfinityLength<Real_t>();
  int isurfcross    = 0;
  Vector3D<Real_t> onsurf;
  NavIndex_t in_navind = in_state.GetNavIndex();
  auto const &cand     = surfdata.fCandidates[in_state.GetId()];
  bool found           = false;
  bool relocated       = false;

  for (auto icand = 0; icand < cand.fNcand; ++icand) {
    int isurf = std::abs(cand[icand]);
    if (isurf == std::abs(skip_surf)) continue;
    auto const &surf = surfdata.fCommonSurfaces[isurf];
    // Check if this is an exiting or entering surface for the current navigation state
    // Get the side of the surface and check if the surface normal needs to be flipped
    bool exiting   = surf.fDefaultState != in_navind;
    bool left_side = cand[icand] > 0;
    // Convert point and direction to surface frame
    auto const &trans         = surfdata.fGlobalTrans[surf.fTrans];
    Vector3D<Real_t> local    = trans.Transform(point);
    Vector3D<Real_t> localdir = trans.TransformDirection(direction);
    // Compute distance to surface
    Real_t dist;
    bool surfhit = surfdata.GetUnplaced(isurf).Intersect(local, localdir, left_side, surfdata, dist);
    if (!surfhit || dist < -vecgeom::kTolerance || dist >= distance) continue;
    Vector3D<Real_t> onsurf_crt = local + dist * localdir;

    // We need to check frame intersection
    if (exiting) {
      // This is an exiting surface for in_state
      // First check the frame of the current state on this surface
      auto const &exit_side = left_side ? surf.fLeftSide : surf.fRightSide;
      // Get the index of the framed surface on the exit side. This assumes a SINGLE frame inprint
      // coming from a touchable on any common surface.
      int frameind           = cand.fFrameInd[icand]; // index of framed surface on the side
      auto const &framedsurf = exit_side.GetSurface(frameind, surfdata);
      // If the current touchable is exited on this surface, it MUST be through the inside of the
      // corresponding frame
      bool inframe = framedsurf.InsideFrame(onsurf_crt, surfdata);
      if (!inframe) continue;

      // If the current touchable has children with surfaces on the same common surface, their
      // frames must NOT be crossed (otherwise there must be another surface cross at smaller distance)
      // Daughters may only be found at indices less than `frameind` (because of pre-sorting by depth)
      bool can_hit = true;
      for (auto ind = 0; ind < frameind; ++ind) {
        auto const &framedsurf = exit_side.GetSurface(ind, surfdata);
        // Only search navigation levels higher than the current one
        if (vecgeom::NavStateIndex::GetLevelImpl(framedsurf.fState) >= current_level) break;
        if (vecgeom::NavStateIndex::IsDescendentImpl(framedsurf.fState, in_navind)) {
          if (framedsurf.InsideFrame(onsurf_crt, surfdata)) {
            can_hit = false;
            break; // exiting a daughter volume on this surface, so discard
          }
        }
      }
      if (!can_hit) continue;
      // the current state is correctly exited, so there is a transition on this surface
      found      = true;
      relocated  = false;
      onsurf     = onsurf_crt;
      distance   = dist;
      isurfcross = cand[icand];
      // backup exited state
      out_state.SetLastExited();
      // the default next navigation index is the one of the common state for the surface
      out_state.SetNavIndex(surf.fDefaultState);
      continue; // there may be closer surfaces being crossed
    }

    // This is an entering surface for in_state
    // First check if there is a parent frame on the entry side. If this is the case
    // and it is missed, then we have a virtual hit so we skip

    // should index sides rather than left/right...
    auto const &entry_side = left_side ? surf.fRightSide : surf.fLeftSide;
    // If there is a parent frame on the entry side, it must be hit.
    if (entry_side.fNumParents == 1) {
      int parent_ind         = entry_side.fNsurf - 1;
      auto const &framedsurf = entry_side.GetSurface(parent_ind, surfdata);
      if (framedsurf.InsideFrame(onsurf_crt, surfdata)) {
        // This surface is certainly hit because the parent frame is hit
        found      = true;
        relocated  = false;
        onsurf     = onsurf_crt;
        distance   = dist;
        isurfcross = cand[icand];
        // backup exited state
        out_state.SetLastExited();
        // the default next navigation index is the state corresponding to the common parent
        out_state.SetNavIndex(framedsurf.fState);
      }
      continue;
    }
    // There is no parent for the entry side.
    // first check the extent of the entry side using onsurf
    if (!entry_side.fExtent.Inside(onsurf_crt, surfdata)) continue;

    // the onsurf_tmp local point can be used as input for a side search optimization structure.
    // for now just loop candidates in order. Since candidates are sorted by depth, the first
    // frame entry is the good one.
    for (auto ind = 0; ind < entry_side.fNsurf; ++ind) {
      auto const &framedsurf = entry_side.GetSurface(ind, surfdata);
      bool inframe           = framedsurf.InsideFrame(onsurf_crt, surfdata);
      if (inframe) {
        // the first hit frame is the good one. This worth as a relocation after crossing.
        found      = true;
        relocated  = true;
        onsurf     = onsurf_crt;
        distance   = dist;
        isurfcross = cand[icand];
        out_state.SetLastExited();
        out_state.SetNavIndex(framedsurf.fState);
        break;
      }
    }
    continue; // check next
  }

  if (found) {
    exit_surf = isurfcross;
    if (!relocated) {
      // do the relocation on the entering side
      auto const &surf       = surfdata.fCommonSurfaces[std::abs(isurfcross)];
      auto const &entry_side = (isurfcross > 0) ? surf.fRightSide : surf.fLeftSide;
      // Last frame may have been already checked if it is a parent
      int indmax =
          (entry_side.fNumParents == 1 && surf.fDefaultState == in_navind) ? entry_side.fNsurf - 1 : entry_side.fNsurf;
      for (auto ind = 0; ind < indmax; ++ind) {
        auto const &framedsurf = entry_side.GetSurface(ind, surfdata);
        bool inframe           = framedsurf.InsideFrame(onsurf, surfdata);
        if (inframe) {
          // the first hit frame is the good one.
          out_state.SetLastExited();
          out_state.SetNavIndex(framedsurf.fState);
          break;
        }
      }
    } // end relocation
  }
  return distance;
}

/// @brief Method computing the distance to the next surface and state after crossing it
/// @tparam Real_t Floating point type for the interface and data storage
/// @param point Global point
/// @param direction Global direction
/// @param in_state Input navigation state before crossing
/// @param out_state Output navigation state after crossing
/// @param surfdata Surface data storage
/// @param exit_surf Input: surface to be skipped, output: crossed surface index
/// @return Distance to next surface
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE
Real_t ComputeSafety(vecgeom::Vector3D<Real_t> const &point, vecgeom::NavStateIndex const &in_state,
                     SurfData<Real_t> const &surfdata, int &closest_surf)
{
  // Get the list of visible candidate surfaces for in_state
  closest_surf  = 0;
  Real_t safety = vecgeom::InfinityLength<Real_t>();
  Vector3D<Real_t> onsurf;
  NavIndex_t in_navind = in_state.GetNavIndex();
  auto const &cand     = surfdata.fCandidates[in_state.GetId()];

  // loop all visible candidates
  for (auto icand = 0; icand < cand.fNcand; ++icand) {
    bool validSafety      = true;
    int isurf             = std::abs(cand[icand]);
    auto const &surf      = surfdata.fCommonSurfaces[isurf];
    bool exiting          = surf.fDefaultState != in_navind;
    bool left_side        = cand[icand] > 0;
    auto const &exit_side = left_side ? surf.fLeftSide : surf.fRightSide;
    // Convert point and direction to surface frame
    auto const &trans      = surfdata.fGlobalTrans[surf.fTrans];
    Vector3D<Real_t> local = trans.Transform(point);
    Vector3D<Real_t> onsurf_crt;
    Real_t safety_surf;
    // Compute signed closest distance to surface. The closest projected point on surface is computed, except for:
    // - negative safety (coming from the wrong side)
    // - exiting framed surfaces for which fUseSurfSafety is true
    // To test if on GPU is better to compute the projection systematically
    bool compute_onsurf = exiting ? !exit_side.GetSurface(cand.fFrameInd[icand], surfdata).fUseSurfSafety : true;
    bool can_compute =
        surfdata.GetUnplaced(isurf).Safety(local, left_side, surfdata, safety_surf, compute_onsurf, onsurf_crt);
    if (!can_compute || safety_surf < -vecgeom::kTolerance || safety_surf >= safety) continue;
    // Check if the current state is exited on this surface. This is true if
    // the in_state does not match the default state for the surface.
    if (exiting) {
      // This is an exiting surface for in_state
      // Only check the frame of the current state on this surface
      auto const &exit_side = (cand[icand] > 0) ? surf.fLeftSide : surf.fRightSide;
      // Get the index of the framed surface on the exit side. This assumes a SINGLE frame inprint
      // coming from a touchable on any common surface.
      int frameind           = cand.fFrameInd[icand]; // index of framed surface on the side
      auto const &framedsurf = exit_side.GetSurface(frameind, surfdata);
      // Check if the exited frame safety is needed at all
      auto safetyFrame     = safety_surf;
      bool use_surf_safety = exit_side.GetSurface(frameind, surfdata).fUseSurfSafety;
      if (!use_surf_safety) {
        // We need to compute also the safety of the projection of the point on surface to the frame
        safetyFrame = framedsurf.SafetyFrame(onsurf_crt, safety_surf, surfdata, validSafety);
      }
      if (validSafety && safety > safetyFrame) {
        safety       = safetyFrame;
        closest_surf = isurf;
      }
    } else {
      // Entering side. We only check the parent frames on the side
      auto const &entry_side = left_side ? surf.fRightSide : surf.fLeftSide;
      const int num_parents  = entry_side.fNumParents;
      int iparent            = 0;
      Real_t safetyParent    = vecgeom::InfinityLength<Real_t>();
      // Parent frames are last in the list
      for (auto ind = entry_side.fNsurf - 1; ind >= 0; --ind) {
        auto const &framedsurf = entry_side.GetSurface(ind, surfdata);
        if (framedsurf.fParent >= 0) continue; // skip children
        iparent++;
        auto safetyFrame = framedsurf.SafetyFrame(onsurf_crt, safety_surf, surfdata, validSafety);
        if (validSafety && safetyFrame < safetyParent) safetyParent = safetyFrame;
        if (iparent == num_parents) break;
      }
      if (safety > safetyParent) {
        safety       = safetyParent;
        closest_surf = isurf;
      }
    }
  }
  return safety;
}

} // namespace protonav
} // namespace vgbrep
#endif
