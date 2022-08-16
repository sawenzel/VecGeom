#ifndef VECGEOM_SURFACE_NAVIGATOR_H_
#define VECGEOM_SURFACE_NAVIGATOR_H_

#include <VecGeom/surfaces/Model.h>
#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/base/Algorithms.h>

namespace vgbrep {
namespace protonav {

///< Sorts N candidate surfaces in increasing order of the distance from point/direction, starting from
///< a given index. Writes into a sorted array of candidate indices, and into a sorted array of distances.
///< Returns the last sorted index.
template <typename Real_t>
int SortDistances(vecgeom::Vector3D<Real_t> const &point, vecgeom::Vector3D<Real_t> const &direction, int ncand,
                  int *candidates, int startind, int maxslots, int skip_surf, bool is_entering,
                  SurfData<Real_t> const &surfdata, int *survivors, int *sorted_ind, Real_t *distances)
{
  int nsorted = 0;
  for (auto icand = startind; icand < ncand; ++icand) {
    int isurf = std::abs(candidates[icand]);
    if (std::abs(isurf) == std::abs(skip_surf)) continue;
    auto const &surf = surfdata.fCommonSurfaces[isurf];
    // Convert point and direction to surface frame
    auto const &trans         = surfdata.fGlobalTrans[surf.fTrans];
    Vector3D<Real_t> local    = trans.Transform(point);
    Vector3D<Real_t> localdir = trans.TransformDirection(direction);
    // Compute distance to surface
    auto dist = surfdata.GetUnplaced(isurf).Intersect(local, localdir, surfdata);
    if (dist < vecgeom::kTolerance) continue;
    // Check the normal
    Vector3D<Real_t> normal;
    Vector3D<Real_t> onsurf_tmp = local + dist * localdir;
    surf.GetNormal(onsurf_tmp, normal, surfdata);
    if ((normal.Dot(direction) > 0) ^ is_entering) continue;
    // Valid candidate, add it to array to be sorted
    distances[nsorted]   = dist;
    survivors[nsorted++] = icand;
  }
  // Do the sorting by increasing distances
  bool success = vecgeom::algo::quickSort<Real_t, 32>(distances, nsorted, sorted_ind);

  return 0;
}

///< Returns the distance to the next surface starting from a global point located in in_state, optionally
///< skipping the surface exit_surf. Computes the new state after crossing.
template <typename Real_t>
Real_t ComputeStepAndHit(vecgeom::Vector3D<Real_t> const &point, vecgeom::Vector3D<Real_t> const &direction,
                         vecgeom::NavStateIndex const &in_state, vecgeom::NavStateIndex &out_state,
                         SurfData<Real_t> const &surfdata, int &exit_surf)
{
  //#define SM_USE_NORMALS
  // Get the list of candidate surfaces for in_state
  out_state         = in_state;
  int current_level = in_state.GetLevel();
  auto skip_surf    = exit_surf;
  exit_surf         = 0;
  Real_t distance   = vecgeom::InfinityLength<Real_t>();
  int isurfcross    = 0;
  bool relocated    = false;
  Vector3D<Real_t> onsurf;
  NavIndex_t in_navind = in_state.GetNavIndex();
  auto const &cand     = surfdata.fCandidates[in_state.GetId()];
  int numroots         = 0; // < number of real, positive distances
  Real_t roots[2];          // < distances to intersection
  // simple loop on all candidates. This should be in future optimized to give the reduced list that may be crossed
  for (auto icand = 0; icand < cand.fNcand; ++icand) {
    bool can_hit = true;
    int isurf    = std::abs(cand[icand]);
    if (std::abs(isurf) == std::abs(skip_surf)) continue;
    auto const &surf = surfdata.fCommonSurfaces[isurf];
    // Convert point and direction to surface frame
    vecgeom::Transformation3D const &trans = surfdata.fGlobalTrans[surf.fTrans];
    Vector3D<Real_t> local                 = trans.Transform(point);
    Vector3D<Real_t> localdir              = trans.TransformDirection(direction);
    // Compute distance to surface
    surfdata.GetUnplaced(isurf).Intersect(local, localdir, surfdata, roots, numroots);
    if (numroots <= 0) continue;
    // Check all distances that were returned. We're interested in the minimal
    // positive value
    for (auto i = 0; i < numroots; ++i) {
      auto dist = roots[i];
      if (dist < vecgeom::kTolerance || dist > distance) continue;
      Vector3D<Real_t> onsurf_tmp = local + dist * localdir;

      // should index sides rather than left/right...
      auto const &exit_side  = (cand[icand] > 0) ? surf.fLeftSide : surf.fRightSide;
      auto const &entry_side = (cand[icand] > 0) ? surf.fRightSide : surf.fLeftSide;

      // Check if the current state is exited on this surface. This is true if
      // the in_state does not match the default state for the surface.
      bool exiting = surf.fDefaultState != in_navind;

#ifdef SM_USE_NORMALS
      Vector3D<Real_t> normal;
      bool left_side = exiting ^ (cand[icand] < 0);

      surf.GetNormal(onsurf_tmp, normal, surfdata, left_side);
      if ((normal.Dot(direction) > 0) ^ exiting) continue;
#endif

      if (exiting) {
        // This is an exiting surface for in_state
        // First check the frame of the current state on this surface
        int frameind           = cand.fFrameInd[icand]; // index of framed surface on the side
        auto const &framedsurf = exit_side.GetSurface(frameind, surfdata);
        bool inframe           = framedsurf.InsideFrame(onsurf_tmp, surfdata);
        if (!inframe) continue;

        // frames of daughters of the current state on the same surface must NOT be crossed
        // Daughters may be found only at indices lesser than frameind
        for (auto ind = 0; ind < frameind; ++ind) {
          auto const &framedsurf = exit_side.GetSurface(ind, surfdata);
          // Only search navigation levels higher than the current one
          if (vecgeom::NavStateIndex::GetLevelImpl(framedsurf.fState) >= current_level) break;
          if (vecgeom::NavStateIndex::IsDescendentImpl(framedsurf.fState, in_navind)) {
            if (framedsurf.InsideFrame(onsurf_tmp, surfdata)) {
              can_hit = false;
              break; // next candidate
            }
          }
        }
        if (!can_hit) continue;
        // the current state is correctly exited, so there is a transition on this surface
        relocated  = false;
        onsurf     = onsurf_tmp;
        distance   = dist;
        isurfcross = cand[icand];
        // backup exited state
        out_state.SetLastExited();
        // the default next navigation index is the one of the common state for the surface
        out_state.SetNavIndex(surf.fDefaultState);
        continue;
      }

      // Now check if something is being entered on the other side
      // First check if there is a parent frame on the entry side. If this is not the case
      // we have a virtual hit so we skip
      if (entry_side.fParentSurf >= 0) {
        auto const &framedsurf = entry_side.GetSurface(entry_side.fParentSurf, surfdata);
        if (framedsurf.InsideFrame(onsurf_tmp, surfdata)) {
          // This surface is certainly hit because the parent frame is hit
          relocated  = false;
          onsurf     = onsurf_tmp;
          distance   = dist;
          isurfcross = cand[icand];
          // backup exited state
          out_state.SetLastExited();
          // the default next navigation index is the state corresponding to the common parent
          out_state.SetNavIndex(framedsurf.fState);
        }
        // Even if the surface is hit, we don't want to do the relocation before checking all candidates
        continue;
      }

      // There is no parent for the entry side.
      // first check the extent of the entry side using onsurf
      if (!entry_side.fExtent.Inside(onsurf_tmp, surfdata)) continue;

      // the onsurf_tmp local point can be used as input for a side search optimization structure.
      // for now just loop candidates in order. Since candidates are sorted by depth, the first
      // frame entry is the good one.
      for (auto ind = 0; ind < entry_side.fNsurf; ++ind) {
        auto const &framedsurf = entry_side.GetSurface(ind, surfdata);
        bool inframe           = framedsurf.InsideFrame(onsurf_tmp, surfdata);
        if (inframe) {
          // the first hit frame is the good one. This worth as a relocation after crossing.
          relocated  = true;
          onsurf     = onsurf_tmp;
          distance   = dist;
          isurfcross = cand[icand];
          out_state.SetLastExited();
          out_state.SetNavIndex(framedsurf.fState);
          break;
        }
      }
    } // end loop on roots
  }   // end loop on candidates

  assert(isurfcross != 0); // something MUST be hit (since no step limitation at this point)
  exit_surf = isurfcross;
  // Now perform relocation after crossing if not yet done
  if (!relocated) {
    auto const &surf       = surfdata.fCommonSurfaces[std::abs(isurfcross)];
    auto const &entry_side = (isurfcross > 0) ? surf.fRightSide : surf.fLeftSide;
    // Last frame may have been already checked if it is a parent
    int indmax =
        (entry_side.fParentSurf >= 0 && surf.fDefaultState == in_navind) ? entry_side.fNsurf - 1 : entry_side.fNsurf;
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
  }

  return distance;
}

} // namespace protonav
} // namespace vgbrep
#endif
