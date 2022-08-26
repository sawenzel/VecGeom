#ifndef VECGEOM_SURFACE_NAVIGATOR_H_
#define VECGEOM_SURFACE_NAVIGATOR_H_

#include <VecGeom/surfaces/Model.h>
#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/base/Algorithms.h>

//#define SM_USE_NORMALS

namespace vgbrep {
namespace protonav {

///< Sorts N candidate surfaces in increasing order of the distance from point/direction, starting from
///< a given index. Writes into a sorted array of candidate indices, and into a sorted array of distances.
///< Returns the number of valid sorted candidates.
template <typename Real_t, size_t MAXSIZE>
int SortCandidateDistances(vecgeom::Vector3D<Real_t> const &point, vecgeom::Vector3D<Real_t> const &direction,
                           NavIndex_t in_navind, Real_t dist_min, Candidates const &cand, int startind, int ncand,
                           int skip_surf, SurfData<Real_t> const &surfdata, int *sorted_cand, Real_t *sorted_dist,
                           vecgeom::Vector3D<Real_t> *sorted_onsurf)
{
  Real_t unsorted_dist[MAXSIZE];
  int unsorted_cand[MAXSIZE];
  size_t sorted_ind[MAXSIZE];
  vecgeom::Vector3D<Real_t> unsorted_onsurf[MAXSIZE];
  size_t nsorted = 0;
  Real_t dist;
  for (auto icand = startind; icand < startind + ncand; ++icand) {
    int isurf = std::abs(cand[icand]);
    if (isurf == std::abs(skip_surf)) continue;
    auto const &surf = surfdata.fCommonSurfaces[isurf];
    bool exiting     = surf.fDefaultState != in_navind;
    bool left_side   = cand[icand] > 0;
    bool flip_normal = exiting ^ left_side;
    // Convert point and direction to surface frame
    auto const &trans         = surfdata.fGlobalTrans[surf.fTrans];
    Vector3D<Real_t> local    = trans.Transform(point);
    Vector3D<Real_t> localdir = trans.TransformDirection(direction);
    Vector3D<Real_t> onsurf;
    // Compute distance to surface
    bool surfhit = surfdata.GetUnplaced(isurf).Intersect(local, localdir, exiting, flip_normal, surfdata, dist);
    if (!surfhit) continue;
    if (dist < -vecgeom::kTolerance || dist >= dist_min) continue;
    onsurf = local + dist * localdir;
    // Valid candidate, add it to array to be sorted
    unsorted_dist[nsorted]     = dist;
    unsorted_cand[nsorted]     = icand;
    unsorted_onsurf[nsorted++] = onsurf;
  }
  // Do the sorting by increasing distances
  bool success = vecgeom::algo::quickSort<Real_t, 32>(unsorted_dist, nsorted, sorted_ind);
  if (!success) return -1;

  for (size_t i = 0; i < nsorted; i++) {
    auto ind         = sorted_ind[i];
    sorted_cand[i]   = unsorted_cand[ind];
    sorted_dist[i]   = unsorted_dist[ind];
    sorted_onsurf[i] = unsorted_onsurf[ind];
  }

  return nsorted;
}

///< Sorts N candidate surfaces in increasing order of the distance from point/direction, starting from
///< a given index. Writes into a sorted array of candidate indices, and into a sorted array of distances.
///< Returns the number of valid sorted candidates.
template <typename Real_t, size_t MAXSIZE>
int SortCandidateSafeties(vecgeom::Vector3D<Real_t> const &point, NavIndex_t in_navind, Real_t safe_min,
                          Candidates const &cand, int startind, int ncand, SurfData<Real_t> const &surfdata,
                          int *sorted_cand, Real_t *sorted_dist, vecgeom::Vector3D<Real_t> *sorted_onsurf)
{
  Real_t unsorted_dist[MAXSIZE];
  int unsorted_cand[MAXSIZE];
  size_t sorted_ind[MAXSIZE];
  vecgeom::Vector3D<Real_t> unsorted_onsurf[MAXSIZE];
  size_t nsorted = 0;
  Real_t safe;
  for (auto icand = startind; icand < startind + ncand; ++icand) {
    int isurf        = std::abs(cand[icand]);
    auto const &surf = surfdata.fCommonSurfaces[isurf];
    bool exiting     = surf.fDefaultState != in_navind;
    bool left_side   = cand[icand] > 0;
    bool flip_normal = exiting ^ left_side;
    // Convert point and direction to surface frame
    auto const &trans      = surfdata.fGlobalTrans[surf.fTrans];
    Vector3D<Real_t> local = trans.Transform(point);
    Vector3D<Real_t> onsurf;
    // Compute signed closest distance to surface. The closest projected point on surface is computed, except for:
    // - negative safety (coming from the wrong side)
    // - exiting framed surfaces for which fUseSurfSafety is true
    // To test if on GPU is better to compute the projection systematically
    bool compute_onsurf = true;
    if (exiting) {
      auto const &exit_side = (cand[icand] > 0) ? surf.fLeftSide : surf.fRightSide;
      int frameind          = cand.fFrameInd[icand];
      compute_onsurf        = !exit_side.GetSurface(frameind, surfdata).fUseSurfSafety;
    }
    bool can_compute =
        surfdata.GetUnplaced(isurf).Safety(local, exiting, flip_normal, surfdata, safe, compute_onsurf, onsurf);
    if (!can_compute) continue;
    if (safe < -vecgeom::kTolerance || safe >= safe_min) continue;
    // Valid candidate, add it to array to be sorted
    unsorted_dist[nsorted]     = safe;
    unsorted_cand[nsorted]     = icand;
    unsorted_onsurf[nsorted++] = onsurf;
  }
  // Do the sorting by increasing distances
  bool success = vecgeom::algo::quickSort<Real_t, 32>(unsorted_dist, nsorted, sorted_ind);
  if (!success) return -1;

  for (size_t i = 0; i < nsorted; i++) {
    auto ind         = sorted_ind[i];
    sorted_cand[i]   = unsorted_cand[ind];
    sorted_dist[i]   = unsorted_dist[ind];
    sorted_onsurf[i] = unsorted_onsurf[ind];
  }

  return nsorted;
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

  // The strategy here is: compute distances to all unplaced candidate surfaces, validated by the dot product
  // dir . normal, taking into account that the normal is always defined by the left side. The distance
  // computation is cheaper than the frame Inside search, so we sort all valid candidates by distance
  // first. To avoid usage of dynamic memory, we sort in fixed batches and select the closest surface in
  // each batch that is validated for frame crossing. We keep than the closest valid surface crossing
  // among all batches.
  int startcand              = 0;
  constexpr size_t kMaxStack = 64;
  int sorted_cand[kMaxStack];
  Real_t sorted_dist[kMaxStack];
  Vector3D<Real_t> onsurf_vec[kMaxStack];
  while (startcand < cand.fNcand) {
    // Process a batch of candidates
    int nchecked = std::min(cand.fNcand - startcand, int(kMaxStack));
    auto nsorted =
        SortCandidateDistances<Real_t, kMaxStack>(point, direction, in_navind, distance, cand, startcand, nchecked,
                                                  skip_surf, surfdata, sorted_cand, sorted_dist, onsurf_vec);
    assert(nsorted >= 0);
    bool found     = false;
    bool relocated = false;

    // loop on sorted candidates
    for (auto isorted = 0; isorted < nsorted; ++isorted) {
      auto icand  = sorted_cand[isorted];
      int isurf   = std::abs(cand[icand]);
      Real_t dist = sorted_dist[isorted];
      assert(dist <= distance);
      Vector3D<Real_t> const &onsurf_tmp = onsurf_vec[isorted];
      auto const &surf                   = surfdata.fCommonSurfaces[isurf];
      // Check if the current state is exited on this surface. This is true if
      // the in_state does not match the default state for the surface.
      bool exiting = surf.fDefaultState != in_navind;
      if (exiting) {
        // This is an exiting surface for in_state
        // First check the frame of the current state on this surface
        auto const &exit_side = (cand[icand] > 0) ? surf.fLeftSide : surf.fRightSide;
        // Get the index of the framed surface on the exit side. This assumes a SINGLE frame inprint
        // coming from a touchable on any common surface.
        int frameind           = cand.fFrameInd[icand]; // index of framed surface on the side
        auto const &framedsurf = exit_side.GetSurface(frameind, surfdata);
        // If the current touchable is exited on this surface, it MUST be through the inside of the
        // corresponding frame
        bool inframe = framedsurf.InsideFrame(onsurf_tmp, surfdata);
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
            if (framedsurf.InsideFrame(onsurf_tmp, surfdata)) {
              can_hit = false;
              break; // exiting a daughter volume on this surface, so discard
            }
          }
        }
        if (!can_hit) continue;
        // the current state is correctly exited, so there is a transition on this surface
        found      = true;
        relocated  = false;
        onsurf     = onsurf_tmp;
        distance   = dist;
        isurfcross = cand[icand];
        // backup exited state
        out_state.SetLastExited();
        // the default next navigation index is the one of the common state for the surface
        out_state.SetNavIndex(surf.fDefaultState);
        break; // surface found, finish batch
      }

      // This is an entering surface for in_state

      // First check if there is a parent frame on the entry side. If this is the case
      // and it is missed, then we have a virtual hit so we skip

      // should index sides rather than left/right...
      auto const &entry_side = (cand[icand] > 0) ? surf.fRightSide : surf.fLeftSide;
      // If there is a parent frame on the entry side, it must be hit.
      if (entry_side.fNumParents == 1) {
        int parent_ind         = entry_side.fNsurf - 1;
        auto const &framedsurf = entry_side.GetSurface(parent_ind, surfdata);
        if (framedsurf.InsideFrame(onsurf_tmp, surfdata)) {
          // This surface is certainly hit because the parent frame is hit
          found      = true;
          relocated  = false;
          onsurf     = onsurf_tmp;
          distance   = dist;
          isurfcross = cand[icand];
          // backup exited state
          out_state.SetLastExited();
          // the default next navigation index is the state corresponding to the common parent
          out_state.SetNavIndex(framedsurf.fState);
          break; // surface found, finish batch
        }
        // The parent framed surface is not hit, so skip surface
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
          found      = true;
          relocated  = true;
          onsurf     = onsurf_tmp;
          distance   = dist;
          isurfcross = cand[icand];
          out_state.SetLastExited();
          out_state.SetNavIndex(framedsurf.fState);
          break;
        }
      }

      if (found) break; // surface found, finish batch
    }                   // end loop on sorted candidates

    if (found) {
      exit_surf = isurfcross;
      if (!relocated) {
        // do the relocation on the entering side
        auto const &surf       = surfdata.fCommonSurfaces[std::abs(isurfcross)];
        auto const &entry_side = (isurfcross > 0) ? surf.fRightSide : surf.fLeftSide;
        // Last frame may have been already checked if it is a parent
        int indmax = (entry_side.fNumParents == 1 && surf.fDefaultState == in_navind) ? entry_side.fNsurf - 1
                                                                                      : entry_side.fNsurf;
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

    startcand += nchecked;
  } // end batch processing

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
Real_t ComputeSafety(vecgeom::Vector3D<Real_t> const &point, vecgeom::NavStateIndex const &in_state,
                     SurfData<Real_t> const &surfdata, int &closest_surf)
{
  // Get the list of visible candidate surfaces for in_state
  closest_surf  = 0;
  Real_t safety = vecgeom::InfinityLength<Real_t>();
  Vector3D<Real_t> onsurf;
  NavIndex_t in_navind = in_state.GetNavIndex();
  auto const &cand     = surfdata.fCandidates[in_state.GetId()];

  // The strategy here is: compute safeties to all candidate surfaces, validated by the point being
  // on the correct side (behind normal if exiting, in front of normal if entering). We first sort all
  // valid candidates by the safety to the unplaced surface. Sorting is done in batches to avoid dynamic
  // memory allocation. In case additional frame checking is needed (exiting non-convex volumes or entering)
  // the safety to the corresponding surface frame is computed to get the final safety.
  int startcand              = 0;
  constexpr size_t kMaxStack = 64;
  int sorted_cand[kMaxStack];
  Real_t sorted_dist[kMaxStack];
  Vector3D<Real_t> onsurf_vec[kMaxStack];
  while (startcand < cand.fNcand) {
    // Process a batch of candidates
    int nchecked = std::min(cand.fNcand - startcand, int(kMaxStack));
    auto nsorted = SortCandidateSafeties<Real_t, kMaxStack>(point, in_navind, safety, cand, startcand, nchecked,
                                                            surfdata, sorted_cand, sorted_dist, onsurf_vec);
    assert(nsorted >= 0);

    // loop on sorted candidates
    for (auto isorted = 0; isorted < nsorted; ++isorted) {
      auto icand        = sorted_cand[isorted];
      int isurf         = std::abs(cand[icand]);
      Real_t safetySurf = sorted_dist[isorted];
      if (safetySurf > safety) break;
      Real_t safetyFrame                 = 0;
      Vector3D<Real_t> const &onsurf_tmp = onsurf_vec[isorted];
      auto const &surf                   = surfdata.fCommonSurfaces[isurf];
      // Check if the current state is exited on this surface. This is true if
      // the in_state does not match the default state for the surface.
      bool exiting = surf.fDefaultState != in_navind;
      if (exiting) {
        // This is an exiting surface for in_state
        // Only check the frame of the current state on this surface
        auto const &exit_side = (cand[icand] > 0) ? surf.fLeftSide : surf.fRightSide;
        // Get the index of the framed surface on the exit side. This assumes a SINGLE frame inprint
        // coming from a touchable on any common surface.
        int frameind           = cand.fFrameInd[icand]; // index of framed surface on the side
        auto const &framedsurf = exit_side.GetSurface(frameind, surfdata);
        // Check if the exited frame safety is needed at all
        bool use_surf_safety = exit_side.GetSurface(frameind, surfdata).fUseSurfSafety;
        if (!use_surf_safety) {
          // We need to compute also the safety of the projection of the point on surface to the frame
          safetyFrame = framedsurf.SafetyFrame(onsurf_tmp, surfdata);
        }
        Real_t safetySq = safetySurf * safetySurf + safetyFrame * safetyFrame;
        if (safety * safety > safetySq) {
          safety       = std::sqrt(safetySq);
          closest_surf = isurf;
        }
      } else {
        // Entering side. We only check the parent frames on the side
        auto const &entry_side = (cand[icand] > 0) ? surf.fRightSide : surf.fLeftSide;
        const int num_parents  = entry_side.fNumParents;
        int iparent            = 0;
        Real_t safetyParent    = vecgeom::InfinityLength<Real_t>();
        // Parent frames are last in the list
        for (auto ind = entry_side.fNsurf - 1; ind >= 0; --ind) {
          auto const &framedsurf = entry_side.GetSurface(ind, surfdata);
          if (framedsurf.fParent >= 0) continue; // skip children
          iparent++;
          auto safetyFrame = framedsurf.SafetyFrame(onsurf_tmp, surfdata);
          if (safetyFrame < safetyParent) safetyParent = safetyFrame;
          if (iparent == num_parents) break;
        }
        Real_t safetySq = safetySurf * safetySurf + safetyParent * safetyParent;
        if (safety * safety > safetySq) {
          safety       = std::sqrt(safetySq);
          closest_surf = isurf;
        }
      }
    }

    startcand += nchecked;
  } // end batch processing

  return safety;
}

} // namespace protonav
} // namespace vgbrep
#endif
