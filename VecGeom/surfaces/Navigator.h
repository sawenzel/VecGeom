#ifndef VECGEOM_SURFACE_NAVIGATOR_H_
#define VECGEOM_SURFACE_NAVIGATOR_H_

#include <VecGeom/management/Logger.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/LogicEvaluator.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/base/Algorithms.h>

#include <iomanip>

namespace vgbrep {
namespace protonav {

/// @brief Check the Inside for the VolumeShell object in local coordinates
/// @param point Point in local volume coordinates
/// @param volId Logical volume id
/// @param surfdata Surface data storage
/// @param logic_id Logical id entering/exiting surface for which the logic is known
/// @param is_inside whether the known entering/exiting surface is inside or not
/// @return Boolean value representing if the point is inside the VolumeShell
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE bool LogicInsideLocal(vecgeom::Vector3D<Real_t> const &localpoint,
                                                                   int volId, SurfData<Real_t> const &surfdata,
                                                                   const int logic_id = -1, const bool is_inside = 0)
{
  auto const &logic = surfdata.fShells[volId].fLogic;
  // Evaluate volume shell logic
  auto inside = EvaluateInside(localpoint, volId, logic, surfdata, logic_id, is_inside);
  return inside;
}

/// @brief Check the Inside for the VolumeShell object associated with a touchable
/// @param point Point in global coordinates
/// @param in_state Navigation state associated with the touchable
/// @param logic_id Logical id of entering/exiting surface for which the logic is known
/// @param is_inside whether the known entering/exiting surface is inside or not
/// @return Boolean value representing if the point is inside the VolumeShell
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE bool LogicInside(vecgeom::Vector3D<Real_t> const &point,
                                                              vecgeom::NavigationState const &in_state,
                                                              SurfData<Real_t> const &surfdata, const int logic_id = -1,
                                                              const bool is_inside = 0)
{
  // Convert point in local VolumeShell coordinates
  Vector3D<Real_t> localpoint;
  Transformation trans;
  in_state.TopMatrix(trans);
  trans.Transform(point, localpoint);
  auto vol    = in_state.Top();
  auto volId  = vol->GetLogicalVolume()->id();
  auto inside = LogicInsideLocal(localpoint, volId, surfdata, logic_id, is_inside);
  return inside;
}

/// @brief Lambda for checking if any frame on a side of common surface side is hit
/// @param isurf common surface index
/// @param left_side Left side
/// @param in_state Navigation state associated with the touchable
/// @param in_navind State index from which the surface is entered
/// @param to_be_checked Which frames to check: 0 = all, 1 = daughters only 2 = parents only
/// @param distance Distance to surface
/// @param point Global point
/// @param direction Global direction
/// @param onsurf_local Propagated point on the checked common surface, in its local reference frame
/// @param surfdata Surface data storage
/// @return Crossed frame index. Negative if none crossed.
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE int CheckFramesEntering(int isurf, bool left_side, vecgeom::NavigationState const &in_state,
                                                NavIndex_t in_navind, int to_be_checked, Real_t distance,
                                                Vector3D<Real_t> const &point, Vector3D<Real_t> const &direction,
                                                Vector3D<Real_t> const &onsurf_local, SurfData<Real_t> const &surfdata)
{
  constexpr char kCheckChildren  = 1;
  constexpr char kCheckParents   = 2;
  constexpr Real_t kPushDistance = 1000 * vecgeom::kToleranceDist<Real_t>; // tolerance for Boolean push
  auto const &common_surface     = surfdata.fCommonSurfaces[isurf];
  auto const &surface_side       = left_side ? common_surface.fLeftSide : common_surface.fRightSide;
  auto start_ind = (to_be_checked != kCheckParents) ? 0 : surface_side.fNsurf - surface_side.fNumParents;
  auto last_ind =
      (to_be_checked == kCheckChildren) ? surface_side.fNsurf - surface_side.fNumParents : surface_side.fNsurf;
  for (auto ind = start_ind; ind < last_ind; ++ind) {
    auto const &framedsurf = surface_side.GetSurface(ind, surfdata);
    if (framedsurf.fNeverCheck) return ind;
    // If this frame has the same state as the exited state (this can happen in Booleans
    // having internal surfaces), it means that the current touchable has an internal common
    // surface being crossed, so this surface must be ignored
    if (framedsurf.fState == in_navind) continue;
    auto inframe = framedsurf.InsideFrame(onsurf_local, surfdata);
    if (!inframe) continue;
    if (framedsurf.fLogicId) {
      auto pushedPoint        = point + (distance + kPushDistance) * direction;
      auto checked_state      = in_state;
      unsigned short scene_id = 0, newscene_id = 0;
      in_state.GetSceneId(scene_id, newscene_id);
      if (common_surface.fSceneId > scene_id)
        checked_state.PushScene(framedsurf.fState);
      else
        checked_state.SetNavIndex(framedsurf.fState);
      // The logic for the frame that is entered can be set to true without a numerical check
      auto inside = LogicInside(pushedPoint, checked_state, surfdata, framedsurf.fLogicId, true);
      // Frame cross does not guarantee a real surface cross in case of Booleans
      // For a real exiting, the post-crossing point must be inside the Boolean
      if (!inside) continue;
    }
    return ind;
  }
  return -1;
}

/// @brief Computes isotropic safety for the logic expression of a volume
/// @param point Point in global coordinates
/// @param in_state
/// @param surfdata
/// @return isotropic safety value
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Real_t LogicSafety(vecgeom::Vector3D<Real_t> const &point, bool exiting,
                                                                vecgeom::NavigationState const &in_state,
                                                                SurfData<Real_t> const &surfdata,
                                                                Real_t safe_max = vecgeom::InfinityLength<Real_t>())
{
  // Convert point in local VolumeShell coordinates
  Vector3D<Real_t> localpoint;
  Transformation trans;
  in_state.TopMatrix(trans);
  trans.Transform(point, localpoint);
  auto vol          = in_state.Top();
  auto volId        = vol->GetLogicalVolume()->id();
  auto const &logic = surfdata.fShells[volId].fLogic;
  Real_t safety     = EvaluateSafety(localpoint, volId, exiting, logic, surfdata, safe_max);
  return safety;
}

/// @brief Locate a point in a volume sub-hierarchy
/// @tparam Real_t Floating point type
/// @param vol Volume to start checking from
/// @param point Point in local volume coordinates
/// @param path Path pointing to the top volume to check
/// @param surfdata Surface data storage
/// @param top The top volume must be checked also
/// @param exclude Placed volume to exclude from checking
/// @return Placed volume pointer containing the point
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE vecgeom::VPlacedVolume const *LocatePointIn(vecgeom::VPlacedVolume const *vol,
                                                                    vecgeom::Vector3D<Real_t> const &point,
                                                                    vecgeom::NavigationState &path, bool top,
                                                                    vecgeom::VPlacedVolume *exclude = nullptr)
{
  using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const *;
  auto const &surfdata     = SurfData<Real_t>::Instance();

  if (top) {
    assert(vol != nullptr);
    auto inside = LogicInsideLocal(point, vol->GetLogicalVolume()->id(), surfdata);
    if (!inside) return nullptr;
  }

  VPlacedVolumePtr_t currentvolume = vol;
  path.Push(currentvolume);

  bool godeeper;
  do {
    godeeper = false;
    for (auto *daughter : currentvolume->GetDaughters()) {
      if (daughter == exclude) {
        continue;
      }
      path.Push(daughter);
      auto inside = LogicInside(point, path, surfdata);

      if (inside) {
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
VECCORE_ATT_HOST_DEVICE Real_t ComputeStepAndHit(vecgeom::Vector3D<Real_t> const &point,
                                                 vecgeom::Vector3D<Real_t> const &direction,
                                                 vecgeom::NavigationState const &in_state,
                                                 vecgeom::NavigationState &out_state, int &exit_surf,
                                                 Real_t stepmax = vecgeom::InfinityLength<Real_t>())
{
  constexpr char kLside         = 1;
  constexpr char kRside         = 2;
  constexpr char kCheckAll      = 0;
  constexpr char kCheckChildren = 1;
  constexpr char kCheckParents  = 2;
  // Get the list of candidate surfaces for in_state
  auto const &surfdata    = SurfData<Real_t>::Instance();
  Real_t distance         = stepmax;
  int isurfcross          = 0;
  auto in_navind          = in_state.GetNavIndex();
  unsigned short scene_id = 0, newscene_id = 0;
  bool is_scene     = in_state.GetSceneId(scene_id, newscene_id);
  auto const &cand  = surfdata.GetCandidates(scene_id, in_state.GetId());
  auto new_cand_ptr = &cand;
  if (is_scene) new_cand_ptr = &surfdata.GetCandidates(newscene_id, 0);
  auto const &new_cand     = *new_cand_ptr;
  bool found               = false;
  bool exiting             = false;
  bool relocated           = false;
  bool relocated_left_side = false;
  bool recompute_onsurf    = false;

  constexpr Real_t kPushDistance = 1000 * vecgeom::kToleranceDist<Real_t>;
  Vector3D<Real_t> onsurf;
  out_state = in_state;
  out_state.SetBoundaryState(false);
  auto skip_surf = exit_surf;
  exit_surf      = 0;

  // Convert the point and direction to the scene coordinate system
  vecgeom::Transformation3D scene_trans;
  in_state.SceneMatrix(scene_trans);
  Vector3D<Real_t> local_scene    = scene_trans.Transform(point);
  Vector3D<Real_t> localdir_scene = scene_trans.TransformDirection(direction);

  // First check Exiting candidates
  for (auto icand = 0; icand < cand.fNExiting; ++icand) {
    int isurf = cand.fCandidates[icand];
    if (isurf == 0 || isurf == skip_surf) continue;
    auto const &surf = surfdata.fCommonSurfaces[isurf];
    char sides       = cand.fSides[icand];

    // Convert point and direction to surface frame
    auto const &trans         = surfdata.fGlobalTrans[surf.fTrans];
    Vector3D<Real_t> local    = trans.Transform(local_scene);
    Vector3D<Real_t> localdir = trans.TransformDirection(localdir_scene);

    // Compute distance to surface
    Real_t dist;
    bool flipped  = false;
    auto unplaced = surfdata.GetUnplaced(isurf, flipped);

    bool left_side        = (sides & kLside) > 0;
    bool right_side       = (sides & kRside) > 0;
    bool check_both_sides = left_side && right_side;
    bool visibility       = left_side ^ flipped;
    bool surfhit          = unplaced.Intersect(local, localdir, visibility, surfdata, dist);
    if (!surfhit && check_both_sides) {
      // Left side already checked, now check right side
      // Note: only one side can have a valid exiting
      left_side  = false;
      visibility = flipped;
      surfhit    = unplaced.Intersect(local, localdir, visibility, surfdata, dist);
    }
    if (!surfhit || dist < -vecgeom::kTolerance || dist >= distance) continue;
#if SURF_NAV_DEBUG > 0
    std::cout << " to out -> surface " << isurf << " hit at dist = " << dist << " -> ";
#endif
    Vector3D<Real_t> onsurf_crt = local + dist * localdir;

    // We need to check frame intersection
    // This is an exiting surface for in_state
    // First check the frame of the current state on this surface
    auto const &exit_side = left_side ? surf.fLeftSide : surf.fRightSide;
    // Get the index of the first framed surface on the exit side.
    int frameind_start = cand.fFrameInd[icand];
    // If the current touchable is exited on this surface, it MUST be through the inside of the
    // corresponding frames. Loop all frames coming from the same touchable
    bool inframe   = false;
    int surf_index = 0;
    for (int ind = frameind_start; ind < exit_side.fNsurf; ++ind) {
      auto const &framedsurf = exit_side.GetSurface(ind, surfdata);
      if (framedsurf.fState != in_navind) continue;
      inframe = framedsurf.fNeverCheck ? true : framedsurf.InsideFrame(onsurf_crt, surfdata);
      if (inframe) {
        // Find the appropriate surface index
        surf_index       = framedsurf.fSurfIndex;
        int parent_frame = framedsurf.fParent;
        while (parent_frame > 0) {
          surf_index   = exit_side.GetSurface(parent_frame, surfdata).fSurfIndex;
          parent_frame = exit_side.GetSurface(parent_frame, surfdata).fParent;
        }
        if (framedsurf.fLogicId) {
          auto pushedPoint = point + (dist + kPushDistance) * direction;
          // The logic for the frame that is exited can be set to false without a numerical check
          auto inside = LogicInside(pushedPoint, in_state, surfdata, framedsurf.fLogicId, false);
#if SURF_NAV_DEBUG > 0
          if (inside) std::cout << " logic exiting still inside -> ";
#endif
          // Frame cross does not guarantee a real surface cross in case of Booleans
          // For a real exiting, the post-crossing point must be outside the Boolean
          if (inside) inframe = false;
        }
        break;
      }
    }
#if SURF_NAV_DEBUG > 0
    if (inframe)
      std::cout << " HIT\n";
    else
      std::cout << " NOT HIT\n";
#endif
    if (!inframe) continue;

    // the current state is correctly exited, so there is a transition on this surface
    found               = true;
    exiting             = true;
    relocated           = false;
    relocated_left_side = !left_side; // opposite to exiting side
    recompute_onsurf    = false;
    onsurf              = onsurf_crt;
    distance            = dist;
    isurfcross          = isurf;
    // backup exited state
    out_state = in_state;
    out_state.SetLastExited();
    // the default next navigation index is the one of the common state for the surface
    out_state.SetNavIndex(vecgeom::NavigationState(surf.fDefaultState).GetNavIndex());
    out_state.SetBoundaryState(true);
    // Check if a scene is exited on this surface
    bool is_scene_surface = surf.IsSceneSurface();
    while (is_scene_surface) {
      // exiting a scene volume, we need to find the matching surface in the parent scene
      // this is the surface among the parent state exiting candidates at surf_index
      // Move to parent scene reference frame
      out_state.PopScene();
      unsigned short parent_scene_id = 0, dummy_id = 0;
      out_state.GetSceneId(parent_scene_id, dummy_id);
      auto parent_state_id    = out_state.GetId();
      auto const &cand_scene  = surfdata.GetCandidates(parent_scene_id, parent_state_id);
      isurfcross              = cand_scene[surf_index];
      char sides_scene        = cand_scene.fSides[surf_index];
      relocated_left_side     = (sides_scene & kLside) == 0;
      auto const &parent_surf = surfdata.fCommonSurfaces[isurfcross];
      // need to convert the onsurf point in the parent scene coordinate system
      recompute_onsurf = true;
      is_scene_surface = parent_surf.IsSceneSurface();
      if (is_scene_surface)
        // Still on a parent scene surface, need to find the parent frame on the left side
        surf_index = parent_surf.fLeftSide.Top(surfdata).fSurfIndex;
    }
    continue; // there may be closer surfaces being crossed
  }

  // Now check entering candidates
  if (is_scene) {
    scene_trans.Clear();
    in_state.TopMatrix(scene_trans);
    local_scene    = scene_trans.Transform(point);
    localdir_scene = scene_trans.TransformDirection(direction);
  }

  for (auto icand = new_cand.fNExiting; icand < new_cand.fNcand; ++icand) {
    int isurf = new_cand[icand];
    if (isurf == skip_surf) continue;
    auto const &surf = surfdata.fCommonSurfaces[isurf];
    char sides       = new_cand.fSides[icand];

    // Convert point and direction to surface frame
    auto const &trans         = surfdata.fGlobalTrans[surf.fTrans];
    Vector3D<Real_t> local    = trans.Transform(local_scene);
    Vector3D<Real_t> localdir = trans.TransformDirection(localdir_scene);

    // Compute distance to surface
    Real_t dist;
    bool flipped          = false;
    auto unplaced         = surfdata.GetUnplaced(isurf, flipped);
    bool left_side        = (sides & kLside) > 0;
    bool right_side       = (sides & kRside) > 0;
    bool check_both_sides = left_side && right_side;
    bool visibility       = !left_side ^ flipped;
    bool surfhit          = unplaced.Intersect(local, localdir, visibility, surfdata, dist);
    if (!surfhit && check_both_sides) {
      // Left side already checked, now check right side
      // Note: only one side can have a valid entering
      left_side  = false;
      visibility = !flipped;
      surfhit    = unplaced.Intersect(local, localdir, visibility, surfdata, dist);
    }
    if (!surfhit || dist < -vecgeom::kTolerance || dist >= distance) continue;
#if SURF_NAV_DEBUG > 0
    std::cout << " to in  -> surface " << isurf << " hit at dist = " << dist << " -> ";
#endif
    Vector3D<Real_t> onsurf_crt = local + dist * localdir;

    // This is an entering surface for in_state
    // First check if there is a parent frame on the entry side. If this is the case
    // and it is missed, then we have a virtual hit so we skip

    auto const &entry_side = left_side ? surf.fLeftSide : surf.fRightSide;
#if SURF_NAV_DEBUG > 0
    if (!entry_side.fExtent.Inside(onsurf_crt, surfdata)) std::cout << " NOT HIT\n";
#endif
    // first check the extent of the entry side using onsurf
    if (entry_side.HasExtent() && !entry_side.fExtent.Inside(onsurf_crt, surfdata)) continue;

    bool has_children = entry_side.fNsurf > entry_side.fNumParents;
    bool full_check   = !has_children;
    // Check first parent frames
    auto iframe = CheckFramesEntering(isurf, left_side, in_state, in_navind, kCheckParents, dist, point, direction,
                                      onsurf_crt, surfdata);
    if (iframe < 0) {
      if (!has_children) continue;
      // We need to check also the children
      iframe     = CheckFramesEntering(isurf, left_side, in_state, in_navind, kCheckChildren, dist, point, direction,
                                       onsurf_crt, surfdata);
      full_check = true;
    }

#if SURF_NAV_DEBUG > 0
    if (iframe >= 0)
      std::cout << " HIT\n";
    else
      std::cout << " NOT HIT\n";
#endif
    if (iframe >= 0) {
      auto const &framedsurf = entry_side.GetSurface(iframe, surfdata);
      // This surface is certainly hit because the parent frame is hit
      found               = true;
      exiting             = false;
      relocated           = full_check;
      relocated_left_side = left_side; // relocated side same as entering side
      recompute_onsurf    = false;
      onsurf              = onsurf_crt;
      distance            = dist;
      isurfcross          = isurf;
      // compute exited state
      out_state = in_state;
      out_state.SetLastExited();
      // the default next navigation index is the state corresponding to the common parent
      if (is_scene)
        out_state.PushScene(framedsurf.fState);
      else
        out_state.SetNavIndex(framedsurf.fState);
      out_state.SetBoundaryState(true);
      // Check if this is a portal
      if (framedsurf.fSceneCS > 0) {
        isurfcross             = framedsurf.fSceneCS;
        relocated_left_side    = true; // a portal frame is always on the left side
        auto const &scene_surf = surfdata.fCommonSurfaces[isurfcross];
        // If there are only parent frames on the left side, we can complete relocation
        auto const &scene_side = scene_surf.fLeftSide;
        if (scene_side.fNsurf == scene_side.fNumParents) {
          relocated = true;
        } else {
          // We need to search the daughter frames only, after recomputing onsurf
          // Recompute onsurf
          relocated        = false;
          recompute_onsurf = true;
        }
      }
      continue; // check next
    }
  }

  if (found) {
    exit_surf = isurfcross;
    if (!relocated) {
      // do the relocation on the entering side
      auto const &surf_check = surfdata.fCommonSurfaces[isurfcross];
      if (recompute_onsurf) {
        // Moving to a parent scene, we need to recompute the local point on surface
        scene_trans.Clear();
        if (exiting)
          out_state.SceneMatrix(scene_trans);
        else
          out_state.TopMatrix(scene_trans);
        local_scene = scene_trans.Transform(point + distance * direction);
        onsurf      = surfdata.fGlobalTrans[surf_check.fTrans].Transform(local_scene);
      }

      // Parent frames have been already checked if entering
      auto to_check = exiting ? kCheckAll : kCheckChildren;
      auto iframe = CheckFramesEntering(isurfcross, relocated_left_side, in_state, in_navind, to_check, distance, point,
                                        direction, onsurf, surfdata);

      if (iframe < 0) {
        relocated    = true;
        auto top_ind = vecgeom::NavigationState(surf_check.fDefaultState).GetNavIndex();
        if (exiting) {
          if ((top_ind == 0) && (surf_check.GetSceneId() > 0))
            out_state.PopScene();
          else
            out_state.SetNavIndex(top_ind);
        }
      }

      while (!relocated && iframe >= 0) {
        //  Found a crossed frame
        auto const &surf       = surfdata.fCommonSurfaces[isurfcross];
        auto const &entry_side = relocated_left_side ? surf.fLeftSide : surf.fRightSide;
        auto const &framedsurf = entry_side.GetSurface(iframe, surfdata);
        if (surf.IsSceneSurface())
          out_state.PushScene(framedsurf.fState);
        else
          out_state.SetNavIndex(framedsurf.fState);
        // We may have entered a new scene, so we need to search the scene surface
        relocated = true;
        if (framedsurf.fSceneCS > 0) {
          isurfcross             = framedsurf.fSceneCS;
          relocated_left_side    = true; // a portal frame is always on the left side
          auto const &scene_surf = surfdata.fCommonSurfaces[isurfcross];
          // If there are only parent frames on the left side, we can complete relocation
          auto const &scene_side = scene_surf.fLeftSide;
          if (scene_side.fNumParents < scene_side.fNsurf) {
            // We need to search the daughter frames only
            relocated = false;
            // Recompute onsurf
            scene_trans.Clear();
            out_state.TopMatrix(scene_trans);
            local_scene = scene_trans.Transform(point + distance * direction);
            onsurf      = surfdata.fGlobalTrans[scene_surf.fTrans].Transform(local_scene);
            iframe = CheckFramesEntering(isurfcross, relocated_left_side, in_state, in_navind, kCheckChildren, distance,
                                         point, direction, onsurf, surfdata);
          }
        }
      }
    } // end relocation
  }
  // Fix the out_state if pointing to a 0 scene
  if (out_state.GetSceneLevel() > 0 && out_state.GetNavIndex() == 0) out_state.PopScene();
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  if (!(in_navind == 0 || distance < vecgeom::InfinityLength<Real_t>())) {
    VECGEOM_LOG(critical) << std::setprecision(16) << "at point " << point << " and direction " << direction
                          << std::endl;
  }
#endif
  assert(in_navind == 0 ||
         (distance < vecgeom::InfinityLength<Real_t>() && "ComputeStepAndHit cannot return infinite distance"));
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
VECCORE_ATT_HOST_DEVICE Real_t ComputeSafety(vecgeom::Vector3D<Real_t> const &point,
                                             vecgeom::NavigationState const &in_state, int &closest_surf)
{
  constexpr char kLside = 0x01;
  constexpr char kRside = 0x02;
  using vecgeom::NavigationState;
  auto const &surfdata = SurfData<Real_t>::Instance();
  closest_surf         = 0;
  int last_logic_volid = 0;
  Real_t safety        = vecgeom::InfinityLength<Real_t>();
  Vector3D<Real_t> onsurf;
  unsigned short scene_id = 0, newscene_id = 0;
  // Get the list of visible candidate surfaces for in_state
  in_state.GetSceneId(scene_id, newscene_id);
  auto const &cand = surfdata.GetCandidates(scene_id, in_state.GetId());

  // Convert the point to the scene coordinate system
  vecgeom::Transformation3D scene_trans;
  in_state.SceneMatrix(scene_trans);
  Vector3D<Real_t> local_scene = scene_trans.Transform(point);

  // First check Exiting candidates
  for (auto icand = 0; icand < cand.fNExiting; ++icand) {
    bool validSafety = true;
    int isurf        = cand[icand];
    if (isurf == 0) continue;
    auto const &surf     = surfdata.fCommonSurfaces[isurf];
    auto const &topframe = surfdata.fFramedSurf[surf.fLeftSide.fSurfaces[0]];
    // Skip already checked logic surfaces. (TO REVIEW AFTER THE CHANGE to SIDES)
    if (topframe.fLogicId && last_logic_volid == topframe.VolumeId()) continue;

    // Convert point to surface frame
    auto const &trans      = surfdata.fGlobalTrans[surf.fTrans];
    Vector3D<Real_t> local = trans.Transform(local_scene);
    Vector3D<Real_t> onsurf_crt;
    Real_t safety_surf;
    bool flipped  = false;
    auto unplaced = surfdata.GetUnplaced(isurf, flipped);

    // left_side is the side which defines the exit normal
    char sides            = cand.fSides[icand];
    bool left_side        = (sides & kLside) > 0;
    bool right_side       = (sides & kRside) > 0;
    bool check_both_sides = left_side && right_side;
    bool visibility       = left_side ^ flipped;

    // Compute signed closest distance to surface. The closest projected point on surface is computed, except for:
    // - negative safety (coming from the wrong side)
    // - exiting framed surfaces for which fUseSurfSafety is true
    // To test if on GPU is better to compute the projection systematically
    // bool compute_onsurf = exiting ? !exit_side.GetSurface(candExiting.fFrameInd[icand], surfdata).fUseSurfSafety :
    // true;
    auto const &check_side = left_side ? surf.fLeftSide : surf.fRightSide;
    bool compute_onsurf    = !check_side.GetSurface(cand.fFrameInd[icand], surfdata).fUseSurfSafety;
    bool can_compute       = unplaced.Safety(local, visibility, surfdata, safety_surf, compute_onsurf, onsurf_crt);
    if (!can_compute && check_both_sides) {
      // Left side already checked, now check right side
      // Note: only one side can have a valid safety
      left_side      = false;
      visibility     = flipped;
      compute_onsurf = !surf.fRightSide.GetSurface(cand.fFrameInd[icand], surfdata).fUseSurfSafety;
      can_compute    = unplaced.Safety(local, visibility, surfdata, safety_surf, compute_onsurf, onsurf_crt);
    }

    if (!can_compute || safety_surf < -vecgeom::kTolerance || safety_surf >= safety) continue;

    // This is an exiting surface for in_state
    // Only check the frame of the current state on this surface

    // Get the index of the framed surface on the exit side. This assumes a SINGLE frame inprint
    // coming from a touchable on any common surface.
    auto const &exit_side  = left_side ? surf.fLeftSide : surf.fRightSide;
    int frameind           = cand.fFrameInd[icand]; // index of framed surface on the side
    auto const &framedsurf = exit_side.GetSurface(frameind, surfdata);
    // Check if the exited frame safety is needed at all
    auto safetyFrame = safety_surf;
    if (compute_onsurf) {
      // We need to compute also the safety of the projection of the point on surface to the frame
      safetyFrame = framedsurf.SafetyFrame(onsurf_crt, safety_surf, surfdata, validSafety);
    }
    if (validSafety && safety > safetyFrame) {
      // If Boolean surface, compute only once safety for the entire volume shell
      if (framedsurf.fLogicId) {
        Real_t safetyLogic = LogicSafety(point, true, NavigationState{framedsurf.fState}, surfdata, safety);
        last_logic_volid   = vecgeom::NavigationState::TopImpl(framedsurf.fState)->GetLogicalVolume()->id();
        if (safety > safetyLogic) {
          safety       = safetyLogic;
          closest_surf = isurf;
        }
      } else {
        safety       = safetyFrame;
        closest_surf = isurf;
      }
    }
  }

  // Now check the entering candidates
  for (auto icand = cand.fNExiting; icand < cand.fNcand; ++icand) {
    bool validSafety     = true;
    int isurf            = cand[icand];
    auto const &surf     = surfdata.fCommonSurfaces[isurf];
    auto const &topframe = surfdata.fFramedSurf[surf.fLeftSide.fSurfaces[0]];
    // Skip already checked logic surfaces.  (TO REVIEW AFTER THE CHANGE to SIDES)
    if (topframe.fLogicId && last_logic_volid == topframe.VolumeId()) continue;

    // Convert point to surface frame
    auto const &trans      = surfdata.fGlobalTrans[surf.fTrans];
    Vector3D<Real_t> local = trans.Transform(local_scene);
    Vector3D<Real_t> onsurf_crt;
    Real_t safety_surf;
    bool flipped  = false;
    auto unplaced = surfdata.GetUnplaced(isurf, flipped);

    // left_side is the side which defines the exit normal
    // left_side is the side which defines the exit normal
    char sides            = cand.fSides[icand];
    bool left_side        = (sides & kLside) > 0;
    bool right_side       = (sides & kRside) > 0;
    bool check_both_sides = left_side && right_side;
    bool visibility       = !left_side ^ flipped;
    // Compute signed closest distance to surface. The closest projected point on surface is computed, except for:
    // - negative safety (coming from the wrong side)
    // - exiting framed surfaces for which fUseSurfSafety is true
    // To test if on GPU is better to compute the projection systematically
    bool compute_onsurf = true;
    bool can_compute    = unplaced.Safety(local, visibility, surfdata, safety_surf, compute_onsurf, onsurf_crt);

    if (!can_compute && check_both_sides) {
      // Left side already checked, now check right side
      // Note: only one side can have a valid safety
      left_side   = false;
      visibility  = !flipped;
      can_compute = unplaced.Safety(local, visibility, surfdata, safety_surf, compute_onsurf, onsurf_crt);
    }
    if (!can_compute || safety_surf < -vecgeom::kTolerance || safety_surf >= safety) continue;

    // Entering side. We only check the parent frames on the side
    auto const &entry_side = left_side ? surf.fLeftSide : surf.fRightSide;
    const int num_parents  = entry_side.fNumParents;
    int iparent            = 0;
    Real_t safetyParent    = safety;
    // Parent frames are last in the list
    for (auto ind = entry_side.fNsurf - 1; ind >= 0; --ind) {
      auto const &framedsurf = entry_side.GetSurface(ind, surfdata);
      if (framedsurf.fParent >= 0) continue; // skip children
      iparent++;
      auto safetyFrame = framedsurf.SafetyFrame(onsurf_crt, safety_surf, surfdata, validSafety);
      if (validSafety && safetyFrame < safetyParent) {
        // If Boolean surface, compute only once safety for the entire volume shell
        if (framedsurf.fLogicId) {
          Real_t safetyLogic = LogicSafety(point, false, NavigationState{framedsurf.fState}, surfdata, safetyFrame);
          last_logic_volid   = vecgeom::NavigationState::TopImpl(framedsurf.fState)->GetLogicalVolume()->id();
          if (safetyParent > safetyLogic) {
            safetyParent = safetyLogic;
            closest_surf = isurf;
          }
        } else {
          safetyParent = safetyFrame;
          closest_surf = isurf;
        }
      }
      if (iparent == num_parents) break;
    }
    if (safety > safetyParent) {
      safety       = safetyParent;
      closest_surf = isurf;
    }
  }
  return safety;
}

} // namespace protonav
} // namespace vgbrep
#endif
