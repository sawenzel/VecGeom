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
  vecgeom::Transformation3DMP<Real_t> trans;
  in_state.TopMatrix(trans);
  trans.Transform(point, localpoint);
  auto vol    = in_state.Top();
  auto volId  = vol->GetLogicalVolume()->id();
  auto inside = LogicInsideLocal(localpoint, volId, surfdata, logic_id, is_inside);
  return inside;
}

/// @brief Function checking is a given framed surface is exited at a distance from a global point
/// @param point Global point
/// @param direction Global direction
/// @param distance Crossing distance
/// @param onsurf Propagated point on surface in the surface coordinate frame
/// @param exited_state State beeing exited
/// @param framedsurf Framed surface to be checked
/// @param last_bool_state Set as last Boolean state checked if this is a Boolean surface
/// @param surfdata Surface data storage
/// @return Exiting the frame
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool IsExitingFrame(Vector3D<Real_t> const &point, Vector3D<Real_t> const &direction,
                                            Real_t distance, Vector3D<Real_t> const &onsurf,
                                            vecgeom::NavigationState const &exited_state,
                                            FramedSurface const &framedsurf, NavIndex_t &last_bool_state,
                                            SurfData<Real_t> const &surfdata)
{
  constexpr Real_t kPushDistance = 1000 * vecgeom::kToleranceDist<Real_t>;
  bool inframe                   = framedsurf.fNeverCheck ? true : framedsurf.InsideFrame(onsurf, surfdata);
  if (inframe && framedsurf.fLogicId) {
    last_bool_state = framedsurf.fState;
    // Frame cross does not guarantee a real surface cross in case of Booleans
    // For a real exiting, the post-crossing point must be outside the Boolean
    auto pushedPoint = point + (distance + kPushDistance) * direction;
    // The logic for the frame that is exited can be set to false without a numerical check
    auto inside = LogicInside(pushedPoint, exited_state, surfdata, framedsurf.fLogicId, false);
    if (inside) inframe = false;
  }
  return inframe;
}

/// @brief Find the topmost exited frame on the common surface pointed by exiting_FS
/// @param exiting_FS Locator of the first frame to be checked
/// @param must_exit_state A frame with the same state as the input state must be exited
/// @param state Touchable state for the volume having the exiting_FS framed surface
/// @param distance Distance from global point to the surface
/// @param point Global point
/// @param direction Global direction
/// @param onsurf Point on the CS, in the surface reference frame
/// @param onscene The checked CS is a portal on the parent scene
/// @param surfdata Surface data storage
/// @param top_exit_state Navigation index of the topmost exited frame
/// @param exiting_scene The exited frame has a TOP_SCENE state
/// @param surf_index Surface index of the topmost exited frame
/// @param out_state Exiting state
/// @return A frame was exited
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool CheckFramesExiting(FSlocator &exiting_FS, vecgeom::NavigationState const &state,
                                                bool onscene, Real_t distance, Vector3D<Real_t> const &point,
                                                Vector3D<Real_t> const &direction, Vector3D<Real_t> const &onsurf,
                                                Vector3D<Real_t> &recomputed_onsurf, SurfData<Real_t> const &surfdata,
                                                NavIndex_t &top_exit_state, bool &exiting_scene, int &surf_index,
                                                vecgeom::NavigationState &out_state)
{
  constexpr NavIndex_t kInvalidState = NavIndex_t(-1);

  int isurf             = exiting_FS.GetCSindex();
  auto const &surf      = surfdata.fCommonSurfaces[isurf];
  bool left_side        = exiting_FS.IsLeftSide();
  bool is_scene_surface = surf.IsSceneSurface();
  auto in_navind        = state.GetNavIndex();
  // We need to check frame intersection
  // This is an exiting surface for in_state
  // First check the frame of the current state on this surface
  auto const &exit_side = left_side ? surf.fLeftSide : surf.fRightSide;
  // Get the index of the first framed surface on the exit side.
  int frameind_start = exiting_FS.frame_id;
  // If the current touchable is exited on this surface, it MUST be through the inside of the
  // corresponding frames. Loop all frames coming from the same touchable
  bool inframe    = onscene;
  int parent_ind  = -1;
  bool has_parent = false;
  bool embedded   = true;

  NavIndex_t last_bool_state = kInvalidState; // last checked Boolean state exited
  auto exited_state          = state;
  if (onscene) {
    // Moving to a parent scene, we need to recompute the local point on surface
    vecgeom::Transformation3DMP<Real_t> scene_trans;
    state.SceneMatrix(scene_trans);
    auto local_scene  = scene_trans.Transform(point + distance * direction);
    recomputed_onsurf = surfdata.fGlobalTrans[surf.fTrans].Transform(local_scene);
  }
  auto const &onsurf_crt = onscene ? recomputed_onsurf : onsurf;

  // Adjust the state to reflect the topmost exited frame
  auto setTopExited = [&](FramedSurface const &framed_surf, int ind) {
    surf_index    = framed_surf.fSurfIndex;
    parent_ind    = framed_surf.fParent;
    has_parent    = framed_surf.GetParentState(top_exit_state);
    exiting_scene = is_scene_surface && (!has_parent);
    embedded      = framed_surf.fEmbedded;
    exiting_FS.Set(isurf, ind, left_side);
  };

  for (int ind = frameind_start; ind < exit_side.fNsurf; ++ind) {
    auto const &framedsurf = exit_side.GetSurface(ind, surfdata);
    // The deepest exited frame surface must belong to the input state
    if (!onscene && !inframe && framedsurf.fState != in_navind) continue;
    // Skip if this is an already checked Boolean
    if (framedsurf.fState == last_bool_state) continue;
    // Check if the frame mask is crossed
    bool inframe_tmp = (ind == frameind_start) && inframe;
    if (!inframe_tmp)
      inframe_tmp = IsExitingFrame<Real_t>(point, direction, distance, onsurf_crt, exited_state, framedsurf,
                                           last_bool_state, surfdata);
    if (inframe_tmp) {
      // The frame is exited
      inframe = true;
      // Cache the top exited state and local surface index for the exited surface
      setTopExited(framedsurf, ind);
      // If this frame has no parent this is the topmost exited one
      if (parent_ind < 0) break;
      // loop over parent frames
      while (parent_ind > 0) {
        // next parent to be checked in case the frame is not embedded is parent_ind
        ind = parent_ind - 1;
        // Not embedding frames must be checked thoroughly
        if (!embedded) break;
        // The frame is embedded in the parent, so the parent is also exited
        auto const &parent_framedsurf = exit_side.GetSurface(parent_ind, surfdata);
        setTopExited(parent_framedsurf, parent_ind);
        if (parent_framedsurf.fState)
          exited_state.SetNavIndex(parent_framedsurf.fState);
        else
          exited_state.PopScene();
      }
      if (embedded) break;
    }
  }
  if (inframe) {
    // backup exited state
    out_state = state;
    out_state.SetLastExited();
    // set navigation state to top exit state which is either the surf.fDefaultState or a boolean in case an exiting
    // frame was found that does not exit through the default state but stays within the boolean solid
    out_state.SetNavIndex(top_exit_state);
    out_state.SetBoundaryState(true);
    exiting_FS.state = out_state;
  }
  return inframe;
}

/// @brief Finction checking if any frame on a side of common surface side is hit
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
                                                NavIndex_t in_navind, bool is_scene, int to_be_checked, Real_t distance,
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
    // surface being crossed, so this surface must be ignored.
    // NOTE: this is NOT the case if a new scene is entered
    if (!common_surface.IsSceneSurface() && framedsurf.fState == in_navind) continue;
    // Same as above, but for scene frames
    if (is_scene && framedsurf.fState == 0) continue;
    auto inframe = framedsurf.InsideFrame(onsurf_local, surfdata);
    if (!inframe) continue;
    if (framedsurf.fLogicId) {
      auto pushedPoint   = point + (distance + kPushDistance) * direction;
      auto checked_state = in_state;
      if (framedsurf.fState) {
        if (common_surface.IsSceneSurface() && checked_state.GetNavIndex() > 0)
          checked_state.PushScene(framedsurf.fState);
        else
          checked_state.SetNavIndex(framedsurf.fState);
      }
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
  vecgeom::Transformation3DMP<Real_t> trans;
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
                                                                    vecgeom::VPlacedVolume const *exclude = nullptr)
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

/// @brief Check whether
/// @tparam Real_t Floating point type
/// @param path Path pointing to the top volume to check
/// @param exiting_FS FSLocator holding the highest exited framed surface
/// @return Placed volume pointer containing the point
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool VolumeHasCommonSurface(vecgeom::NavigationState &path, FSlocator const &exiting_FS)
{

  // always included in outside
  if (path.GetNavIndex() == 0) return false;
  // for extruding overlaps no common surface needs to be excluded
  if (exiting_FS.GetFSindex() == -1) return false;

  auto const &surfdata  = SurfData<Real_t>::Instance();
  auto const &surf      = surfdata.fCommonSurfaces[exiting_FS.GetCSindex()];
  auto const &exit_side = exiting_FS.IsLeftSide() ? surf.fLeftSide : surf.fRightSide;

  auto state_id = path.GetNavIndex();

  // non-embedded surfaces cannot be excluded based on this approach, since this could exclude the correct parent that
  // is entered through the non-embedded part of the frame.
  if (!surfdata.IsFramedSurfaceEmbedded(exiting_FS)) return false;

  // loop over all framed surfaces to see if booleans are on the common surface
  // booleans can have the same state while having a different framed surface, so they need a full inside check
  for (int isurf = 0; isurf < exit_side.fNsurf; isurf++) {
    if (surfdata.fFramedSurf[exit_side.fSurfaces[isurf]].fLogicId) return false;
  }

  // loop over all states of the exited common surface and exclude volumes of that state
  // NOTE: this might not be correct for booleans
  for (int isurf = 0; isurf < exit_side.fNsurf; isurf++) {
    if (surfdata.fFramedSurf[exit_side.fSurfaces[isurf]].fState == state_id) return true;
  }
  return false;
}

/// @brief Locate a point in a volume sub-hierarchy
/// @tparam Real_t Floating point type
/// @param starting_path initial path before the ComputeStepAndHit
/// @param point Point in local volume coordinates
/// @param path Path pointing to the top volume to check
/// @param exiting_FS FSLocator holding the highest exited framed surface
/// @param top_exit_state index of state of highest exited framed surface
/// @return Placed volume pointer containing the point
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE vecgeom::VPlacedVolume const *ReLocatePointIn(vecgeom::NavigationState &starting_path,
                                                                      vecgeom::Vector3D<Real_t> const &point,
                                                                      vecgeom::Vector3D<Real_t> const &direction,
                                                                      vecgeom::NavigationState &path,
                                                                      FSlocator const &exiting_FS)
{
  using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const *;
  auto const &surfdata     = SurfData<Real_t>::Instance();

  // set path to be starting path to check for daughters
  auto model_out_state             = path;
  path                             = starting_path;
  VPlacedVolumePtr_t currentvolume = starting_path.Top();
  VPlacedVolumePtr_t prev_volume   = path.GetLastExited();

  constexpr Real_t kPushDistance = 1000 * vecgeom::kToleranceDist<Real_t>;
  bool in_boolean                = false;

  // first, check daughter volumes of the current path to find if the point lies within any of the daughters
  bool godeeper;
  bool inside_daughter = false;
  if (exiting_FS.GetFSindex() != -1) { // if not exiting surface was found, we don't need to check for children
    do {
      godeeper = false;
      for (auto *daughter : currentvolume->GetDaughters()) {
        // if (daughter == prev_volume) {
        //   // Only exclude the placed volume once since we could enter it again via a
        //   // different volume history.
        //   prev_volume = nullptr;
        //   continue;
        // }
        path.Push(daughter);
        bool inside = LogicInside(point, path, surfdata);

        if (inside) {
          inside_daughter = true;
          currentvolume   = daughter;
          godeeper        = true;
          break;
        } else {
          path.Pop();
        }
      }
    } while (godeeper);
  }

  // if point was located in daughter and this is not the previous starting path, return
  if (inside_daughter && !path.HasSamePathAsOther(starting_path)) return currentvolume;

  // prev_volume = starting_path.Top();
  //   // else, reset to initial output path
  //   path = starting_path;
  //   int surf_index;

  // // ########### WORK IN PROGRESS #################
  //   if (exiting_FS.GetCSindex() != -1) {
  //     constexpr char kLside         = 1;
  //     bool exiting_scene = false;
  //     bool left_side = exiting_FS.IsLeftSide();
  //     int isurf = exiting_FS.GetCSindex();
  //     int frame_id = exiting_FS.GetFSindex();

  //     unsigned short parent_scene_id = 0, dummy_id = 0;
  //     vecgeom::NavStateIndex_t parent_state_id;
  //     do {
  //       auto const &surf = surfdata.fCommonSurfaces[isurf];
  //       auto const &exit_side = left_side ? surf.fLeftSide : surf.fRightSide;

  //       // navigate to highest parent frame
  //       auto const &framedsurf = exit_side.GetSurface(frame_id, surfdata);
  //       // Find the appropriate surface index
  //       surf_index       = framedsurf.fSurfIndex;
  //       int parent_frame = framedsurf.fParent;
  //       while (parent_frame > 0) {

  //         if (exit_side.GetSurface(parent_frame, surfdata).fLogicId) {
  //           auto pushedPoint = point + kPushDistance * direction;
  //           vecgeom::NavigationState boolean_state;
  //           boolean_state.SetNavIndex(exit_side.GetSurface(parent_frame, surfdata).fState);
  //           auto inside = LogicInside(pushedPoint, boolean_state, surfdata, 0, false);
  //           if (inside) {
  //             in_boolean = true;
  //             // boolean is not exited, top_exit_frame is not the boolean itself but the current frame
  //             // boolean_state.SetNavIndex(vecgeom::NavigationState(exit_side.GetSurface(parent_frame,
  //             surfdata).fState).GetNavIndex());
  //             // exiting_scene = false; // if we are still in the boolean, we should not exit the scene ()
  //             if (exiting_scene) {
  //               // Get parent scene and state id's before exiting to the next scene
  //               path.GetSceneId(parent_scene_id, dummy_id);
  //               parent_state_id = path.GetId();
  //             }
  //             break;
  //           }
  //         }

  //         surf_index   = exit_side.GetSurface(parent_frame, surfdata).fSurfIndex;
  //         parent_frame = exit_side.GetSurface(parent_frame, surfdata).fParent;
  //         exiting_scene = (surf.IsSceneSurface()) && (path.GetNavIndex() > 0);
  //         // exiting_scene = (exit_side.GetSurface(parent_frame, surfdata).fState == 0) && (surf.IsSceneSurface());

  //         path.Pop();
  //         path.SetLastExited();

  //         if (exiting_scene) {
  //           // Get parent scene and state id's before exiting to the next scene
  //           path.GetSceneId(parent_scene_id, dummy_id);
  //           parent_state_id = path.GetId();
  //         }
  //       }

  //       if (exiting_scene) {
  //         // Move to parent scene state
  //         // path.Pop(); // not clear whether path should be popped here or only later
  //         // unsigned short parent_scene_id = 0, dummy_id = 0;
  //         // Get parent scene and state id's before exiting to the next scene
  //         // path.GetSceneId(parent_scene_id, dummy_id);
  //         // parent_state_id = path.GetId();
  //         // Get crossed parent scene common surface, frame id, and side from exiting candidates
  //         auto const &cand_scene  = surfdata.GetCandidates(parent_scene_id, parent_state_id);
  //         isurf                   = cand_scene[surf_index];
  //         frame_id                = cand_scene.fFrameInd[surf_index];
  //         char sides_scene        = cand_scene.fSides[surf_index];
  //         left_side               = (sides_scene & kLside) == 0;
  //         // path.Pop();

  //        }
  //     } while (exiting_scene);
  //   }

  if (exiting_FS.GetFSindex() != -1) {

    // reset to highest exited state
    path = exiting_FS.state;
    // set to the state of the exited surface itself
    path.SetNavIndex(surfdata.FramedSurfaceState(exiting_FS));
  } else {
    path = starting_path;
  }

  // we need to check if the exiting framed surface belongs to a boolean.
  // in case of a boolean, an exiting surface does not necessarily mean that the full boolean is exited,
  // thus, in that case we should not go up in the path before searching in the boolean itself
  if (surfdata.FramedSurfaceLogicId(exiting_FS)) {
    auto pushedPoint = point + kPushDistance * direction;
    vecgeom::NavigationState boolean_state;
    boolean_state = path;
    in_boolean    = LogicInside(pushedPoint, boolean_state, surfdata, 0, false);
  }

  if (!in_boolean) {
    // exclude volume of the highest parent of the exited framed surface
    if (path.GetNavIndex() > 1) path.SetLastExited();
    prev_volume = path.GetLastExited();
    // navigate one level higher to search for inside
    if (path.GetNavIndex() > 1) path.Pop(); // go one level higher, unless we are in the top volume
  } else {
    prev_volume = starting_path.Top();
  }

  currentvolume = path.Top();

  // check whether the point is in the parent volume, otherwise go higher until it is found
  bool gohigher = false;

  bool inside;
  do {
    gohigher     = false;
    auto same_cs = VolumeHasCommonSurface<Real_t>(path, exiting_FS);
    if (same_cs && !in_boolean) {
      inside = false;
    } else {
      inside = LogicInside(point, path, surfdata);
    }

    if (inside == false) {
      prev_volume = currentvolume;
      path.Pop();
      currentvolume = path.Top();
      gohigher      = true;
    }
  } while (gohigher);

  do {
    godeeper = false;
    for (auto *daughter : currentvolume->GetDaughters()) {
      if (daughter == prev_volume) {
        // Only exclude the placed volume once since we could enter it again via a
        // different volume history.
        prev_volume = nullptr;
        continue;
      }
      unsigned short scene_id = 0, newscene_id = 0;
      bool is_scene = path.GetSceneId(scene_id, newscene_id);
      path.Push(daughter);
      auto same_cs = VolumeHasCommonSurface<Real_t>(path, exiting_FS);
      if (same_cs && !is_scene) {
        inside = false;
      } else {
        inside = LogicInside(point, path, surfdata);
      }
      if (inside) {
        currentvolume = daughter;
        godeeper      = true;
        break;
      } else {
        path.Pop();
      }
    }
  } while (godeeper);

  return currentvolume;
}

/// @brief Compute the distance to the unplaced surface of a common surface
/// @tparam Real_t Precision type
/// @param point Point in scene coordinates
/// @param direction Direction in scene coordinates
/// @param surfdata Surface data
/// @param exiting Is the surface an exiting candidate
/// @param isurf Common surface index
/// @param sides Visible sides of the common surface
/// @param surfhit Returned validity of the crossing
/// @param onsurf Point on surface in the CS frame
/// @return Distance to unplaced surface
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t DistanceToUnplaced(vecgeom::Vector3D<Real_t> const &point,
                                                  vecgeom::Vector3D<Real_t> const &direction,
                                                  SurfData<Real_t> const &surfdata, int isurf, char sides, bool exiting,
                                                  bool &left_side, bool &surfhit, vecgeom::Vector3D<Real_t> &onsurf)
{
  constexpr char kLside = 1;
  constexpr char kRside = 2;
  auto const &surf      = surfdata.fCommonSurfaces[isurf];

  // Convert point and direction to surface frame
  auto const &trans         = surfdata.fGlobalTrans[surf.fTrans];
  Vector3D<Real_t> local    = trans.Transform(point);
  Vector3D<Real_t> localdir = trans.TransformDirection(direction);

  // Compute distance to surface
  Real_t dist;
  bool flipped  = false;
  auto unplaced = surfdata.GetUnplaced(isurf, flipped);

  left_side             = (sides & kLside) > 0;
  bool check_both_sides = left_side && (sides & kRside) > 0;
  bool visibility       = !exiting ^ left_side ^ flipped;
  surfhit               = unplaced.Intersect(local, localdir, visibility, surfdata, dist);
  if (!surfhit && check_both_sides) {
    // Left side already checked, now check right side
    // Note: only one side can have a valid exiting
    left_side  = false;
    visibility = !exiting ^ flipped;
    surfhit    = unplaced.Intersect(local, localdir, visibility, surfdata, dist);
  }
  if (surfhit) onsurf = local + dist * localdir;
  return dist;
}

/// @brief Method computing the distance to the next surface and state after crossing it
/// @tparam Real_t Floating point type for the interface and data storage
/// @param point Global point
/// @param direction Global direction
/// @param in_state Input navigation state before crossing
/// @param out_state Output navigation state after crossing
/// @param surfdata Surface data storage
/// @param exiting_FS FSLocator storing common surface id, framed surface id, and side of highest exited framed surface
/// @param stepmax maximum step
/// @return Distance to next surface.
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t ComputeStepAndHit(vecgeom::Vector3D<Real_t> const &point,
                                                 vecgeom::Vector3D<Real_t> const &direction,
                                                 vecgeom::NavigationState const &in_state,
                                                 vecgeom::NavigationState &out_state, FSlocator &exiting_FS,
                                                 Real_t stepmax = vecgeom::InfinityLength<Real_t>())
{
  constexpr char kCheckAll      = 0;
  constexpr char kCheckChildren = 1;
  constexpr char kCheckParents  = 2;
  // Get the list of candidate surfaces for in_state
  auto in_navind = in_state.GetNavIndex();
  if (in_navind == 0) return vecgeom::InfinityLength<Real_t>();
  auto const &surfdata    = SurfData<Real_t>::Instance();
  Real_t distance         = stepmax;
  int isurfcross          = 0;
  unsigned short scene_id = 0, newscene_id = 0;
  bool is_scene             = in_state.GetSceneId(scene_id, newscene_id);
  auto const &cand          = surfdata.GetCandidates(scene_id, in_state.GetId());
  auto const &new_cand      = is_scene ? surfdata.GetCandidates(newscene_id, 0) : cand;
  bool found                = false;
  bool exiting              = false;
  bool exiting_scene        = false;
  bool exited_scene         = false;
  bool relocated            = false;
  bool relocated_left_side  = false;
  bool recompute_onsurf     = false;
  NavIndex_t top_exit_state = 0;
  FSlocator tmp_exiting_FS;

  Vector3D<Real_t> onsurf, recomputed_onsurf;
  vecgeom::NavigationState exited_state;
  out_state = in_state;
  out_state.SetBoundaryState(false);
  auto skip_surf = exiting_FS.GetCSindex();

  // Convert the point and direction to the scene coordinate system
  vecgeom::Transformation3DMP<Real_t> scene_trans;
  in_state.SceneMatrix(scene_trans);
  Vector3D<Real_t> local_scene    = scene_trans.Transform(point);
  Vector3D<Real_t> localdir_scene = scene_trans.TransformDirection(direction);

  // First check Exiting candidates
  for (auto icand = 0; icand < cand.fNExiting; ++icand) {
    int isurf  = cand.fCandidates[icand];
    char sides = cand.fSides[icand];
    if (isurf == 0) continue;
    bool left_side, surfhit;
    Vector3D<Real_t> onsurf_crt;
    // Compute distance to the unplaced surface
    auto dist =
        DistanceToUnplaced(local_scene, localdir_scene, surfdata, isurf, sides, true, left_side, surfhit, onsurf_crt);
    if (!surfhit || dist < -vecgeom::kToleranceDist<Real_t> || dist >= distance) continue;

    tmp_exiting_FS.Set(isurf, cand.fFrameInd[icand], left_side);
    int surf_index = 0;
    auto inframe =
        CheckFramesExiting(tmp_exiting_FS, in_state, /*onscene=*/false, dist, point, direction, onsurf_crt,
                           recomputed_onsurf, surfdata, top_exit_state, exiting_scene, surf_index, out_state);
    if (!inframe) continue;
    // the current state is correctly exited, so there is a transition on this surface
    exiting_FS          = tmp_exiting_FS;
    found               = true;
    exiting             = true;
    exited_state        = in_state;
    relocated           = false;
    relocated_left_side = !left_side; // opposite to exiting side
    recompute_onsurf    = false;
    onsurf              = onsurf_crt;
    distance            = dist;
    isurfcross          = isurf;
    // Check if a scene surface was crossed
    while (exiting_scene) {
      exited_scene = true;
      // Get the exit frame locator in the parent scene
      out_state.PopScene();
      surfdata.SceneToTouchableLocator(out_state, surf_index, exiting_FS);
      // Find the topmost exit frame on the portal
      exited_state = out_state;
      inframe      = CheckFramesExiting(exiting_FS, exited_state, /*onscene=*/true, dist, point, direction, onsurf_crt,
                                        recomputed_onsurf, surfdata, top_exit_state, exiting_scene, surf_index, out_state);
      assert(inframe == true);
      isurfcross          = exiting_FS.GetCSindex();
      relocated_left_side = !exiting_FS.IsLeftSide();
      recompute_onsurf    = true;
    }
    continue; // there may be closer surfaces being crossed
  }
  // If there is no physics step limitation, an exiting surface must be found
  if (!found && stepmax == vecgeom::InfinityLength<Real_t>()) {
    exiting_FS.Set(0, -1, 0); // frame_id = -1 indicates extruding overlap
    // This can happen if the exit point is outside the mother volume (extrusion)
    // To recover, one can return the mother state as output and a zero distance
    return stepmax;
  }

  // Now check entering candidates
  if (is_scene) {
    scene_trans.Clear();
    in_state.TopMatrix(scene_trans);
    local_scene    = scene_trans.Transform(point);
    localdir_scene = scene_trans.TransformDirection(direction);
  }

  for (auto icand = new_cand.fNExiting; icand < new_cand.fNcand; ++icand) {
    int isurf          = vecCore::math::Abs(new_cand[icand]);
    bool self_entering = new_cand[icand] < 0;
    if (isurf == skip_surf) continue;
    char sides = new_cand.fSides[icand];

    bool left_side, surfhit;
    Vector3D<Real_t> onsurf_crt;
    // Compute distance to the unplaced surface
    auto dist = DistanceToUnplaced(local_scene, localdir_scene, surfdata, isurf, sides, /*exiting=*/false, left_side,
                                   surfhit, onsurf_crt);

    if (!surfhit || dist < -vecgeom::kToleranceDist<Real_t> || dist >= distance) continue;
    // Temporary ugly solution to avoid self-entering the volume at 0 distance on the same surface
    if (self_entering && vecCore::math::Abs(dist) < vecgeom::kToleranceDist<Real_t>) continue;

    // This is an entering surface for in_state
    // First check if there is a parent frame on the entry side. If this is the case
    // and it is missed, then we have a virtual hit so we skip

    auto const &surf       = surfdata.fCommonSurfaces[isurf];
    auto const &entry_side = left_side ? surf.fLeftSide : surf.fRightSide;
    // first check the extent of the entry side using onsurf
    if (entry_side.HasExtent() && !entry_side.fExtent.Inside(onsurf_crt, surfdata)) continue;

    bool has_children = entry_side.fNsurf > entry_side.fNumParents;
    bool full_check   = !has_children;
    // Check first parent frames
    auto iframe = CheckFramesEntering(isurf, left_side, in_state, in_navind, is_scene, kCheckParents, dist, point,
                                      direction, onsurf_crt, surfdata);
    if (iframe < 0) {
      if (!has_children) continue;
      // We need to check also the children
      iframe     = CheckFramesEntering(isurf, left_side, in_state, in_navind, is_scene, kCheckChildren, dist, point,
                                       direction, onsurf_crt, surfdata);
      full_check = true;
    }

    if (iframe >= 0) {
      auto const &framedsurf = entry_side.GetSurface(iframe, surfdata);
      // safe exit surf information for entering surfaces of the lowest frame exited
      exiting_FS.Set(isurf, iframe,
                     left_side); // FIXME: set exiting FS for entering surface as well to faster detect overlaps?
      // keeping the comment below for scene debugging, will clean with scene fix FIXME
      // exit_surf.frame_id  = iframe;
      // exit_surf.left_side = left_side ? 1 : 0;
      // exit_surf.overlap   = framedsurf.fOverlapping ? 1 : 0;
      // exit_surf.common_id = isurf;
      // if (framedsurf.fSceneCS) {
      //   exit_surf.common_id = framedsurf.fSceneCS;
      //   exit_surf.left_side = framedsurf.fSceneCSind > 0 ? true : false;
      //   exit_surf.frame_id  = vecCore::math::Abs(framedsurf.fSceneCSind) - 1;
      //   // not sure if the ovelap info is connected to the scene frame or parent scene frame
      // }

      // Set top exit state to default state, this could have previously been set to a different state by a exiting
      // surface that is further away than this entering surface
      top_exit_state = surf.fDefaultState;

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
        relocated_left_side    = framedsurf.fSceneCSind > 0 ? true : false;
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
    if (!relocated) {
      // do the relocation on the entering side
      auto const &surf_check = surfdata.fCommonSurfaces[isurfcross];
      if (recompute_onsurf) {
        // Moving to a parent scene, we need to recompute the local point on surface
        scene_trans.Clear();
        if (exiting)
          exited_state.SceneMatrix(scene_trans);
        else
          out_state.TopMatrix(scene_trans);
        local_scene = scene_trans.Transform(point + distance * direction);
        onsurf      = surfdata.fGlobalTrans[surf_check.fTrans].Transform(local_scene);
      }

      // Parent frames have been already checked if entering
      auto to_check = exiting ? kCheckAll : kCheckChildren;
      auto iframe   = CheckFramesEntering(isurfcross, relocated_left_side, out_state, in_navind, is_scene, to_check,
                                          distance, point, direction, onsurf, surfdata);

      if (iframe < 0) {
        relocated = true;
        // if we crossed a scene we need to set to the default state of the entering common surface on the new scene,
        // otherwise we need to set to the top exit state
        auto top_ind = exited_scene ? surf_check.fDefaultState : top_exit_state;
        if (exiting) {
          if ((top_ind == 0) && (surf_check.GetSceneId() > 0))
            out_state.PopScene();
          else {
            out_state.SetNavIndex(top_ind);
          }
        }
      }

      while (!relocated && iframe >= 0) {
        //  Found a crossed frame
        auto const &surf       = surfdata.fCommonSurfaces[isurfcross];
        auto const &entry_side = relocated_left_side ? surf.fLeftSide : surf.fRightSide;
        auto const &framedsurf = entry_side.GetSurface(iframe, surfdata);
        if (framedsurf.fState) {
          if (surf.IsSceneSurface() && out_state.GetNavIndex() > 0)
            out_state.PushScene(framedsurf.fState);
          else {
            out_state.SetNavIndex(framedsurf.fState);
          }
        }
        // We may have entered a new scene, so we need to search the scene surface
        relocated = true;
        if (framedsurf.fSceneCS > 0) {
          // We have entered a new scene, first get to the TOP_SCENE frame
          isurfcross             = framedsurf.fSceneCS;
          relocated_left_side    = framedsurf.fSceneCSind > 0 ? true : false;
          auto const &scene_surf = surfdata.fCommonSurfaces[isurfcross];
          // If there are only parent frames on the left side, we can complete relocation
          auto const &scene_side = relocated_left_side ? scene_surf.fLeftSide : scene_surf.fRightSide;

          // We need to search the daughter states only, if none we have already the solution
          // If the crossed scene frame is non-embedding, daughter frames will not be parented to it
          // but they need to be checked nonetheless
          if (scene_side.HasChildren()) {
            relocated = false;
            // Recompute onsurf
            scene_trans.Clear();
            out_state.TopMatrix(scene_trans);
            local_scene = scene_trans.Transform(point + distance * direction);
            onsurf      = surfdata.fGlobalTrans[scene_surf.fTrans].Transform(local_scene);
            // Need to check frames up to the index of the first frame pointing to the parent state
            iframe = CheckFramesEntering(isurfcross, relocated_left_side, out_state, in_navind, is_scene,
                                         kCheckChildren, distance, point, direction, onsurf, surfdata);
          }
        }
      }
    } // end relocation
  }
  // Fix the out_state if pointing to a 0 scene
  if (out_state.GetSceneLevel() > 0 && out_state.GetNavIndex() == 0) out_state.PopScene();
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
  if (in_state.GetNavIndex() == 0) return safety;
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
    // auto const &check_side = left_side ? surf.fLeftSide : surf.fRightSide;
    bool can_compute = unplaced.Safety(local, visibility, surfdata, safety_surf, onsurf_crt);
    if (!can_compute && check_both_sides) {
      // Left side already checked, now check right side
      // Note: only one side can have a valid safety
      left_side   = false;
      visibility  = flipped;
      can_compute = unplaced.Safety(local, visibility, surfdata, safety_surf, onsurf_crt);
    }

    if (!can_compute || safety_surf < -vecgeom::kToleranceDist<Real_t> || safety_surf >= safety) continue;

    // This is an exiting surface for in_state
    // Only check the frame of the current state on this surface

    // Get the index of the framed surface on the exit side. This assumes a SINGLE frame inprint
    // coming from a touchable on any common surface.
    auto const &exit_side  = left_side ? surf.fLeftSide : surf.fRightSide;
    int frameind           = cand.fFrameInd[icand]; // index of framed surface on the side
    auto const &framedsurf = exit_side.GetSurface(frameind, surfdata);
    // Check if the exited frame safety is needed at all
    auto safetyFrame = safety_surf;
    // We need to compute also the safety of the projection of the point on surface to the frame
    safetyFrame = framedsurf.SafetyFrame(onsurf_crt, safety_surf, surfdata, validSafety);
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
    int isurf            = vecCore::math::Abs(cand[icand]);
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
    bool can_compute = unplaced.Safety(local, visibility, surfdata, safety_surf, onsurf_crt);

    if (!can_compute && check_both_sides) {
      // Left side already checked, now check right side
      // Note: only one side can have a valid safety
      left_side   = false;
      visibility  = !flipped;
      can_compute = unplaced.Safety(local, visibility, surfdata, safety_surf, onsurf_crt);
    }
    if (!can_compute || safety_surf < -vecgeom::kToleranceDist<Real_t> || safety_surf >= safety) continue;

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
