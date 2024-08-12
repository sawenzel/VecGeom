#ifndef VECGEOM_SURFACE_NAVIGATOR_H_
#define VECGEOM_SURFACE_NAVIGATOR_H_

#include <VecGeom/management/Logger.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/LogicEvaluator.h>
#include <VecGeom/surfaces/BVHSurfNavigator.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/base/Algorithms.h>

#include <VecGeom/volumes/utilities/VolumeUtilities.h>
#include <VecGeom/base/BVH.h>

#include <VecGeom/surfaces/BVHSurfNavigator.h>

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
                                                                   const int logic_id   = vecgeom::kMaximumInt,
                                                                   const bool is_inside = 0)
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
                                                              SurfData<Real_t> const &surfdata,
                                                              const int logic_id   = vecgeom::kMaximumInt,
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
    auto pushedPoint = std::is_same<Real_t, double>::value
                           ? point + (distance + kPushDistance) * direction
                           : point + (distance + vecgeom::kRelTolerance<Real_t>(distance + point.Mag())) * direction;
    // The logic for the frame that is exited can be set to false without a numerical check
    auto inside = LogicInside(pushedPoint, exited_state, surfdata, framedsurf.fLogicId, false);
    if (inside) inframe = false;
  }
  return inframe;
}

/// @brief Find the topmost exited frame on the common surface pointed by crossed_surf
/// @param crossed_surf Locator of the first frame to be checked
/// @param inframe The first checked frame is known to be hit
/// @param point Global point
/// @param direction Global direction
/// @param distance Distance from global point to the surface
/// @param onsurf Point on the CS, in the surface reference frame
/// @param surfdata Surface data storage
/// @param exiting_scene Input: A scene is exited, false if not known
/// @param exiting_scene Output: The exited frame has a TOP_SCENE state
/// @param surf_index Output: Surface index of the topmost exited frame
/// @param crossed_surf Output: Locator of the crossed framed surface, including the state after exiting
/// @return A frame was exited
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool CheckFramesExiting(FSlocator &crossed_surf, bool frame_hit, Vector3D<Real_t> const &point,
                                                Vector3D<Real_t> const &direction, Real_t distance,
                                                Vector3D<Real_t> const &onsurf, SurfData<Real_t> const &surfdata,
                                                bool &exiting_scene, int &surf_index)
{
  constexpr NavIndex_t kInvalidState = NavIndex_t(-1);

  int isurf                 = crossed_surf.GetCSindex();
  auto const &surf          = surfdata.fCommonSurfaces[isurf];
  bool left_side            = crossed_surf.IsLeftSide();
  bool is_scene_surface     = surf.IsSceneSurface();
  auto const state          = crossed_surf.state;
  auto in_navind            = state.GetNavIndex();
  NavIndex_t top_exit_state = NavIndex_t(-1);
  // We need to check frame intersection
  // This is an exiting surface for in_state
  // First check the frame of the current state on this surface
  auto const &exit_side = left_side ? surf.fLeftSide : surf.fRightSide;
  // Get the index of the first framed surface on the exit side.
  int frameind_start = crossed_surf.frame_id;
  // If the current touchable is exited on this surface, it MUST be through the inside of the
  // corresponding frames. Loop all frames coming from the same touchable
  bool inframe   = frame_hit;
  int parent_ind = -1;
  bool embedded  = true;

  NavIndex_t last_bool_state = kInvalidState; // last checked Boolean state exited
  auto exited_state          = state;

  auto onsurf_crt = onsurf;
  if (exiting_scene) {
    // Moving to a parent scene, we need to recompute the local point on surface
    vecgeom::Transformation3DMP<Real_t> scene_trans;
    state.SceneMatrix(scene_trans);
    auto local_scene = scene_trans.Transform(point + distance * direction);
    onsurf_crt       = surfdata.fGlobalTrans[surf.fTrans].Transform(local_scene);
  }

  // Adjust the state to reflect the topmost exited frame
  auto setTopExited = [&](FramedSurface const &framed_surf, int ind) {
    surf_index = framed_surf.fSurfIndex;
    parent_ind = framed_surf.fParent;
    framed_surf.GetParentState(top_exit_state);
    exiting_scene = is_scene_surface && (framed_surf.fState == 0);
    embedded      = framed_surf.fEmbedded;
    crossed_surf.Set(isurf, ind, left_side);
  };

  for (SideIterator it(exit_side, onsurf_crt, surfdata, 1, frameind_start); !it.Done(); ++it) {
    auto ind               = it();
    auto const &framedsurf = exit_side.GetSurface(ind, surfdata);
    // The deepest exited frame surface must belong to the input state
    if (!exiting_scene && !inframe && framedsurf.fState != in_navind) continue;
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
        it.SetNextIndex(parent_ind);
        // Not embedded frames must be checked thoroughly
        // Note: frames could be non-embedded because they are extruding overlaps. This can be confirmed if the parent
        // surface is embedding. In that case, the parent is exited as well. For these extruding overlaps the
        // crossed_surf must be set to the highest exited frame for the overlap detection to work
        auto const &parent_framedsurf = exit_side.GetSurface(parent_ind, surfdata);
        if (!embedded && !parent_framedsurf.fEmbedding) break;
        // The frame is embedded in the parent, so the parent is also exited
        setTopExited(parent_framedsurf, parent_ind);
        if (parent_framedsurf.fState)
          exited_state.SetNavIndex(parent_framedsurf.fState);
        else
          exited_state.PopScene();
      }
      if (embedded) break;
    }
  }

  /*
    for (int ind = frameind_start; ind < exit_side.fNsurf; ++ind) {
      auto const &framedsurf = exit_side.GetSurface(ind, surfdata);
      // The deepest exited frame surface must belong to the input state
      if (!exiting_scene && !inframe && framedsurf.fState != in_navind) continue;
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
          // Not embedded frames must be checked thoroughly
          // Note: frames could be non-embedded because they are extruding overlaps. This can be confirmed if the parent
          // surface is embedding. In that case, the parent is exited as well. For these extruding overlaps the
          // crossed_surf must be set to the highest exited frame for the overlap detection to work
          auto const &parent_framedsurf = exit_side.GetSurface(parent_ind, surfdata);
          if (!embedded && !parent_framedsurf.fEmbedding) break;
          // The frame is embedded in the parent, so the parent is also exited
          setTopExited(parent_framedsurf, parent_ind);
          if (parent_framedsurf.fState)
            exited_state.SetNavIndex(parent_framedsurf.fState);
          else
            exited_state.PopScene();
        }
        if (embedded) break;
      }
    }
  */
  if (inframe) {
    // backup exited state
    crossed_surf.state = state;
    crossed_surf.state.SetLastExited();
    // set navigation state to top exit state which is either the surf.fDefaultState or a boolean in case an exiting
    // frame was found that does not exit through the default state but stays within the boolean solid
    crossed_surf.state.SetNavIndex(top_exit_state);
    crossed_surf.state.SetBoundaryState(true);
  }
  return inframe;
}

/// @brief Find the deepest hit child frame knowing that the parent is also hit
/// @param side Side to search for hit frames
/// @param in_state Initial state of the track
/// @param in_navind Scene navigation index corresponding to the initial state
/// @param is_scene The initial state is in a scene volume
/// @param is_scene_surface The CS is a scene surface
/// @param onsurf_local Point propagated on the CS the side belongs to, in local CS coordinates
/// @param pushed_point Pushed propagated point in global coordinated, for logical frame checks
/// @param surfdata Surface data storage
/// @return Index of frame being hit
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE int FindFrameOnEnteringSide(Side const &side, vecgeom::NavigationState const &in_state,
                                                    NavIndex_t in_navind, bool is_scene, bool is_scene_surface,
                                                    Vector3D<Real_t> const &onsurf_local,
                                                    Vector3D<Real_t> const &pushed_point,
                                                    SurfData<Real_t> const &surfdata, int &ihit)
{
  // Cached last checked Boolean volume and the inside on the pushed point
  bool found      = ihit >= 0;
  int ifound      = ihit;
  bool lastinside = false;
  int lastbool    = -1;
  // Lambda for getting the first frame with same state as the provided one
  auto getFirstParent = [&](int iframe) {
    int ifirst    = iframe;
    auto refState = side.GetSurface(iframe, surfdata).fState;
    for (auto ind = iframe - 1; ind >= 0; --ind) {
      if (side.GetSurface(ind, surfdata).fState == refState)
        ifirst = ind;
      else
        return ifirst;
    }
    return ifirst;
  };

  // Lambda for checking if the pushed point is in the Boolean volume
  auto insideBoolean = [&](FramedSurface const &surf) {
    auto checked_state = in_state;
    if (surf.fState) {
      if (is_scene_surface && checked_state.GetNavIndex() > 0)
        checked_state.PushScene(surf.fState);
      else
        checked_state.SetNavIndex(surf.fState);
    }
    int ivol = checked_state.GetLogicalId();
    if (lastbool == ivol) return lastinside;
    // The logic for the frame that is entered can be set to true without a numerical check
    lastbool   = ivol;
    lastinside = LogicInside(pushed_point, checked_state, surfdata, surf.fLogicId, true);
    return lastinside;
  };

  // loop frames, parents first, starting from the hit parent
  int iparent = ifound;
  if (found) iparent = getFirstParent(ifound);
  int indstart = (iparent >= 0) ? iparent - 1 : side.fNsurf - 1;
  for (SideIterator it(side, onsurf_local, surfdata, -1, indstart); !it.Done(); --it) {
    auto ind               = it();
    auto const &framedsurf = side.GetSurface(ind, surfdata);
    // NeverCheck frames are always hit
    if (framedsurf.fNeverCheck) {
      ihit = ind;
      return ind;
    }
    // Check only direct children if a frame was already found
    if (found && framedsurf.fParent != iparent) continue;
    // Skip same state frames unless this is a side of a scene surface
    if (!is_scene_surface && framedsurf.fState == in_navind) continue;
    // Hitting a topscene frame starting from the same scene must be discarded
    if (is_scene && framedsurf.fState == 0) continue;
    // Skip embedded children if a parent was not yet found,
    // unless the parent is virtual, then we need to check the children since the parent will not be found
    if (!found && framedsurf.fParent >= 0)
      if ((framedsurf.fEmbedded && !(framedsurf.fVirtualParent)) ||
          side.GetSurface(framedsurf.fParent, surfdata).fEmbedding)
        continue;

    //=== Do the real check ===//
    auto inframe = framedsurf.InsideFrame(onsurf_local, surfdata);
    // For Booleans make sure we don't cross a virtual frame
    if (inframe && framedsurf.fLogicId) inframe = insideBoolean(framedsurf);
    if (inframe) {
      found   = true;
      ifound  = ind;
      iparent = getFirstParent(ifound);
      // Set index of the first frame being hit
      if (ihit < 0) ihit = ind;
    }
  }

  /*
    for (auto ind = indstart; ind >= 0; --ind) {
      auto const &framedsurf = side.GetSurface(ind, surfdata);
      // NeverCheck frames are always hit
      if (framedsurf.fNeverCheck) {
        ihit = ind;
        return ind;
      }
      // Check only direct children if a frame was already found
      if (found && framedsurf.fParent != iparent) continue;
      // Skip same state frames unless this is a side of a scene surface
      if (!is_scene_surface && framedsurf.fState == in_navind) continue;
      // Hitting a topscene frame starting from the same scene must be discarded
      if (is_scene && framedsurf.fState == 0) continue;
      // Skip embedded children if a parent was not yet found,
      // unless the parent is virtual, then we need to check the children since the parent will not be found
      if (!found && framedsurf.fParent >= 0)
        if ((framedsurf.fEmbedded && !(framedsurf.fVirtualParent)) ||
            side.GetSurface(framedsurf.fParent, surfdata).fEmbedding)
          continue;

      //=== Do the real check ===//
      auto inframe = framedsurf.InsideFrame(onsurf_local, surfdata);
      // For Booleans make sure we don't cross a virtual frame
      if (inframe && framedsurf.fLogicId) inframe = insideBoolean(framedsurf);
      if (inframe) {
        found   = true;
        ifound  = ind;
        iparent = getFirstParent(ifound);
        // Set index of the first frame being hit
        if (ihit < 0) ihit = ind;
      }
    }
  */
  return ifound;
}

/// @brief Computes isotropic safety for the logic expression of a volume
/// @param localpoint Point in local volume coordinates
/// @param lvol_id LogicalVolume id
/// @param surfdata
/// @return isotropic safety value
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Real_t LocalLogicSafety(vecgeom::Vector3D<Real_t> const &localpoint,
                                                                     bool exiting, int lvol_id,
                                                                     SurfData<Real_t> const &surfdata,
                                                                     Real_t safe_max) // Temporarily non default
{
  auto const &logic = surfdata.fShells[lvol_id].fLogic;
  Real_t safety     = EvaluateSafety(localpoint, lvol_id, exiting, logic, surfdata, safe_max);
  return safety;
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
/// @param crossed_surf FSLocator holding the highest exited framed surface
/// @return Placed volume pointer containing the point
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool VolumeHasCommonSurface(vecgeom::NavigationState &path, FSlocator const &crossed_surf)
{

  // always included in outside
  if (path.GetNavIndex() == 0) return false;
  // for extruding overlaps no common surface needs to be excluded
  if (crossed_surf.GetFSindex() == -1) return false;

  auto const &surfdata  = SurfData<Real_t>::Instance();
  auto const &surf      = surfdata.fCommonSurfaces[crossed_surf.GetCSindex()];
  auto const &exit_side = crossed_surf.IsLeftSide() ? surf.fLeftSide : surf.fRightSide;

  auto state_id = path.GetNavIndex();

  // non-embedded surfaces cannot be excluded based on this approach, since this could exclude the correct parent that
  // is entered through the non-embedded part of the frame.
  if (!surfdata.IsFramedSurfaceEmbedded(crossed_surf) && surfdata.FramedSurfaceParentInd(crossed_surf) >= 0)
    return false;

  // loop over all framed surfaces to see if booleans are on the common surface
  // booleans can have the same state while having a different framed surface, so they need a full inside check
  // for (int isurf = 0; isurf < exit_side.fNsurf; isurf++) {
  //   if (surfdata.fFramedSurf[exit_side.fSurfaces[isurf]].fLogicId) return false;
  // }

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
/// @param direction Direction in local volume coordinates
/// @param path Path pointing to the top volume to check
/// @param crossed_surf FSLocator holding the highest exited framed surface
/// @param distance distance of the last step
/// @return Placed volume pointer containing the point
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE vecgeom::VPlacedVolume const *ReLocatePointIn(vecgeom::NavigationState &starting_path,
                                                                      vecgeom::Vector3D<Real_t> const &point,
                                                                      vecgeom::Vector3D<Real_t> const &direction,
                                                                      vecgeom::NavigationState &path,
                                                                      FSlocator const &crossed_surf, Real_t distance)
{
  using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const *;
  auto const &surfdata     = SurfData<Real_t>::Instance();

  // set path to be starting path to check for daughters
  path                             = starting_path;
  VPlacedVolumePtr_t currentvolume = starting_path.Top();
  VPlacedVolumePtr_t prev_volume   = path.GetLastExited();

  constexpr Real_t kPushDistance = 1000 * vecgeom::kToleranceDist<Real_t>;
  bool is_boolean                = false;
  bool is_self_entering = false; // this flag is intended for booleans that we start in and don't exit because we cross
                                 // a virtual surface within the boolean
  // in case of 0 steps, a push is needed to exclude the previously exited volume
  bool is_zero_step = distance < 10000 * vecgeom::kToleranceDist<Real_t>;
  auto final_point  = is_zero_step ? point + kPushDistance * direction : point;

  int logic_id_exit = vecgeom::kMaximumInt;

  // first, check daughter volumes of the current path to find if the point lies within any of the daughters
  bool godeeper;
  bool inside_daughter = false;
  if (crossed_surf.GetFSindex() != -1) { // if not exiting surface was found, we don't need to check for children
    do {
      godeeper = false;
      for (auto *daughter : currentvolume->GetDaughters()) {
        is_boolean = daughter->GetUnplacedVolume()->IsBoolean();
        // if (daughter == prev_volume && !(daughter->GetUnplacedVolume()->IsBoolean())) {
        //   // Only exclude the placed volume once since we could enter it again via a
        //   // different volume history.
        //   prev_volume = nullptr;
        //   continue;
        // }
        if (is_boolean) {
          final_point = point + kPushDistance * direction;
          // logic_id_exit = surfdata.FramedSurfaceLogicId(crossed_surf);
        }
        path.Push(daughter);
        bool inside = LogicInside(final_point, path, surfdata); //, logic_id_exit, false);
        final_point = is_zero_step ? point + kPushDistance * direction : point;
        //        logic_id_exit = -1;

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

  if (crossed_surf.GetFSindex() != -1 && crossed_surf.GetCSindex() != 0) {

    // reset to highest exited state
    path = crossed_surf.state;
    // set to the state of the exited surface itself
    path.SetNavIndex(surfdata.FramedSurfaceState(crossed_surf));
  } else {
    path = starting_path;
  }

  // we need to check if the exiting framed surface belongs to a boolean.
  // in case of a boolean, an exiting surface does not necessarily mean that the full boolean is exited,
  // thus, in that case we should not go up in the path before searching in the boolean itself
  // if (surfdata.FramedSurfaceLogicId(crossed_surf)) {
  //   auto pushedPoint = point + kPushDistance * direction;
  //   vecgeom::NavigationState boolean_state;
  //   boolean_state = path;
  //   in_boolean    = LogicInside(pushedPoint, boolean_state, surfdata, 0, false);
  // }

  currentvolume = path.Top();
  assert(currentvolume != nullptr &&
         " currentvolume is nullptr in overlap detection! Most likely due to incorrect path!");
  is_boolean = currentvolume->GetUnplacedVolume()->IsBoolean();

  if (!is_boolean) {
    // exclude volume of the highest parent of the exited framed surface
    if (path.GetNavIndex() > 1) path.SetLastExited();
    prev_volume = path.GetLastExited();
    // navigate one level higher to search for inside
    if (path.GetNavIndex() > 1) path.Pop(); // go one level higher, unless we are in the top volume
  } else {
    prev_volume      = starting_path.Top();
    is_self_entering = true;
  }

  currentvolume = path.Top();
  assert(currentvolume != nullptr && " currentvolume is nullptr in overlap detection! This might due to incorrect path "
                                     "or due to a surface previously being falsely flagged as overlapping!");
  is_boolean = currentvolume->GetUnplacedVolume()->IsBoolean();

  // check whether the point is in the parent volume, otherwise go higher until it is found
  bool gohigher = false;

  bool inside;
  do {
    gohigher     = false;
    auto same_cs = VolumeHasCommonSurface<Real_t>(path, crossed_surf);
    if (same_cs && !is_boolean) {
      inside = false;
    } else {

      // if (vecgeom::BooleanHelper::GetBooleanStruct(currentvolume->GetUnplacedVolume())) {
      if (currentvolume->GetUnplacedVolume()->IsBoolean()) {
        final_point   = point + kPushDistance * direction;
        logic_id_exit = surfdata.FramedSurfaceLogicId(crossed_surf);
      }

      inside = LogicInside(final_point, path, surfdata, logic_id_exit, false);
      // in the case of surfaces that are closer than the push distance together, it can happen that a boolean is exited
      // into another boolean. however, the other boolean extents only with a very small distance into the next one, so
      // a push would mark this as false, leading to an incorrect overlap therefore, we need to check whether the point
      // is with and without a push inside the next boolean. however, we should not do this check when we check whether
      // we are staying within the same boolean because without the push this would always return true, although we are
      // exiting the boolean
      if (currentvolume->GetUnplacedVolume()->IsBoolean() && !is_self_entering)
        inside |= LogicInside(point, path, surfdata, logic_id_exit, false);
      final_point   = is_zero_step ? point + kPushDistance * direction : point;
      logic_id_exit = vecgeom::kMaximumInt;
    }

    if (inside == false) {
      prev_volume = currentvolume;
      path.Pop();
      currentvolume = path.Top();
      assert(
          currentvolume != nullptr &&
          " currentvolume is nullptr in overlap detection! That means some inside call failed or was falsely excluded");
      gohigher = true;
    }
  } while (gohigher);

  do {
    godeeper = false;
    for (auto *daughter : currentvolume->GetDaughters()) {
      is_boolean = daughter->GetUnplacedVolume()->IsBoolean();
      if (daughter == prev_volume && !is_boolean) {
        // Only exclude the placed volume once since we could enter it again via a
        // different volume history.
        prev_volume = nullptr;
        continue;
      }
      path.Push(daughter);
      auto same_cs = VolumeHasCommonSurface<Real_t>(path, crossed_surf);
      if (same_cs && !is_boolean) {
        inside = false;
      } else {

        // if (vecgeom::BooleanHelper::GetBooleanStruct(currentvolume->GetUnplacedVolume())) {
        if (is_boolean) {
          final_point = point + kPushDistance * direction;
          if (daughter == prev_volume) {
            logic_id_exit = surfdata.FramedSurfaceLogicId(crossed_surf);
          }
        }

        inside        = LogicInside(final_point, path, surfdata, logic_id_exit, false);
        final_point   = is_zero_step ? point + kPushDistance * direction : point;
        logic_id_exit = -1;
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

/// @brief Find the deepest hit frame being entered on a CS side
/// @param hit_frame Must contain the input state, candidate CS index and side, initial hit frame if any
/// @param point Point in global coordinates
/// @param direction Direction in global coordinates
/// @param hit_dist Hit distance to the unplaced surface
/// @param onsurf_local Point propagated on the CS the side belongs to, in local CS coordinates
/// @param out_frame Locator for the final hit frame
/// @return A frame was hit flag. Fills hit_frame.frame_id and out_frame, pointing to the final state crossed into
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool EnterCS(FSlocator &hit_frame, Vector3D<Real_t> const &point,
                                     Vector3D<Real_t> const &direction, Real_t hit_dist,
                                     Vector3D<Real_t> const &onsurf_local, FSlocator &out_frame)
{
  constexpr Real_t kPushDistance = 1000 * vecgeom::kTolerance;
  auto const &surfdata           = SurfData<Real_t>::Instance();
  auto const &in_state           = hit_frame.state;
  unsigned short scene_id = 0, newscene_id = 0;
  bool is_scene            = in_state.GetSceneId(scene_id, newscene_id);
  auto in_navind           = in_state.GetNavIndex();
  auto pushed_point        = std::is_same<Real_t, double>::value
                                 ? point + (hit_dist + kPushDistance) * direction
                                 : point + (hit_dist + vecgeom::kRelTolerance<Real_t>(hit_dist + point.Mag())) * direction;
  auto const &surf_crossed = surfdata.GetCommonSurface(hit_frame);
  auto const &enter_side   = surfdata.GetSide(hit_frame);
  int iframe = FindFrameOnEnteringSide(enter_side, in_state, in_navind, is_scene, surf_crossed.IsSceneSurface(),
                                       onsurf_local, pushed_point, surfdata, hit_frame.frame_id);
  out_frame.Set(hit_frame.GetCSindex(), iframe, hit_frame.IsLeftSide());
  out_frame.state = in_state;
  if (iframe < 0) return false;
  while (iframe >= 0) {
    auto const &framedsurf = surfdata.GetFramedSurface(out_frame);
    // Update out_frame to point to the deppest frame being hit
    out_frame.state.SetLastExited();
    out_frame.state.SetBoundaryState(true);
    if (is_scene && out_frame.state.GetNavIndex() > 0)
      // Entering from a scene volume must push the scene (if state not already on the scene)
      out_frame.state.PushScene(framedsurf.fState);
    else
      // Normal entering within the scene
      out_frame.state.SetNavIndex(framedsurf.fState);

    iframe = -1;
    // If entering a scene volume, we need to scan the entered portal CS
    if (framedsurf.fSceneCS > 0) {
      // A top frame was hit on the portal
      out_frame.Set(framedsurf.fSceneCS, vecCore::math::Abs(framedsurf.fSceneCSind) - 1, framedsurf.fSceneCSind > 0);
      auto const &portal      = surfdata.GetCommonSurface(out_frame);
      auto const &portal_side = surfdata.GetSide(out_frame);
      // if the portal has no children, we are already in the final state, entering a scene volume
      if (!portal_side.HasChildren()) return true;
      // We need to do relocation on the portal, after recomputing onsurf
      vecgeom::Transformation3DMP<Real_t> scene_trans;
      // We are in a scene volume, so use TopMatrix
      out_frame.state.TopMatrix(scene_trans);
      auto local_scene = scene_trans.Transform(point + hit_dist * direction);
      auto onsurf      = surfdata.fGlobalTrans[portal.fTrans].Transform(local_scene);
      // Is this a new scene
      is_scene = out_frame.state.GetSceneId(scene_id, newscene_id);
      iframe   = FindFrameOnEnteringSide(portal_side, out_frame.state, out_frame.state.GetNavIndex(), is_scene,
                                         portal.IsSceneSurface(), onsurf, pushed_point, surfdata, out_frame.frame_id);
      if (iframe == out_frame.frame_id) return true;
      // A daughter frame was found on the portal, so continue the loop
      out_frame.frame_id = iframe;
    }
  }
  return true;
}

/// @brief Find the deepest hit frame being entered on a CS side
/// @param hit_frame Must contain the input state, candidate CS index and side, initial hit frame if any
/// @param point Point in global coordinates
/// @param direction Direction in global coordinates
/// @param hit_dist Hit distance to the unplaced surface
/// @param onsurf_local Point propagated on the CS the side belongs to, in local CS coordinates
/// @param out_frame Locator for the final hit frame
/// @return Fills hit_frame.frame_id and out_frame, pointing to the final state crossed into
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool ExitCS(FSlocator &hit_frame, bool is_hit, Vector3D<Real_t> const &point,
                                    Vector3D<Real_t> const &direction, Real_t hit_dist,
                                    Vector3D<Real_t> const &onsurf_local, FSlocator &hit_FS, FSlocator &out_frame)
{
  auto const &surfdata = SurfData<Real_t>::Instance();
  int surf_index       = 0;
  auto exiting_scene   = false;
  auto inframe         = CheckFramesExiting(hit_frame, is_hit, point, direction, hit_dist, onsurf_local, surfdata,
                                            exiting_scene, surf_index);
  if (!inframe) return false;
  // the current state is correctly exited, so there is a transition on this surface
  auto &out_state          = out_frame.state;
  hit_FS                   = hit_frame;
  out_state                = hit_FS.state;
  auto isurfcross          = hit_frame.GetCSindex();
  auto relocated_left_side = !hit_frame.IsLeftSide(); // opposite to exiting side
  auto onsurf              = onsurf_local;
  // Check if a scene surface was crossed
  while (exiting_scene) {
    out_state.PopScene();
    // Convert point in scene coordinates
    vecgeom::Transformation3DMP<Real_t> scene_trans;
    out_state.SceneMatrix(scene_trans);
    // Get the exit frame locator in the parent scene
    surfdata.SceneToTouchableLocator(out_state, surf_index, hit_FS);
    // Find the topmost exit frame on the portal
    hit_FS.state          = out_state;
    auto local_propagated = scene_trans.Transform(point + hit_dist * direction);
    inframe               = CheckFramesExiting(hit_FS, /*frame_hit=*/true, point, direction, hit_dist, onsurf, surfdata,
                                               exiting_scene, surf_index);
    isurfcross            = hit_FS.GetCSindex();
    relocated_left_side   = !hit_FS.IsLeftSide();
    out_state             = hit_FS.state;
    // Moving to a parent scene, we need to recompute the local point on surface
    auto const &surf_crossed = surfdata.GetCommonSurface(hit_FS);
    onsurf                   = surfdata.fGlobalTrans[surf_crossed.fTrans].Transform(local_propagated);
  }
  // Relocate on the other side of the CS
  hit_frame.Set(isurfcross, -1, relocated_left_side);
  // If nothing on the other side no need to relocate
  if (surfdata.GetSide(hit_frame).fNsurf == 0) return true;
  hit_frame.state = out_state;
  // Seek and cross entering frames
  EnterCS(hit_frame, point, direction, hit_dist, onsurf, out_frame);
  return true;
}

/// @brief Compute the distance to a local framed surface in case the frame is hit
/// @tparam Real_t Precision type
/// @param point_volume Point in volume coordinates
/// @param direction_volume Direction in volume coordinates
/// @param volId Id of the volume to which the framed surface belongs
/// @param surfdata Surface data
/// @param framedsurf Framed surface to be checked
/// @param exiting The surface should be an exiting one
/// @param surfhit Returned validity of the crossing
/// @return Distance to unplaced surface
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t DistanceToLocalFS(vecgeom::Vector3D<Real_t> const &point_volume,
                                                 vecgeom::Vector3D<Real_t> const &direction_volume, int volId,
                                                 SurfData<Real_t> const &surfdata, FramedSurface const &framedsurf,
                                                 bool exiting, bool &surfhit, Real_t &safety)
{
  bool two_solutions             = false;
  constexpr Real_t kPushDistance = 1000 * vecgeom::kToleranceDist<Real_t>;
  // Convert point and direction to surface frame
  auto const &trans         = surfdata.fLocalTrans[framedsurf.fTrans];
  Vector3D<Real_t> local    = trans.Transform(point_volume);
  Vector3D<Real_t> localdir = trans.TransformDirection(direction_volume);
  // Check if the surface is a flipped one
  bool flipped_bool = framedsurf.fLogicId < 0;
  bool visibility   = exiting ^ flipped_bool;
  // Compute distance taking into account the surface visibility (normal orientation)
  Real_t dist = vecgeom::InfinityLength<Real_t>();
  surfhit     = framedsurf.fSurface.Intersect(local, localdir, visibility, surfdata, dist, two_solutions, safety);
  if (!surfhit) return dist;
  // Do the frame intersection using the propagated point on surface
  auto onsurf = local + localdir * dist;
  surfhit     = framedsurf.fFrame.Inside(onsurf, surfdata);
  if (!surfhit) return dist;
  // If the frame belongs to a Boolean, we need to check if the solid is really exited/entered
  if (framedsurf.fLogicId) {
    onsurf = std::is_same<Real_t, double>::value
                 ? point_volume + (dist + kPushDistance) * direction_volume
                 : point_volume + (dist + vecgeom::kRelTolerance<Real_t>(dist + point_volume.Mag())) * direction_volume;
    // The logic for the frame that is entered can be set to true without a numerical check
    auto inside = LogicInsideLocal(onsurf, volId, surfdata, framedsurf.fLogicId, !exiting);
    surfhit     = inside ^ exiting;
  }

  return dist;
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
/// @param dist2 distance to the possible second solution
/// @param local point in common surface coordinate
/// @param localdir direction in common surface coordinate
/// @return Distance to unplaced surface
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t DistanceToUnplaced(vecgeom::Vector3D<Real_t> const &point,
                                                  vecgeom::Vector3D<Real_t> const &direction,
                                                  SurfData<Real_t> const &surfdata, int isurf, char sides, bool exiting,
                                                  bool &left_side, bool &surfhit, vecgeom::Vector3D<Real_t> &onsurf,
                                                  Real_t &dist2, vecgeom::Vector3D<Real_t> &local,
                                                  vecgeom::Vector3D<Real_t> &localdir, Real_t &safety)
{
  constexpr char kLside = 1;
  constexpr char kRside = 2;
  auto const &surf      = surfdata.fCommonSurfaces[isurf];

  // Convert point and direction to surface frame
  auto const &trans = surfdata.fGlobalTrans[surf.fTrans];
  local             = trans.Transform(point);
  localdir          = trans.TransformDirection(direction);

  // Compute distance to surface
  Real_t dist;
  bool flipped  = false;
  auto unplaced = surfdata.GetUnplaced(isurf, flipped);

  left_side             = (sides & kLside) > 0;
  bool check_both_sides = left_side && (sides & kRside) > 0;
  bool visibility       = !exiting ^ left_side ^ flipped;
  bool two_solutions    = false;
  surfhit               = unplaced.Intersect(local, localdir, visibility, surfdata, dist, two_solutions, safety);
  if ((!surfhit && check_both_sides) || (two_solutions && check_both_sides)) {
    // Left side already checked, now check right side
    // Note: only one side can have a valid exiting
    left_side          = false;
    visibility         = !exiting ^ flipped;
    bool surfhit_right = unplaced.Intersect(local, localdir, visibility, surfdata, dist2, two_solutions, safety);
    if (surfhit_right) {
      if (surfhit) {
        // both sides have valid hits, we need to chose the one with the smaller distance
        // such that two valid solutions are returned with dist being the closer one and dist2 the one that is further
        // away if the solution on the right side is closer, use it and store the left side solution as dist2
        if (dist2 < dist) {
          Real_t tmp_dist = dist;
          dist            = dist2;
          dist2           = tmp_dist;
        } else {
          // set left_side back to true if left_side solution is closer
          left_side = true;
        }
      } else {
        // surfhit is false so only the right side hit was valid. Thus, we return only one valid distance dist
        dist  = dist2;
        dist2 = -vecgeom::InfinityLength<Real_t>();
      }
      // since surfhit_right is true, we definitively have one valid hit
      surfhit = surfhit_right;
    } else {
      // surfhit_right is false, therefore the original surfhit, dist and dist2 = -infinity is returned
      left_side = true;
    }
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
/// @param exit_FS CrossedSurface storing common surface id, framed surface id, and side of highest exited framed surface and of last hit framed surface (entered or exited)
/// @param stepmax maximum step
/// @return Distance to next surface.
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t ComputeStepAndHit(vecgeom::Vector3D<Real_t> const &point,
                                                 vecgeom::Vector3D<Real_t> const &direction,
                                                 vecgeom::NavigationState const &in_state,
                                                 vecgeom::NavigationState &out_state, CrossedSurface &exit_FS,
                                                 Real_t stepmax = vecgeom::InfinityLength<Real_t>())
{
  // Get the list of candidate surfaces for in_state
  auto in_navind = in_state.GetNavIndex();
  if (in_navind == 0) return vecgeom::InfinityLength<Real_t>();
  auto const &surfdata    = SurfData<Real_t>::Instance();
  Real_t distance         = stepmax;
  unsigned short scene_id = 0, newscene_id = 0;
  bool is_scene        = in_state.GetSceneId(scene_id, newscene_id);
  auto const &cand     = surfdata.GetCandidates(scene_id, in_state.GetId());
  auto const &new_cand = is_scene ? surfdata.GetCandidates(newscene_id, 0) : cand;
  bool found           = false;
  FSlocator tmp_hit_FS;

  Vector3D<Real_t> onsurf;
  out_state = in_state;
  out_state.SetBoundaryState(false);
  auto skip_surf = exit_FS.hit_surf.GetCSindex();

  // Convert the point and direction to the scene coordinate system
  vecgeom::Transformation3DMP<Real_t> scene_trans;
  in_state.SceneMatrix(scene_trans);
  Vector3D<Real_t> local_scene    = scene_trans.Transform(point);
  Vector3D<Real_t> localdir_scene = scene_trans.TransformDirection(direction);
  Vector3D<Real_t> local;
  Vector3D<Real_t> localdir;

  // First check Exiting candidates
  for (auto icand = 0; icand < cand.fNExiting; ++icand) {
    int isurf  = cand.fCandidates[icand];
    char sides = cand.fSides[icand];
    if (isurf == 0) continue;
    bool left_side, surfhit;
    Vector3D<Real_t> onsurf_crt;
    Real_t safety;
    Real_t dist2 = -vecgeom::InfinityLength<Real_t>(); // possible second solution
    // Compute distance to the unplaced surface
    auto dist = DistanceToUnplaced(local_scene, localdir_scene, surfdata, isurf, sides, /*exiting=*/true, left_side,
                                   surfhit, onsurf_crt, dist2, local, localdir, safety);
    if (!surfhit || (dist < -vecgeom::kToleranceStrict<Real_t> && !(Abs(safety) < vecgeom::kToleranceStrict<Real_t>)) ||
        dist >= distance)
      continue;
    // if (dist < 0) dist = Real_t(0.);

    tmp_hit_FS.Set(isurf, cand.fFrameInd[icand], left_side);
    tmp_hit_FS.state = in_state;

    FSlocator out_frame;
    auto inframe =
        ExitCS(tmp_hit_FS, /*is_hit=*/false, point, direction, dist, onsurf_crt, exit_FS.hit_surf, out_frame);
    exit_FS.exit_surf = exit_FS.hit_surf; // store exiting information
    // if no frame was hit and no second solution exists, continue
    if (!inframe && dist2 < -vecgeom::kToleranceStrict<Real_t>) continue;
    // if no frame was hit, try second solution
    if (!inframe && dist2 > -vecgeom::kToleranceStrict<Real_t>) {
      tmp_hit_FS.Set(isurf, cand.fFrameInd[icand], !left_side);
      onsurf_crt = local + dist2 * localdir;
      inframe = ExitCS(tmp_hit_FS, /*is_hit=*/false, point, direction, dist2, onsurf_crt, exit_FS.hit_surf, out_frame);
      exit_FS.exit_surf = exit_FS.hit_surf;
      // store exiting information
      if (inframe) dist = dist2;
    }
    if (!inframe) continue;
    found     = true;
    distance  = dist;
    out_state = out_frame.state;
    continue; // there may be closer surfaces being crossed
  }
  // If there is no physics step limitation, an exiting surface must be found
  if (!found && stepmax == vecgeom::InfinityLength<Real_t>()) {
    // frame_id = -1 indicates extruding overlap
    exit_FS.Set(0, -1, 0);

    // This can happen if the exit point is outside the mother volume (extrusion)
    // To recover, one can return the mother state as output and a zero distance
    return stepmax;
  }

  // Now check entering candidates
  if (is_scene) {
    // If we are on a scene we need to work in the volume reference
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
    Real_t safety;
    Real_t dist2 = -vecgeom::InfinityLength<Real_t>(); // possible second solution
    // Compute distance to the unplaced surface
    auto dist = DistanceToUnplaced(local_scene, localdir_scene, surfdata, isurf, sides, /*exiting=*/false, left_side,
                                   surfhit, onsurf_crt, dist2, local, localdir, safety);
    if (!surfhit || (dist < -vecgeom::kToleranceStrict<Real_t> && !(Abs(safety) < vecgeom::kToleranceStrict<Real_t>)) ||
        dist >= distance)
      continue;
    // if (dist < 0) dist = Real_t(0.);

    // Temporary ugly solution to avoid self-entering the volume at 0 distance on the same surface
    if (self_entering && vecCore::math::Abs(dist) < vecgeom::kToleranceStrict<Real_t>) continue;

    FSlocator out_frame;
    auto EnterFrameCheck = [&](auto left_side, auto &onsurf, auto dist) -> int {
      auto const &surf = surfdata.fCommonSurfaces[isurf];
      auto &entry_side = left_side ? surf.fLeftSide : surf.fRightSide;
      // first check the extent of the entry side using onsurf
      if (entry_side.HasExtent() && !entry_side.fExtent.Inside(onsurf, surfdata)) return -1;

      // Seek and cross entering frames
      // Bootstrap the temporary frame locator with the hit CS side
      tmp_hit_FS.Set(isurf, -1, left_side);
      tmp_hit_FS.state = in_state;
      if (is_scene && !self_entering) {
        EnterCS(tmp_hit_FS, local_scene, localdir_scene, dist, onsurf_crt, out_frame);
      } else {
        EnterCS(tmp_hit_FS, point, direction, dist, onsurf_crt, out_frame);
      }
      return tmp_hit_FS.frame_id;
    };
    auto iframe = EnterFrameCheck(left_side, onsurf_crt, dist);

    // if frame is not hit and not second solution was found, continue
    if (iframe < 0 && dist2 < -vecgeom::kToleranceStrict<Real_t>) continue;

    // if no frame is hit and second solution is valid, check second solution
    if (iframe < 0 && dist2 > -vecgeom::kToleranceStrict<Real_t> && dist2 < distance) {
      onsurf_crt = local + dist2 * localdir;
      iframe     = EnterFrameCheck(!left_side, onsurf_crt, dist2);
      if (iframe >= 0) dist = dist2;
    }

    if (iframe < 0) continue;
    // We do have the final hit frame as out_frame now
    exit_FS.hit_surf = tmp_hit_FS; // store the entering surface information in hit_FS. Needed for correctly marking
                                   // surfaces as overlapping
    // This surface is certainly hit because the parent frame is hit
    distance = dist;
    // compute exited state
    out_state = out_frame.state;
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
  vecgeom::Transformation3DMP<Real_t> scene_trans;
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
    bool can_compute      = unplaced.Safety(local, visibility, surfdata, safety_surf, onsurf_crt);

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
