#ifndef BVH_SURF_NAVIGATOR_H
#define BVH_SURF_NAVIGATOR_H

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/base/BVH.h>
#include <VecGeom/surfaces/Navigator.h>

namespace vgbrep {
namespace protonav {

// Forward declaration
// This is temporary, just as long as we need to call the BVH from the loop navigator for testing
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t DistanceToLocalFS(vecgeom::Vector3D<Real_t> const &local,
                                                 vecgeom::Vector3D<Real_t> const &localdir, int volId,
                                                 SurfData<Real_t> const &surfdata,
                                                 FramedSurface<Real_t, TransformationMP<Real_t>> const &framedsurf,
                                                 bool exiting, bool &surfhit, Real_t &safety);

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Real_t LocalLogicSafety(vecgeom::Vector3D<Real_t> const &localpoint,
                                                                     bool exiting, int lvol_id,
                                                                     SurfData<Real_t> const &surfdata,
                                                                     Real_t safe_max); // Temporarily non default

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool EnterCS(FSlocator &hit_frame, Vector3D<Real_t> const &point,
                                     Vector3D<Real_t> const &direction, Real_t hit_dist,
                                     Vector3D<Real_t> const &onsurf_local, FSlocator &out_frame);

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool ExitCS(FSlocator &hit_frame, bool is_hit, Vector3D<Real_t> const &point,
                                    Vector3D<Real_t> const &direction, Real_t hit_dist,
                                    Vector3D<Real_t> const &onsurf_local, FSlocator &hit_FS, FSlocator &out_frame);

template <typename Real_t>
class BVHSurfNavigator {
public:
  BVHSurfNavigator()  = default;
  ~BVHSurfNavigator() = default;

  /*
   * @param[in] lv_index Global index of a LogicalVolume
   * @param[in] index Index within the list of visible entering surfaces of the specified LogicalVolume
   * @param[in] localpoint Point in the local coordinates of the LV specified by @lv_index
   * @param[in] localdir Direction in the local coordinates of the LV specified by @lv_index
   * @param[in] step Maximum step length
   * @returns The distance to in to the Framed surface defined by @p lv_index and @p index for the point @p localpoint
   * and direction @p localdir
   */
  VECCORE_ATT_HOST_DEVICE
  static Real_t CandidateDistanceToIn(int lv_index, int index, Vector3D<Real_t> localpoint, Vector3D<Real_t> localdir,
                                      Real_t step)
  {
    auto const &surfdata = vgbrep::SurfData<Real_t>::Instance();
    // Get the shell for this volume
    auto const &shell = surfdata.fShells[lv_index];

    FramedSurface<Real_t, TransformationMP<Real_t>> *framed_surface;
    bool exiting{false};

    // Retrieve the candidate local surface
    // Different treatment for entering and exiting surfaces
    if (index >= shell.fNExitingSurfaces) // Entering surfaces
    {
      int entering_index = index - shell.fNExitingSurfaces;
      framed_surface     = &(surfdata.fLocalSurf[shell.fEnteringSurfaces[entering_index]]);

      // Get the transformation
      auto const &pvol_trans = surfdata.fPVolTrans[shell.fEnteringSurfacesPvolTrans[entering_index]];

      // Transform the points to the pvol frame
      localpoint = pvol_trans.Transform(localpoint);
      localdir   = pvol_trans.TransformDirection(localdir);

      // Update the LV index to that of the daughter
      lv_index = shell.fEnteringSurfacesLvolIds[entering_index];

    } else { // Exiting surfaces
      exiting            = true;
      auto exiting_index = shell.fExitingSurfaces[index];
      framed_surface     = &(surfdata.fLocalSurf[shell.fSurfaces[exiting_index]]);
    }

    // Common part for entering and exiting surfaces

    // Check if we intersect the unplaced and the distance
    Real_t intersect_distance{0};
    bool surfhit{false};
    Real_t safety;
    intersect_distance =
        DistanceToLocalFS(localpoint, localdir, lv_index, surfdata, *framed_surface, exiting, surfhit, safety);
    if (surfhit &&
        (intersect_distance > -vecgeom::kToleranceStrict<Real_t> || Abs(safety) < vecgeom::kToleranceStrict<Real_t>)) {
      return intersect_distance;
    } else {
      return vecgeom::InfinityLength<Real_t>();
    }
  }

  VECCORE_ATT_HOST_DEVICE
  static int GetLogicalId(int lv_index, int surfindex, bool &isBoolean)
  {
    auto const &surfdata = vgbrep::SurfData<Real_t>::Instance();
    auto const &shell    = surfdata.fShells[lv_index];
    if (surfindex >= shell.fNExitingSurfaces) {
      auto entering_index = surfindex - shell.fNExitingSurfaces;
      lv_index            = shell.fEnteringSurfacesLvolIds[entering_index];
      isBoolean           = surfdata.fLocalSurf[shell.fEnteringSurfaces[entering_index]].fLogicId > 0;
      return lv_index;
    }
    isBoolean = surfdata.fLocalSurf[shell.fSurfaces[shell.fExitingSurfaces[surfindex]]].fLogicId > 0;
    return lv_index;
  }

  VECCORE_ATT_HOST_DEVICE
  static bool IsNegated(int lv_index, int surfindex)
  {
    auto const &surfdata = vgbrep::SurfData<Real_t>::Instance();
    auto const &shell    = surfdata.fShells[lv_index];
    if (surfindex >= shell.fNExitingSurfaces)
      return (surfdata.fLocalSurf[shell.fEnteringSurfaces[surfindex - shell.fNExitingSurfaces]].fLogicId < 0);
    else
      return (surfdata.fLocalSurf[shell.fSurfaces[shell.fExitingSurfaces[surfindex]]].fLogicId < 0);
  }

  /*
   * @param[in] lv_index Global index of a LogicalVolume
   * @param[in] index Index within the list of visible entering surfaces of the specified LogicalVolume
   * @param[in] localpoint Point in the local coordinates of the LV specified by @lv_index
   * @param[in] limit Upper limit for the search in the minimization process (e.g. previous computed safety candidate)
   * @returns The safety to in to the Framed surface defined by @p lv_index and @p index for the point @p localpoint
   */
  VECCORE_ATT_HOST_DEVICE
  static Real_t CandidateSafetyToIn(int lv_index, int index, Vector3D<Real_t> localpoint,
                                    Real_t limit = vecgeom::InfinityLength<Real_t>())
  {
    auto const &surfdata = vgbrep::SurfData<Real_t>::Instance();
    // Get the shell for this volume
    auto const &shell = surfdata.fShells[lv_index];

    Vector3D<Real_t> surface_point;
    Vector3D<Real_t> surface_dir;
    FramedSurface<Real_t, TransformationMP<Real_t>> *framed_surface;
    bool exiting{false};

    // Retrieve the candidate local surface
    // Different treatment for entering and exiting surfaces
    if (index >= shell.fNExitingSurfaces) // Entering surfaces
    {
      int entering_index = index - shell.fNExitingSurfaces;
      framed_surface     = &(surfdata.fLocalSurf[shell.fEnteringSurfaces[entering_index]]);
      // Update the LV index to that of the daughter
      lv_index = shell.fEnteringSurfacesLvolIds[entering_index];

      // Get the transformation
      auto const &pvol_trans = surfdata.fPVolTrans[shell.fEnteringSurfacesPvolTrans[entering_index]];
      // Compute the transformation to the surface reference frame
      auto const &surf_trans = framed_surface->fTrans;
      auto volume_trans      = surf_trans * pvol_trans;

      // Convert the point to surface coordinates
      // surface_point = volume_trans.Transform(localpoint);
      surface_point = pvol_trans.Transform(localpoint);
      surface_point = surf_trans.Transform(surface_point);

    } else { // Exiting surfaces
      exiting            = true;
      auto exiting_index = shell.fExitingSurfaces[index];
      framed_surface     = &(surfdata.fLocalSurf[shell.fSurfaces[exiting_index]]);
      // Get the local transformation of the surface
      TransformationMP<Real_t> &local_trans = framed_surface->fTrans;
      // In the case of exiting surfaces we only need to apply this transformation
      surface_point = local_trans.Transform(localpoint);
    }

    // Get the unplaced
    UnplacedSurface<Real_t> &unplaced_surface = framed_surface->fSurface;

    // Compute the safety
    // First check coming from left side
    Vector3D<Real_t> onsurf_crt;
    Real_t safety_surf{0}, safety_frame{0};
    bool valid_safety{true};
    bool can_compute{false};

    bool flipped = framed_surface->fLogicId < 0;
    can_compute  = unplaced_surface.Safety(surface_point, exiting ^ flipped, surfdata, safety_surf, onsurf_crt);

    if (!can_compute || safety_surf >= limit) return safety_surf;

    // Now compute the safety from the projection of the point on the surface to the frame
    safety_frame = safety_surf;
    // This function returns either the maximum of safety_surf and safety_frame, or the accurate safety
    // computed using both
    if (!framed_surface->fNeverCheck)
      safety_frame = framed_surface->LocalSafetyFrame(onsurf_crt, safety_surf, surfdata, valid_safety);

    if (valid_safety) return safety_frame;
    return limit;
  }

  /*
   * Used by the BVH to determine if it needs to skip checking a framed surface. The global index of the surface
   * defined by @p lv_index and @p index can only be accessed from the navigator
   * @param[in] lv_index Global index of a LogicalVolume
   * @param[in] index Index within the list of visible entering surfaces of the specified LogicalVolume
   * @param[in] global_id Global id of a FramedSurface
   * @returns Whether the global id of the FramedSurface defined by @p lv_index and @p index is the same as @p global_id
   */
  VECCORE_ATT_HOST_DEVICE
  static VECGEOM_FORCE_INLINE bool SkipItem(int lv_index, int index, long const global_id)
  {
    // There are no global IDs for framed surfaces, return false
    return false;
  }

  template <typename BVH_t>
  VECCORE_ATT_HOST_DEVICE static long TestBVHCheckDaughterIntersections(BVH_t &bvh, Vector3D<Real_t> &localpoint,
                                                                        Vector3D<Real_t> &localdir, double &bvhstep)
  {
    long hitcandidate_index = -1;
    long last_exited_id     = -1;
    bvh.template CheckDaughterIntersections<BVHSurfNavigator>(localpoint, localdir, bvhstep, last_exited_id,
                                                              hitcandidate_index);
    return hitcandidate_index;
  }

  template <typename BVH_t>
  VECCORE_ATT_HOST_DEVICE static double TestBVHComputeSafety(BVH_t &bvh, Vector3D<Real_t> &localpoint, Real_t safety)
  {
    return bvh.template ComputeSafety<BVHSurfNavigator>(localpoint, safety);
  }

  VECCORE_ATT_HOST_DEVICE
  static Precision ComputeSafety(Vector3D<Precision> const &point_i, vecgeom::NavigationState const &in_state,
                                 Precision limit = vecgeom::InfinityLength<Precision>())
  {
    auto in_navind = in_state.GetNavIndex();
    if (in_navind == 0) return vecgeom::InfinityLength<Precision>();

    // Get the SurfData instance
    auto const &surfdata = SurfData<Real_t>::Instance();
    vecgeom::Vector3D<Real_t> point(point_i);

    // Get the shell
    auto lv_index     = in_state.GetLogicalId();
    auto const &shell = surfdata.fShells[lv_index];

    // Get the BVH
    auto &bvh = surfdata.fBVH[shell.fBVH];

    vecgeom::Transformation3D lv_trans;
    in_state.TopMatrix(lv_trans);
    auto localpoint = lv_trans.Transform(point);

    auto safety = bvh.template ComputeSafety<BVHSurfNavigator<Real_t>>(localpoint, limit);
    // Safety is rounded from float, so round down with float relative tolerance
    safety = vecgeom::MakeMinusTolerantRel<float>(safety);
    return static_cast<Real_t>(safety);
  }

  /*
   * @param[in] aLVIndex Global index of a LogicalVolume
   * @param[in] index Index within the list of daughters of the specified LogicalVolume
   * @returns The global id of the PlacedVolume defined by @p aLVIndex and @p index
   */
  VECCORE_ATT_HOST_DEVICE
  static uint ItemId(int lv_index, int index)
  {
    auto const &surfdata = vgbrep::SurfData<Real_t>::Instance();
    // Get the shell for this volume
    auto const &shell = surfdata.fShells[lv_index];
    return shell.fDaughterPvolIds[index];
  }

  /*
   * @param[in] lv_index Global index of a LogicalVolume
   * @param[in] index Index within the list of daughters of the specified LogicalVolume
   * @param[in] localpoint Point in the local coordinates of the LV specified by @aLVIndex
   * @returns Whether @localpoint falls within the PlacedVolume defined by @p aLVIndex and @p index
   */
  VECCORE_ATT_HOST_DEVICE
  static bool CandidateContains(int lv_index, int index, Vector3D<Real_t> const &localpoint,
                                vecgeom::NavigationState &path)
  {
    // Get the SurfData instance
    auto const &surfdata = SurfData<Real_t>::Instance();
    // Get the shell
    auto const &shell = surfdata.fShells[lv_index];

    Vector3D<Real_t> daughterlocalpoint;
    auto const &trans = surfdata.fPVolTrans[shell.fDaughterPvolTrans[index]];
    trans.Transform(localpoint, daughterlocalpoint);

    // Push to path
    path.PushDaughter(index);

    // Call LogicInside
    auto inside = vgbrep::protonav::LogicInsideLocal(daughterlocalpoint, path.GetLogicalId(), surfdata);

    // Pop from the path
    path.Pop();

    return inside;
  };

  VECCORE_ATT_HOST_DEVICE
  static int LocatePointIn(int pvol_id, Vector3D<Precision> const &point_i, vecgeom::NavigationState &path, bool top,
                           int *exclude = nullptr)
  {
    using NavigationState = vecgeom::NavigationState;
    vecgeom::Vector3D<Real_t> point(point_i);

    // Get the SurfData instance
    auto const &surfdata = SurfData<Real_t>::Instance();

    if (top) {
      VECGEOM_ASSERT(pvol_id >= 0);
      auto ivol   = NavigationState::ToPlacedId(pvol_id).fVolume.fId;
      auto inside = vgbrep::protonav::LogicInsideLocal(point, ivol, surfdata);
      if (!inside) return -1;
    }

    path.Push(pvol_id);
    Vector3D<Real_t> currentpoint(point);
    Vector3D<Real_t> daughterlocalpoint;
    int exclude_id  = -1;
    int daughter_id = -1;

    // Get the shell
    auto shell = surfdata.fShells[path.GetLogicalId()]; // Assuming path corresponds to pvol_id

    // Keep going down as long as the current volume has daughters
    while (shell.fNEnteringSurfaces > 0) {
      // Get the BVH
      auto &bvh = surfdata.fBVHSolids[shell.fBVH];

      exclude_id = -1;
      if (exclude != nullptr) {
        exclude_id = *exclude;
      }
      daughter_id = -1;

      // daughter_id will be the global id of the placed volume
      if (!bvh.template LevelLocate<BVHSurfNavigator<Real_t>>(exclude_id, currentpoint, daughter_id, path)) break;

      path.Push(daughter_id);

      // Compute the transformed point
      // TODO: This can be done cheaper if we use the transformation stored in the shell
      // This saves computing the full global transformation in TopMatrix, but for now we don't have
      // the index of the daughter from the BVH
      vecgeom::Transformation3DMP<Real_t> trans;
      path.TopMatrix(trans);
      trans.Transform(point, daughterlocalpoint);

      // And update the current point
      currentpoint = daughterlocalpoint;

      // Update the current volume shell
      shell = surfdata.fShells[path.GetLogicalId()];

      // Only exclude the placed volume once since we could enter it again via a
      // different volume history.
      if (exclude != nullptr) {
        *exclude = -1;
      }
    }

    return path.TopId();
  }

  /// @brief Method computing the distance to the next surface.
  /// @details The method returns the same ouy_state as the in_state, but alters the boundary condition.
  /// @tparam Real_i Input floating point type
  /// @param point_i Global point
  /// @param direction_i Global direction
  /// @param in_state Input navigation state before crossing
  /// @param out_state Output navigation state after crossing
  /// @param hitsurf_index Hit surface index
  /// @param stepmax Maximum allowed step
  /// @return Distance to next surface.
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE static Real_i ComputeStepAndNextSurface(vecgeom::Vector3D<Real_i> const &point_i,
                                                                  vecgeom::Vector3D<Real_i> const &direction_i,
                                                                  vecgeom::NavigationState const &in_state,
                                                                  vecgeom::NavigationState &out_state,
                                                                  long &hitsurf_index,
                                                                  Real_i stepmax = vecgeom::InfinityLength<Real_i>())
  {
    out_state      = in_state;
    auto in_navind = in_state.GetNavIndex();
    // For the moment we only support a starting state within the setup
    if (in_navind == 0) return vecgeom::InfinityLength<Real_i>();
    auto const &surfdata = SurfData<Real_t>::Instance();

    auto ivol = in_state.GetLogicalId();
    auto &bvh = surfdata.fBVH[surfdata.fShells[ivol].fBVH];

    // casting point into Real_t (float in mixed precision) and using Real_t transformation for local points.
    // Since this is for a frame check, float is sufficient.
    vecgeom::Vector3D<Real_t> point(point_i);
    vecgeom::Vector3D<Real_t> direction(direction_i);
    vecgeom::Transformation3DMP<Real_t> lv_trans;
    in_state.TopMatrix(lv_trans);
    auto localpoint = lv_trans.Transform(point);
    auto localdir   = lv_trans.TransformDirection(direction);

    // Draft: potentially one can still transform in double precision and then convert that point to single precision
    // (or even propagate the point and do the conversion later). To be tested in the future
    // if (localpoint.Length2() > Real_t(1e8)) {
    //   vecgeom::Transformation3DMP<Real_i> lv_trans_DP;
    //   in_state.TopMatrix(lv_trans_DP);
    //   // precise point in Real_i (double) since this may be large values where rounding errors propagate to our
    //   distance calculation auto localpoint = lv_trans_DP.Transform(point); auto localdir   =
    //   lv_trans_DP.TransformDirection(direction);

    //   localpoint = {static_cast<Real_t>(localpoint[0]), static_cast<Real_t>(localpoint[1]),
    //   static_cast<Real_t>(localpoint[2])}; localdir = {static_cast<Real_t>(localdir[0]),
    //   static_cast<Real_t>(localdir[1]), static_cast<Real_t>(localdir[2])};
    // }

    auto bvhstep = stepmax;
    // long last_exited_id     = -1;
    bvh.template CheckDaughterIntersections<BVHSurfNavigator>(localpoint, localdir, bvhstep, -1, hitsurf_index);
    // If there is no physics step limitation, a surface must be found
    if (hitsurf_index < 0) {
      // Nothing is hit within the step limit
      out_state.SetBoundaryState(false);
      return stepmax;
    }

    // By default we hit a boundary and exit the in_state volume
    out_state.SetBoundaryState(true);
    out_state.SetLastExited();

    // Return the step to the hit surface
    return bvhstep;
  }

  /// @brief Method that relocates the point after CS traversal
  /// @tparam Real_i Input floating point type
  /// @param point_i Global point
  /// @param direction_i Global direction
  /// @param step Step to the next boundary
  /// @param out_state Input state before crossing and output state after crossing
  /// @param hit_FS Hit surface information
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE static void RelocateToNextVolume(vecgeom::Vector3D<Real_i> const &point_i,
                                                           vecgeom::Vector3D<Real_i> const &direction_i, Real_i step,
                                                           long hitsurf_index, vecgeom::NavigationState &out_state,
                                                           CrossedSurface &hit_FS)
  {
    auto const &surfdata = SurfData<Real_t>::Instance();
    auto const in_state  = out_state;
    // out_state            = in_state;
    Real_t bvhstep_RT = Real_t(step); // cast only once and use in later functions
    vecgeom::Vector3D<Real_t> point(point_i);
    vecgeom::Vector3D<Real_t> direction(direction_i);
    vecgeom::Transformation3DMP<Real_t> scene_trans;
    in_state.SceneMatrix(scene_trans);
    auto local_scene    = scene_trans.Transform(point);
    auto localdir_scene = scene_trans.TransformDirection(direction);
    // Identify the common surface
    auto const &currentShell = surfdata.fShells[in_state.GetLogicalId()];
    Vector3D<Real_t> CS_local, CS_localdir, onsurf_crt;
    if (hitsurf_index < currentShell.fNExitingSurfaces) {
      // If the hit candidate is an exiting surface
      auto exiting_index = currentShell.fExitingSurfaces[hitsurf_index];
      FSlocator hit_FS_tmp;
      surfdata.SceneToTouchableLocator(in_state, exiting_index, hit_FS_tmp);
      FSlocator out_frame;
      hit_FS_tmp.state = in_state;
      // Get the onsurf point in CS coordinates
      auto const &surf     = surfdata.fCommonSurfaces[hit_FS_tmp.GetCSindex()];
      auto const &CS_trans = surf.fTrans;
      CS_local             = CS_trans.Transform(local_scene);
      CS_localdir          = CS_trans.TransformDirection(localdir_scene);
      onsurf_crt           = CS_local + bvhstep_RT * CS_localdir;
      ExitCS(hit_FS_tmp, /*is_hit=*/true, point, direction, bvhstep_RT, onsurf_crt, hit_FS.hit_surf, out_frame);
      hit_FS.exit_surf = hit_FS.hit_surf;
      out_state        = out_frame.state;

    } else {

      auto entering_index        = hitsurf_index - currentShell.fNExitingSurfaces;
      auto local_surface_id      = currentShell.fEnteringSurfaces[entering_index];
      auto const &framed_surface = surfdata.fLocalSurf[local_surface_id];
      auto pvol_id               = currentShell.fEnteringSurfacesPvol[entering_index];
      // Create a copy of the navigation state
      auto pvol_navstate(in_state);
      // Get the navigation state of the daughter
      // pvol_navstate.Push(pvol);
      pvol_navstate.Push(pvol_id);
      // set hit_FS.hit_surf
      surfdata.SceneToTouchableLocator(pvol_navstate, framed_surface.fSurfIndex, hit_FS.hit_surf);
      hit_FS.hit_surf.state = in_state;
      int traversal         = surfdata.GetFramedSurface(hit_FS.hit_surf).fTraversal;
      // Get the onsurf point in CS coordinates
      auto const &surf     = surfdata.fCommonSurfaces[hit_FS.hit_surf.GetCSindex()];
      auto const &CS_trans = surf.fTrans;

      unsigned short scene_id = 0, newscene_id = 0;
      bool is_scene = in_state.GetSceneId(scene_id, newscene_id);

      if (!is_scene) {
        CS_local    = CS_trans.Transform(local_scene);
        CS_localdir = CS_trans.TransformDirection(localdir_scene);
      } else {
        vecgeom::Transformation3DMP<Real_t> lv_trans;
        // local_scene is already transformed by SceneMatrix, therefore now TopInSceneMatrix is sufficient
        in_state.TopInSceneMatrix(lv_trans);
        auto localpoint = lv_trans.Transform(local_scene);
        auto localdir   = lv_trans.TransformDirection(localdir_scene);
        CS_local        = CS_trans.Transform(localpoint);
        CS_localdir     = CS_trans.TransformDirection(localdir);
      }
      onsurf_crt = CS_local + bvhstep_RT * CS_localdir;

      // Seek and cross entering frames
      FSlocator out_frame;
      EnterCS(hit_FS.hit_surf, point, direction, bvhstep_RT, onsurf_crt, out_frame, traversal);
      out_state = out_frame.state;
    }

    // Fix the out_state if pointing to a 0 scene
    if (out_state.GetSceneLevel() > 0 && out_state.GetNavIndex() == 0) out_state.PopScene();
    out_state.SetBoundaryState(true);
  }

  /// @brief Method computing the distance to the next surface and state after crossing it
  /// @tparam Real_i Input floating point type
  /// @param point_i Global point
  /// @param direction_i Global direction
  /// @param in_state Input navigation state before crossing
  /// @param out_state Output navigation state after crossing
  /// @param surfdata Surface data storage
  /// @param exit_surf data container storing the exited common surface, the side, the frame id, and whether there is an overlap
  /// @param stepmax maximum step
  /// @return Distance to next surface.
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE static Real_i ComputeStepAndHit(vecgeom::Vector3D<Real_i> const &point_i,
                                                          vecgeom::Vector3D<Real_i> const &direction_i,
                                                          vecgeom::NavigationState const &in_state,
                                                          vecgeom::NavigationState &out_state, CrossedSurface &hit_FS,
                                                          Real_i stepmax = vecgeom::InfinityLength<Real_i>())
  {
    long hitsurf_index = -1;
    auto bvhstep       = ComputeStepAndNextSurface(point_i, direction_i, in_state, out_state, hitsurf_index, stepmax);
    // If there is no physics step limitation, a surface must be found
    if (hitsurf_index < 0) {
      if (in_state.GetNavIndex() > 0 && stepmax == vecgeom::InfinityLength<Real_i>())
        hit_FS.Set(0, -1, 0); // frame_id = -1 indicates extruding overlap
      // This can happen if the exit point is outside the mother volume (extrusion)
      // To recover, one can return the mother state as output and a zero distance
      return stepmax;
    }
    RelocateToNextVolume(point_i, direction_i, bvhstep, hitsurf_index, out_state, hit_FS);
    return bvhstep;
  }
};

} // namespace protonav
} // namespace vgbrep

#endif