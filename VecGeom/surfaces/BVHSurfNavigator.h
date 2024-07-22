#ifndef BVH_SURF_NAVIGATOR_H
#define BVH_SURF_NAVIGATOR_H

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/surfaces/bvh/BVHsurf.h>
#include <VecGeom/surfaces/Navigator.h>

namespace vgbrep {
namespace protonav {

// Forward declaration
// This is temporary, just as long as we need to call the BVH from the loop navigator for testing
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t DistanceToLocalFS(vecgeom::Vector3D<Real_t> const &local,
                                                 vecgeom::Vector3D<Real_t> const &localdir, int volId,
                                                 SurfData<Real_t> const &surfdata, FramedSurface const &framedsurf,
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
    auto surfdata = vgbrep::SurfData<Real_t>::Instance();
    // Get the shell for this volume
    auto shell = surfdata.fShells[lv_index];

    FramedSurface *framed_surface;
    bool exiting{false};
    const vecgeom::VPlacedVolume *pvol;

    // Retrieve the candidate local surface
    // Different treatment for entering and exiting surfaces
    if (index >= shell.fNExitingSurfaces) // Entering surfaces
    {
      int exiting_index = index - shell.fNExitingSurfaces;
      framed_surface    = &(surfdata.fLocalSurf[shell.fEnteringSurfaces[exiting_index]]);

      // Get the pvol
      pvol = vecgeom::NavigationState::ToPlacedVolume(shell.fEnteringSurfacesPvol[exiting_index]);

      // Get the transformation
      auto pvol_trans = pvol->GetTransformation();

      // Transform the points to the pvol frame
      localpoint = pvol_trans->Transform(localpoint);
      localdir   = pvol_trans->TransformDirection(localdir);

      // Update the LV index to that of the daughter
      lv_index = pvol->GetLogicalVolume()->id();

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
      ;
    }
  }

  /*
   * @param[in] lv_index Global index of a LogicalVolume
   * @param[in] index Index within the list of visible entering surfaces of the specified LogicalVolume
   * @param[in] localpoint Point in the local coordinates of the LV specified by @lv_index
   * @returns The safety to in to the Framed surface defined by @p lv_index and @p index for the point @p localpoint
   */
  VECCORE_ATT_HOST_DEVICE
  static Real_t CandidateSafetyToIn(int lv_index, int index, Vector3D<Real_t> localpoint)
  {
    auto surfdata = vgbrep::SurfData<Real_t>::Instance();
    // Get the shell for this volume
    auto shell = surfdata.fShells[lv_index];

    Vector3D<Real_t> surface_point;
    Vector3D<Real_t> surface_dir;
    FramedSurface *framed_surface;
    const vecgeom::VPlacedVolume *pvol;
    bool exiting{false};

    // Retrieve the candidate local surface
    // Different treatment for entering and exiting surfaces
    if (index >= shell.fNExitingSurfaces) // Entering surfaces
    {
      int entering_index = index - shell.fNExitingSurfaces;
      framed_surface     = &(surfdata.fLocalSurf[shell.fEnteringSurfaces[entering_index]]);
      // Get the pvol
      pvol = vecgeom::NavigationState::ToPlacedVolume(shell.fEnteringSurfacesPvol[entering_index]);
      // Get the transformation
      auto const &pvol_trans = *pvol->GetTransformation();
      // Compute the transformation to the surface reference frame
      auto const &surf_trans = surfdata.fLocalTrans[framed_surface->fTrans];
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
      TransformationMP<Real_t> &local_trans = surfdata.fLocalTrans[framed_surface->fTrans];
      // In the case of exiting surfaces we only need to apply this transformation
      surface_point = local_trans.Transform(localpoint);
    }

    // Get the unplaced
    UnplacedSurface &unplaced_surface = framed_surface->fSurface;

    // Compute the safety
    // First check coming from left side
    Vector3D<Real_t> onsurf_crt;
    Real_t safety{0}, safety_surf{0}, safety_frame{0};
    bool valid_safety{false};
    bool can_compute{false};

    if (exiting) {
      can_compute = unplaced_surface.Safety(surface_point, 1, surfdata, safety_surf, onsurf_crt);
    } else {
      can_compute = unplaced_surface.Safety(surface_point, 0, surfdata, safety_surf, onsurf_crt);
    }

    if (!can_compute || safety_surf < -vecgeom::kToleranceDist<Real_t>) {
      return vecgeom::InfinityLength<Real_t>();
    }

    // Now compute the safety from the projection of the point on the surface to the frame
    safety_frame = safety_surf;
    // This function returns either the maximum of safety_surf and safety_frame, or the accurate safety
    // computed using both
    safety_frame = framed_surface->LocalSafetyFrame(onsurf_crt, safety_surf, surfdata, valid_safety);

    if (valid_safety) {
      // If Boolean surface, compute only once safety for the entire volume shell
      if (framed_surface->fLogicId) {
        Real_t safety_logic =
            LocalLogicSafety(localpoint, exiting, lv_index, surfdata, vecgeom::InfinityLength<Real_t>());
        safety = safety_logic;
      } else {
        safety = safety_frame;
      }
    }

    return safety;
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
                                                                        Vector3D<Real_t> &localdir, Real_t &bvhstep)
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

  /// @brief Method computing the distance to the next surface and state after crossing it
  /// @tparam Real_t Floating point type for the interface and data storage
  /// @param point Global point
  /// @param direction Global direction
  /// @param in_state Input navigation state before crossing
  /// @param out_state Output navigation state after crossing
  /// @param surfdata Surface data storage
  /// @param exit_surf data container storing the exited common surface, the side, the frame id, and whether there is an overlap
  /// @param stepmax maximum step
  /// @return Distance to next surface.
  VECCORE_ATT_HOST_DEVICE static Real_t ComputeStepAndHit(vecgeom::Vector3D<Real_t> const &point,
                                                          vecgeom::Vector3D<Real_t> const &direction,
                                                          vecgeom::NavigationState const &in_state,
                                                          vecgeom::NavigationState &out_state, CrossedSurface &hit_FS,
                                                          Real_t stepmax = vecgeom::InfinityLength<Real_t>())
  {
    auto in_navind = in_state.GetNavIndex();
    if (in_navind == 0) return vecgeom::InfinityLength<Real_t>();

    vecgeom::Transformation3DMP<Real_t> scene_trans;
    in_state.SceneMatrix(scene_trans);
    Vector3D<Real_t> local_scene    = scene_trans.Transform(point);
    Vector3D<Real_t> localdir_scene = scene_trans.TransformDirection(direction);

    auto const &surfdata = SurfData<Real_t>::Instance();

    // printf("LVol ID: %d\n", in_state.GetLogicalId());

    auto &bvh = surfdata.fBVH[surfdata.fShells[in_state.GetLogicalId()].fBVH];

    vecgeom::Transformation3D lv_trans;
    in_state.TopMatrix(lv_trans);
    auto localpoint = lv_trans.Transform(point);
    auto localdir   = lv_trans.TransformDirection(direction);

    auto bvhstep            = stepmax;
    auto hitcandidate_index = BVHSurfNavigator::TestBVHCheckDaughterIntersections(bvh, localpoint, localdir, bvhstep);
    // If there is no physics step limitation, a surface must be found
    if (hitcandidate_index < 0 && stepmax == vecgeom::InfinityLength<Real_t>()) {
      hit_FS.Set(0, -1, 0); // frame_id = -1 indicates extruding overlap
      // This can happen if the exit point is outside the mother volume (extrusion)
      // To recover, one can return the mother state as output and a zero distance
      return stepmax;
    }

    // Now identify the common surface
    auto currentShell = surfdata.fShells[in_state.GetLogicalId()];
    if (hitcandidate_index < currentShell.fNExitingSurfaces) {
      // If the hit candidate is an exiting surface
      auto exiting_index = currentShell.fExitingSurfaces[hitcandidate_index];
      FSlocator hit_FS_tmp;
      surfdata.SceneToTouchableLocator(in_state, exiting_index, hit_FS_tmp);
      FSlocator out_frame;
      hit_FS_tmp.state = in_state;
      // Get the onsurf point in CS coordinates
      auto surf                    = surfdata.fCommonSurfaces[hit_FS_tmp.GetCSindex()];
      auto CS_trans                = surfdata.fGlobalTrans[surf.fTrans];
      Vector3D<Real_t> CS_local    = CS_trans.Transform(local_scene);
      Vector3D<Real_t> CS_localdir = CS_trans.TransformDirection(localdir_scene);

      auto onsurf_crt = CS_local + Real_t(bvhstep) * CS_localdir;
      /* auto inframe    =  */ ExitCS(hit_FS_tmp, /*is_hit=*/true, point, direction, Real_t(bvhstep), onsurf_crt,
                                      hit_FS.hit_surf, out_frame);
      hit_FS.exit_surf = hit_FS.hit_surf;
      out_state        = out_frame.state;

    } else {

      auto entering_index   = hitcandidate_index - currentShell.fNExitingSurfaces;
      auto local_surface_id = currentShell.fEnteringSurfaces[entering_index];
      auto framed_surface   = surfdata.fLocalSurf[local_surface_id];
      auto pvol_id          = currentShell.fEnteringSurfacesPvol[entering_index];
      // Get the placed volume the hit candidate belongs to
      auto pvol = vecgeom::NavigationState::ToPlacedVolume(pvol_id);
      // Create a copy of the navigation state
      auto pvol_navstate(in_state);
      // Get the navigation state of the daughter
      pvol_navstate.Push(pvol);
      // set hit_FS.hit_surf
      surfdata.SceneToTouchableLocator(pvol_navstate, framed_surface.fSurfIndex, hit_FS.hit_surf);
      hit_FS.hit_surf.state = in_state;
      // Get the onsurf point in CS coordinates
      auto surf     = surfdata.fCommonSurfaces[hit_FS.hit_surf.GetCSindex()];
      auto CS_trans = surfdata.fGlobalTrans[surf.fTrans];
      Vector3D<Real_t> CS_local, CS_localdir;

      unsigned short scene_id = 0, newscene_id = 0;
      bool is_scene = in_state.GetSceneId(scene_id, newscene_id);

      if (!is_scene) {
        CS_local    = CS_trans.Transform(local_scene);
        CS_localdir = CS_trans.TransformDirection(localdir_scene);
      } else {
        CS_local    = CS_trans.Transform(localpoint);
        CS_localdir = CS_trans.TransformDirection(localdir);
      }
      auto onsurf_crt = CS_local + Real_t(bvhstep) * CS_localdir;

      // Seek and cross entering frames
      FSlocator out_frame;
      EnterCS(hit_FS.hit_surf, point, direction, Real_t(bvhstep), onsurf_crt, out_frame);
      out_state = out_frame.state;
    }

    // Fix the out_state if pointing to a 0 scene
    if (out_state.GetSceneLevel() > 0 && out_state.GetNavIndex() == 0) out_state.PopScene();

    return bvhstep;
  }
};

} // namespace protonav
} // namespace vgbrep

#endif