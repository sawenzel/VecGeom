// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file LoopNavigator.h
 * @brief Navigation methods for geometry.
 */

#ifndef LOOP_NAVIGATOR_H_
#define LOOP_NAVIGATOR_H_

#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/LogicalVolume.h>

#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

namespace vecgeom {

class LoopNavigator {

public:
  // using Vector3D<Precision>           = vecgeom::Vector3D<Precision><vecgeom::Precision>;

  static constexpr Precision kBoundaryPush = 10 * vecgeom::kTolerance;

  /**
   * @brief Locate the deepest daughter volume that contains a point, starting from a given placed volume.
   *
   * ## Coordinate frames
   * - **Input point** @p point is in the **parent frame of @p vol** (i.e. the same frame expected by
   *   `VPlacedVolume::Inside` for @p vol).
   * - The NavigationState @p path should point to the **parent frame of @p vol**.
   * @param[in]  vol
   *   The starting placed volume (the mother at which the descent begins). Must be non-null if @p top is true.
   * @param[in]  point
   *   Query point **in the parent frame of @p vol**.
   * @param[in,out] path
   *   Navigation state to be filled. Must start at the  **parent frame of @p vol**.
   * @param[in]  top
   *   If true, the function first validates that the point is inside @p vol. If false, a nullptr is returned
   * @param[in]  exclude
   *   Optional placed volume to exclude once during the descent (useful to avoid immediately
   *   re-entering a volume you just exited). The exclusion is applied only to the first BVH
   *   query and then cleared.
   * @return
   *   The deepest placed volume that contains the point, or `nullptr` if @p top is true and
   *   `vol->Inside(point)` reports outside. When the descent stops at @p vol (i.e. no daughter
   *   contains the point), returns @p vol.
   */
  VECCORE_ATT_HOST_DEVICE
  static Daughter LocatePointIn(Daughter vol, Vector3D<Precision> const &point, vecgeom::NavigationState &path,
                                bool top, Daughter exclude = nullptr)
  {
    if (top) {
      VECGEOM_ASSERT(vol != nullptr);
      auto inside = vol->Inside(point);
      if (inside == kOutside) return nullptr;
      // Set the boundary state to the path
      if (inside == kSurface) path.SetBoundaryState(true);
    }

    Daughter currentvolume = vol;
    // transform the point into the reference frame of the `vol`
    Vector3D<Precision> currentpoint(vol->GetTransformation()->Transform<Precision>(point));
    path.Push(currentvolume);

    bool godeeper;
    do {
      godeeper = false;
      for (auto *daughter : currentvolume->GetDaughters()) {
        if (daughter == exclude) {
          continue;
        }
        auto transformedpoint = daughter->GetTransformation()->Transform<Precision>(currentpoint);
        auto inside           = daughter->GetUnplacedVolume()->Inside(transformedpoint);
        if (inside == EnumInside::kOutside) continue;
        if (inside == kSurface) path.SetBoundaryState(true);
        // Point inside child
        path.Push(daughter);
        currentpoint  = transformedpoint;
        currentvolume = daughter;
        godeeper      = true;
        // Skip checking other children
        break;
      }

      // Only exclude the placed volume once since we could enter it again via a
      // different volume history.
      exclude = nullptr;
    } while (godeeper);

    return currentvolume;
  }

  /**
   * @brief Find new final NavigationState after a NavigationState has just been left (i.e., after exiting its corresponding top volume)
   * The function will climb up the NavigationStates until the point is inside, and then descend again to find the
   * deepest volume containing the point. The just exited volume will be skipped
   * @param[in]  localpoint   Point in the **local frame of the navigation state (path.Top())**
   * @param[in,out] path      Navigation state of the volume that was just left; updated to the new deepest containing path.
   * @return The deepest placed volume that contains the point (after re-location), or nullptr if no mother exists.
   */
  VECCORE_ATT_HOST_DEVICE
  static Daughter RelocatePoint(Vector3D<Precision> const &localpoint, vecgeom::NavigationState &path)
  {
    Daughter currentmother          = path.Top();
    Daughter skip                   = nullptr;
    Vector3D<Precision> transformed = localpoint;
    // continue to climb up the Navigation tree until a volume is found which contains the volume
    do {
      skip = currentmother;
      path.Pop();
      transformed   = currentmother->GetTransformation()->InverseTransform(transformed);
      currentmother = path.Top();
    } while (currentmother && (currentmother->IsAssembly() || !currentmother->UnplacedContains(transformed)));

    // first volume up the hierarchy is found that contains the point. Now descend to find the deepest state that
    // contains the point
    if (currentmother) {
      return LocatePointInNavState(transformed, path, false, skip);
    }
    return currentmother;
  }

  /**
   * @brief Update NavState path to the deepest state that contains given point starting from the given path.
   * @param[in]  localpoint
   *   localpoint **in the reference frame of @p path.**.
   * @param[in,out] path
   *   Starting NavigationState; output of the final NavigationState
   * @param[in]  top
   *   If true, the function first validates that the point is inside @p path. If false, a nullptr is returned
   * @param[in]  exclude
   *   Optional placed volume to exclude once during the descent
   * @return
   *   The final top-volume of the NavState after finding the deepest NavState that contains the point
   */
  VECCORE_ATT_HOST_DEVICE
  static Daughter LocatePointInNavState(Vector3D<Precision> const &localpoint, vecgeom::NavigationState &path, bool top,
                                        vecgeom::VPlacedVolume const *exclude = nullptr)
  {
    VECGEOM_ASSERT(path.Top() != nullptr);

    auto currentLogical = path.Top()->GetLogicalVolume();
    VECGEOM_ASSERT(currentLogical != nullptr);

    // If requested, validate at the logical level (unplaced containment)
    if (top) {
      auto inside = currentLogical->GetUnplacedVolume()->Inside(localpoint);
      if (inside == kOutside) return nullptr;
      // Set the boundary state to the path
      if (inside == kSurface) path.SetBoundaryState(true);
    }

    Vector3D<Precision> currentpoint(localpoint);

    while (currentLogical->GetDaughters().size() > 0) {
      bool godeeper = false;

      // Linear scan over placed daughters of the current logical volume
      for (auto *daughter : currentLogical->GetDaughters()) {
        if (exclude && daughter == exclude) continue;

        // Transform the point from the current logical frame to the daughter's local frame
        const auto transformedpoint = daughter->GetTransformation()->Transform<Precision>(currentpoint);
        const auto inside           = daughter->GetUnplacedVolume()->Inside(transformedpoint);
        if (inside == kOutside) continue;
        if (inside == kSurface) path.SetBoundaryState(true);
        // Point inside child
        path.Push(daughter);
        currentpoint   = transformedpoint;
        currentLogical = daughter->GetLogicalVolume();
        godeeper       = true;
        // Skip checking other children
        break;
      }

      // Only exclude the placed volume once since we could enter it again via a
      // different volume history.
      exclude = nullptr;
      if (!godeeper) break; // no containing daughter at this level
    }

    return path.Top();
  }

private:
  // Computes a step in the current volume from the localpoint into localdir,
  // taking step_limit into account. If a volume is hit, the function calls
  // out_state.SetBoundaryState(true) and hitcandidate is set to the hit
  // daughter volume, or kept unchanged if the current volume is left.
  VECCORE_ATT_HOST_DEVICE
  static Precision ComputeStepAndHit(Vector3D<Precision> const &localpoint, Vector3D<Precision> const &localdir,
                                     Precision step_limit, vecgeom::NavigationState const &in_state,
                                     vecgeom::NavigationState &out_state, Daughter &hitcandidate)
  {
    if (step_limit <= 0) {
      // We don't need to ask any solid, this step is not limited by geometry.
      in_state.CopyTo(&out_state);
      out_state.SetBoundaryState(false);
      return 0;
    }

    Precision step = step_limit;
    Daughter pvol  = in_state.Top();

    // need to calc DistanceToOut first
    step = pvol->DistanceToOut(localpoint, localdir, step_limit);

    if (step < 0) step = 0;
    step = Min(step, step_limit);

    for (auto *daughter : pvol->GetDaughters()) {
      double ddistance = daughter->DistanceToIn(localpoint, localdir, step);
      ddistance        = vecCore::math::Max(ddistance, 0.);
      const bool valid = ddistance < step;
      hitcandidate     = valid ? daughter : hitcandidate;
      step             = valid ? ddistance : step;
    }

    // now we have the candidates and we prepare the out_state
    in_state.CopyTo(&out_state);
    if (step == vecgeom::kInfLength && step_limit > 0.) {
      out_state.SetBoundaryState(true);
      do {
        out_state.Pop();
      } while (out_state.Top()->IsAssembly());

      return vecgeom::kTolerance;
    }

    // Is geometry further away than physics step?
    if (step >= step_limit) {
      // Then this is a phyics step and we don't need to do anything.
      out_state.SetBoundaryState(false);
      return step_limit;
    }

    // Otherwise it is a geometry step and we push the point to the boundary.
    out_state.SetBoundaryState(true);

    if (step < 0) {
      step = 0;
    }

    return step;
  }

public:
  // Computes the isotropic safety from the globalpoint. The safety must be accurate only below the provided limit.
  VECCORE_ATT_HOST_DEVICE
  static Precision ComputeSafety(Vector3D<Precision> const &globalpoint, vecgeom::NavigationState const &state,
                                 Precision limit = InfinityLength<Precision>())
  {
    Daughter pvol = state.Top();
    if (pvol == nullptr) return kInfLength;
    vecgeom::Transformation3D m;
    state.TopMatrix(m);
    Vector3D<Precision> localpoint = m.Transform(globalpoint);

    Precision safety = pvol->SafetyToOut(localpoint);

    for (auto *daughter : pvol->GetDaughters()) {
      double dsafety = daughter->SafetyToIn(localpoint);
      safety         = dsafety < safety ? dsafety : safety;
    }

    return safety;
  }

  // Computes a step from the globalpoint (which must be in the current volume)
  // into globaldir, taking step_limit into account. If a volume is hit, the
  // function calls out_state.SetBoundaryState(true) and relocates the state to
  // the next volume.
  VECCORE_ATT_HOST_DEVICE
  static Precision ComputeStepAndPropagatedState(Vector3D<Precision> const &globalpoint,
                                                 Vector3D<Precision> const &globaldir, Precision step_limit,
                                                 vecgeom::NavigationState const &in_state,
                                                 vecgeom::NavigationState &out_state)
  {
    if (in_state.Top() == nullptr) return kInfLength;
    const Precision push = in_state.IsOnBoundary() ? kBoundaryPush : 0.;

    if (step_limit < push) {
      // Go as far as the step limit says, assuming there is no boundary.
      // TODO: Does this make sense?
      in_state.CopyTo(&out_state);
      out_state.SetBoundaryState(false);
      return step_limit;
    }
    step_limit -= push;

    // calculate local point/dir from global point/dir
    Vector3D<Precision> localpoint;
    Vector3D<Precision> localdir;
    // Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);
    vecgeom::Transformation3D m;
    in_state.TopMatrix(m);
    localpoint = m.Transform(globalpoint);
    localdir   = m.TransformDirection(globaldir);

    Daughter hitcandidate = nullptr;
    // Avoid computing the distance from boundary by pushing the point
    Precision step =
        ComputeStepAndHit(localpoint + push * localdir, localdir, step_limit, in_state, out_state, hitcandidate);
    step += push;

    if (out_state.IsOnBoundary()) {
      // Relocate the point after the step to refine out_state.
      localpoint += (step + kBoundaryPush) * localdir;

      if (!hitcandidate) {
        // We didn't hit a daughter but instead we're exiting the current volume.
        RelocatePoint(localpoint, out_state);
      } else {
        // Otherwise check if we're directly entering other daughters transitively.
        localpoint = hitcandidate->GetTransformation()->Transform(localpoint);
        LocatePointIn(hitcandidate, localpoint, out_state, false);
      }

      if (out_state.Top() != nullptr) {
        while (out_state.Top()->IsAssembly() || out_state.HasSamePathAsOther(in_state)) {
          out_state.Pop();
        }
        VECGEOM_ASSERT(!out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      }
    }

    return step;
  }

  // Computes a step from the globalpoint (which must be in the current volume)
  // into globaldir, taking step_limit into account. If a volume is hit, the
  // function calls out_state.SetBoundaryState(true) and
  //  - removes all volumes from out_state if the current volume is left, or
  //  - adds the hit daughter volume to out_state if one is hit.
  // However the function does _NOT_ relocate the state to the next volume,
  // that is entering multiple volumes that share a boundary.
  VECCORE_ATT_HOST_DEVICE
  static Precision ComputeStepAndNextVolume(Vector3D<Precision> const &globalpoint,
                                            Vector3D<Precision> const &globaldir, Precision step_limit,
                                            vecgeom::NavigationState const &in_state,
                                            vecgeom::NavigationState &out_state)
  {
    if (in_state.Top() == nullptr) return kInfLength;
    const Precision push = in_state.IsOnBoundary() ? kBoundaryPush : 0.;

    if (step_limit < push) {
      // Go as far as the step limit says, assuming there is no boundary.
      // TODO: Does this make sense?
      in_state.CopyTo(&out_state);
      if (step_limit > kTolerance) out_state.SetBoundaryState(false);
      return step_limit;
    }
    step_limit -= push;

    // calculate local point/dir from global point/dir
    Vector3D<Precision> localpoint;
    Vector3D<Precision> localdir;
    // Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);
    vecgeom::Transformation3D m;
    in_state.TopMatrix(m);
    localpoint = m.Transform(globalpoint);
    localdir   = m.TransformDirection(globaldir);

    Daughter hitcandidate = nullptr;
    // Avoid computing the distance from boundary by pushing the point
    Precision step =
        ComputeStepAndHit(localpoint + push * localdir, localdir, step_limit, in_state, out_state, hitcandidate);
    step += (step > 0.) * push;

    if (out_state.IsOnBoundary()) {
      if (!hitcandidate) {
        Daughter currentmother          = out_state.Top();
        Vector3D<Precision> transformed = localpoint;
        // Push the point inside the next volume.
        transformed += (step + kBoundaryPush) * localdir;
        do {
          // move to deepest parent still containing the point (not on boundary)
          out_state.SetLastExited();
          out_state.Pop();
          transformed   = currentmother->GetTransformation()->InverseTransform(transformed);
          currentmother = out_state.Top();
        } while (currentmother &&
                 (currentmother->IsAssembly() || currentmother->GetUnplacedVolume()->Inside(transformed) != kInside));
      } else {
        out_state.Push(hitcandidate);
      }
    }

    return step;
  }

  // Relocate a state that was returned from ComputeStepAndNextVolume: It
  // recursively locates the pushed point in the containing volume.
  VECCORE_ATT_HOST_DEVICE
  static void RelocateToNextVolume(Vector3D<Precision> const &globalpoint, Vector3D<Precision> const &globaldir,
                                   vecgeom::NavigationState &state)
  {
    // if already outside, don't do anything
    if (state.IsOutside()) return;

    // Push the point inside the next volume.
    // A.G. This should not be needed now since LocatePointIn is boundary-aware
    Vector3D<Precision> pushed = globalpoint /* + kBoundaryPush * globaldir*/;

    // Calculate local point from global point.
    vecgeom::Transformation3D m;
    state.TopMatrix(m);
    Vector3D<Precision> localpoint = m.Transform(pushed);

    // passing the state to check in + the local point in the reference frame of the state
    LocatePointInNavState(localpoint, state, false, state.GetLastExited());

    if (state.Top() != nullptr) {
      while (state.Top()->IsAssembly()) {
        state.Pop();
      }
      VECGEOM_ASSERT(!state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
    }
  }
};

} // namespace vecgeom

#endif // RT_LOOP_NAVIGATOR_H_
