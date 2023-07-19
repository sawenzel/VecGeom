/*
 * EmbreeNavigator.h
 *
 *  Created on: May 18, 2018
 *      Author: swenzel
 */

#ifndef VECGEOM_NAVIGATION_EMBREENAVIGATOR_H_
#define VECGEOM_NAVIGATION_EMBREENAVIGATOR_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/management/EmbreeManager.h"
#include "VecGeom/navigation/VNavigator.h"
#include "VecGeom/navigation/HybridSafetyEstimator.h"
#include "VecGeom/navigation/SimpleABBoxNavigator.h"

#include <vector>
#include <stack>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

extern double g_step;
extern LogicalVolume const *g_lvol;
extern VPlacedVolume const *g_pvol;
extern VPlacedVolume const *g_lastexited;
extern Vector3D<double> const *g_pos;
extern Vector3D<double> const *g_dir;
extern Vector3D<float> const *g_normals;
extern int g_count;
extern std::vector<int> *g_geomIDs;
extern EmbreeManager::BoxIdDistancePair_t *g_hitlist;

// A navigator using Intel Embree as the underlying acceleration library to
// exclude hit targets quickly
template <bool MotherIsConvex = false>
class EmbreeNavigator : public VNavigatorHelper<EmbreeNavigator<MotherIsConvex>, MotherIsConvex> {

private:
  EmbreeManager &fAccelerationManager;
  EmbreeNavigator()
      : VNavigatorHelper<EmbreeNavigator<MotherIsConvex>, MotherIsConvex>(SimpleABBoxSafetyEstimator::Instance()),
        fAccelerationManager(EmbreeManager::Instance())
  {
  }

  static VPlacedVolume const *LookupDaughter(LogicalVolume const *lvol, int const daughterIndex)
  {
    return lvol->GetDaughters()[daughterIndex];
  }

  // a simple sort class (based on insertionsort)
  template <typename T> //, typename Cmp>
  static void insertionsort(T *arr, unsigned int N)
  {
    for (unsigned short i = 1; i < N; ++i) {
      T value    = arr[i];
      short hole = i;

      for (; hole > 0 && value.second < arr[hole - 1].second; --hole)
        arr[hole] = arr[hole - 1];

      arr[hole] = value;
    }
  }

  /**
   * Returns hitlist of daughter candidates (pairs of [daughter index, step to bounding box]) crossed by ray.
   */
  size_t GetHitCandidates_v(EmbreeManager::EmbreeAccelerationStructure const &accstructure,
                            Vector3D<Precision> const &point, Vector3D<Precision> const &dir,
                            EmbreeManager::BoxIdDistancePair_t *hitlist, float step) const
  {
    if (accstructure.fNumberObjects == 0) return false;
    // we need to setup an Embree ray
    RTCRayHit ray;
    ray.ray.flags = 0;
    ray.ray.org_x = point.x();
    ray.ray.org_y = point.y();
    ray.ray.org_z = point.z();
    ray.ray.dir_x = dir.x();
    ray.ray.dir_y = dir.y();
    ray.ray.dir_z = dir.z();
    ray.ray.tnear = 0.f;
    ray.ray.tfar  = 1E20f; // step is the current limit

    g_normals = accstructure.fNormals;
    g_hitlist = hitlist;
    g_count   = 0;
    g_geomIDs->clear(); // = geom_seen;

    RTCIntersectContext context;
    // we can't do a real capture ... so we do it via global variables
    auto hitFilter = [](const RTCFilterFunctionNArguments *args) {
      // check if a hit in this Embree structure leads to a hit
      // in the real geometry
      const auto hit = (RTCHit *)args->hit;
      int *hitvalid  = args->valid;
      const auto id  = hit->geomID;
      // if (g_geomIDs[id]) {
      //  hitvalid[0] = 0;
      //  return;
      // }
      // g_geomIDs[id] = true;
      if (std::find(g_geomIDs->begin(), g_geomIDs->end(), id) != g_geomIDs->end()) {
        hitvalid[0] = 0;
        return;
      }
      g_geomIDs->push_back(id);

      const auto ray      = (RTCRay *)args->ray;
      const auto normalID = id * 6 + hit->primID;
      const auto normal   = g_normals[normalID];
      const bool backface = ray->dir_x * normal.x() + ray->dir_y * normal.y() + ray->dir_z * normal.z() > 0;
      float dist          = backface ? -1.f : ray->tfar;

      // no need to sort twice !!! (in principle this thing is giving sorted inters??)

      // std::cerr << "putting " << id << " " << dist << " " << ray->tfar << "\n";
      g_hitlist[g_count++] = HybridManager2::BoxIdDistancePair_t(id, dist);
      // we strictly take all hits
      hitvalid[0] = 0;
    };

    rtcInitIntersectContext(&context);
    context.filter = hitFilter;
    rtcIntersect1(accstructure.fScene, &context, &ray);

    // at this moment we have the result
    return g_count;
  }

public:
  // a generic looper function that
  // given an acceleration structure (an aligned bounding box hierarchy),
  // a hit-query will be performed, the intersected boxes sorted, looped over
  // and a user hook called for processing
  // the user hook needs to indicate with a boolean return value whether to continue looping (false)
  // or whether we are done (true) and can exit

  // FIXME: might be generic enough to work for all possible kinds of BVH structures
  // FIXME: offer various sorting directions, etc.
  template <typename AccStructure, typename Func>
  VECGEOM_FORCE_INLINE
  void BVHSortedIntersectionsLooper(AccStructure const &accstructure, Vector3D<Precision> const &localpoint,
                                    Vector3D<Precision> const &localdir, float stepmax, Func &&userhook) const
  {
    // The following construct reserves stackspace for objects
    // of type IdDistPair_t WITHOUT initializing those objects
    using IdDistPair_t = HybridManager2::BoxIdDistancePair_t;
    char stackspace[VECGEOM_MAXFACETS * sizeof(IdDistPair_t)];
    IdDistPair_t *hitlist = reinterpret_cast<IdDistPair_t *>(&stackspace);

    auto ncandidates = GetHitCandidates_v(accstructure, localpoint, localdir, hitlist, stepmax);
    // sort candidates according to their bounding volume hit distance
    insertionsort(hitlist, ncandidates);

    // for (int c = 0; c < ncandidates; ++c) {
    //	std::cerr << "CAND " << 0 << hitlist[c].first << " " << hitlist[c].second << "\n";
    // }

    for (size_t index = 0; index < ncandidates; ++index) {
      auto hitbox = hitlist[index];
      // here we got the hit candidates
      // now we execute user specific code to process this "hitbox"
      auto done = userhook(hitbox);
      if (done) break;
    }
  }

  VECGEOM_FORCE_INLINE
  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                          Vector3D<Precision> const &localdir, NavigationState const *in_state,
                                          NavigationState * /*out_state*/, Precision &step,
                                          VPlacedVolume const *&hitcandidate) const override
  {
    if (lvol->GetDaughtersp()->size() == 0) return false;
    auto &accstructure = *fAccelerationManager.GetAccStructure(lvol);

    BVHSortedIntersectionsLooper(
        accstructure, localpoint, localdir, step, [&](HybridManager2::BoxIdDistancePair_t hitbox) {
          // only consider those hitboxes which are within potential reach of this step
          if (!(step < hitbox.second)) {
            VPlacedVolume const *candidate = LookupDaughter(lvol, hitbox.first);
            Precision ddistance            = candidate->DistanceToIn(localpoint, localdir, step);
            const auto valid               = !IsInf(ddistance) && ddistance < step &&
                               !((ddistance <= 0.) && in_state && in_state->GetLastExited() == candidate);
            hitcandidate = valid ? candidate : hitcandidate;
            step         = valid ? ddistance : step;
            return false; // not yet done; need to continue in looper
          }
          return true; // mark done in this case
        });
    return false;
  }

  //  VECGEOM_FORCE_INLINE
  //  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
  //                                          Vector3D<Precision> const &localdir, NavigationState const *in_state,
  //                                          NavigationState * /*out_state*/, Precision &step,
  //                                          VPlacedVolume const *&hitcandidate) const override
  //  {
  //	bool stackspace[VECGEOM_MAXFACETS * sizeof(bool)]; // alternatives are a thread_local static vector?
  //
  //    if (lvol->GetDaughtersp()->size() == 0) return false;
  //    const auto &accstructure = *fAccelerationManager.GetAccStructure(lvol);
  //
  //    for (int i = 0; i < lvol->GetDaughtersp()->size(); ++i) {
  //      stackspace[i] = false;
  //    }
  //
  //    // we need to setup an Embree ray
  //    RTCRayHit ray;
  //    ray.ray.flags = 0;
  //    ray.ray.org_x = localpoint.x();
  //    ray.ray.org_y = localpoint.y();
  //    ray.ray.org_z = localpoint.z();
  //    ray.ray.dir_x = localdir.x();
  //    ray.ray.dir_y = localdir.y();
  //    ray.ray.dir_z = localdir.z();
  //    ray.ray.tnear = 0.f;
  //    ray.ray.tfar  = 1E20; // step is the current limit
  //
  //    g_pos = &localpoint;
  //    g_dir = &localdir;
  //    g_step = step;
  //    g_lastexited = in_state ? in_state->GetLastExited() : nullptr;
  //    g_normals = accstructure.fNormals;
  //
  //    g_lvol = lvol;
  //    g_pvol = nullptr;
  //
  //    // NOT THREAD SAFE and VERY DANGEROUS; I WOULD INSTEAD VERY MUCH
  //    // LIKE TO BE ABLE TO CAPTURE THINGS IN THE FILTER LAMBDA
  //    g_geomIDs = stackspace;
  //    RTCIntersectContext context;
  //
  //    static int counter = 0;
  //    std::cerr << "# " << counter++ << "\n";
  //
  //    // we can't do a real capture ... so we do it via global variables
  //    auto hitFilter = [](const RTCFilterFunctionNArguments *args) {
  //      // check if a hit in this Embree structure leads to a hit
  //      // in the real geometry
  //
  //      assert(args->N == 1);
  //      const auto hit = (RTCHit *)args->hit;
  //      const auto id  = hit->geomID;
  //      int *hitvalid  = args->valid;
  //
  //      // if geometryID (aka bounding volume mesh) already treated we can exit
  //      if (g_geomIDs[id] == true) {
  //        hitvalid[0] = 0;
  //        return;
  //      }
  //      g_geomIDs[id] = true;
  //
  //      const auto ray = (RTCRay *)args->ray;
  //      std::cerr << id << " "
  //                << " " << hit->primID << " " << ray->tfar << " (" << hit->Ng_x << "," << hit->Ng_y << "," <<
  //                hit->Ng_z
  //                << ") ";
  //      const auto normalID = id * 12 + hit->primID;
  //      const auto normal   = g_normals[normalID];
  //      const bool backface = ray->dir_x * normal.x() + ray->dir_y * normal.y() + ray->dir_z * normal.z() < 0;
  //      std::cerr << "backface " << backface << "\n";
  //
  //      // this is assuming we get the hits in increasing distance order which somehow is not true???
  //      //if (!backface && g_step < ray->tfar) {
  //        // we can return early here (and not invalidating the hit) --> we are done!!
  //       // return;
  //      //}
  //
  //      const auto candidate = LookupDaughter(g_lvol, id);
  //      const auto ddistance = candidate->DistanceToIn(*g_pos, *g_dir, g_step);
  //      const auto valid = !IsInf(ddistance) && ddistance < g_step && !((ddistance <= 0.) && g_lastexited ==
  //      candidate);
  //      g_pvol           = valid ? candidate : g_pvol;
  //      g_step           = valid ? ddistance : g_step;
  //      hitvalid[0]      = 0;
  //    };
  //
  //    rtcInitIntersectContext(&context);
  //    context.filter = hitFilter;
  //    rtcIntersect1(accstructure.fScene, &context, &ray);
  //
  //    // at this moment we have the result
  //    hitcandidate = g_pvol;
  //    step = g_step;
  //    return false;
  //  }

  static VNavigator *Instance()
  {
    static EmbreeNavigator instance;
    return &instance;
  }

  static constexpr const char *gClassNameString = "EmbreeNavigator";
  typedef SimpleABBoxSafetyEstimator SafetyEstimator_t;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VECGEOM_NAVIGATION_EMBREENAVIGATOR_H_ */
