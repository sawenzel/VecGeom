// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \author created by Sandro Wenzel

#include "VecGeom/management/FlatVoxelManager.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/navigation/SimpleABBoxSafetyEstimator.h"
#include "VecGeom/management/Logger.h"
#include <thread>
#include <future>
#include <random> // C++11 random numbers
#include <sstream>
#include <set>

// for timing measurement
#include "VecGeom/base/Stopwatch.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void FlatVoxelManager::InitStructure(LogicalVolume const *lvol)
{
  auto numregisteredlvols = GeoManager::Instance().GetRegisteredVolumesCount();
  if (fStructureHolder.size() != numregisteredlvols) {
    fStructureHolder.resize(numregisteredlvols, nullptr);
  }
  if (fStructureHolder[lvol->id()] != nullptr) {
    RemoveStructure(lvol);
  }
  fStructureHolder[lvol->id()] = BuildStructure(lvol);
}

// #define EXTREMELOOKUP

FlatVoxelHashMap<int, false> *FlatVoxelManager::BuildSafetyVoxels(LogicalVolume const *vol)
{
// TODO: How/Where is this supposed to be used, or is it obsolete?
#ifdef EXTREMELOOKUP
  Vector3D<Precision> lower, upper;
  vol->GetUnplacedVolume()->Extent(lower, upper);
  // these numbers have to be chosen dynamically to best match the situation
  int Nx      = 500;
  int Ny      = 500;
  int Nz      = 500;
  auto voxels = new FlatVoxelHashMap<float, true>(lower, 1.005 * (upper - lower), Nx, Ny, Nz);

  // fill the values ... we will sample this using a point cloud so as to avoid
  // filling useless values (completely inside volumes)
  // We should then try to fill the remaining holes via some kind of connecting/clustering
  // algorithm

  int numtasks = std::thread::hardware_concurrency();
  // int npointstotal{1000000000};
  int npointstotal{1000000};
  int pointspertask = npointstotal / numtasks;

  std::vector<std::mt19937> engines(numtasks);
  std::vector<SOA3D<float> *> points(numtasks);
  std::vector<std::vector<long> *> keyspertask(numtasks);
  // reserve space and init engines
  for (int t = 0; t < numtasks; ++t) {
    engines[t].seed(t * 13 + 11);
    points[t] = new SOA3D<float>(pointspertask);
  }

  Stopwatch timer;
  timer.Start();
  std::vector<std::future<void>> futures;
  for (int t = 0; t < numtasks; ++t) {
    bool verbose= (t==0);
    auto fut = std::async([&engines, &points, &keyspertask, vol, t, voxels] {
       bool foundPoints = volumeUtilities::FillUncontainedPoints(*vol, engines[t], *points[t], verbose);
      if (foundPoints) {
        keyspertask[t] = new std::vector<long>;
        voxels->getKeys(*points[t], *keyspertask[t]);
      } else {
        keyspertask[t] = nullptr;
      }
    });
    futures.push_back(std::move(fut));
  }
  std::for_each(futures.begin(), futures.end(), [](std::future<void> &fut) { fut.wait(); });
  auto elapsed = timer.Stop();
  VECGEOM_LOG(info) << "Sampling points and keys took " << elapsed << "s \n";

  timer.Start();
  // merge all keys
  std::vector<long> allkeys;
  for (int t = 0; t < numtasks; ++t) {
    if (keyspertask[t]) std::copy(keyspertask[t]->begin(), keyspertask[t]->end(), std::back_inserter(allkeys));
  }
  // Do we need to delete keyspertask[t] ??  JA

  if (allkeys.begin() == allkeys.end() || allkeys.size() <= 1) {
    // No points found -- the daughthers fill the current volume
    return nullptr;
  }
  std::sort(allkeys.begin(), allkeys.end());

  // get rid of duplicates easily since they are sorted
  std::vector<long> sortedkeys;
  for (auto &k : allkeys) {
    if (sortedkeys.size() == 0) {
      sortedkeys.push_back(k);
    } else if (k != sortedkeys.back()) {
      sortedkeys.push_back(k);
    }
  }
  VECGEOM_LOG(info) << "Generating unique keys took " << timer.Stop() << "s";
  VECGEOM_LOG(info) << "We have " << sortedkeys.size() << " sorted unique keys; fraction "
            << sortedkeys.size() / (1. * Nx * Ny * Nz) << " estimated volume "
            << sortedkeys.size() * safetyvoxels->getVoxelVolume();

  timer.Start();
  // make a vector where to collect safeties
  std::vector<float> safeties(sortedkeys.size());
  const auto safetyestimator = static_cast<SimpleABBoxSafetyEstimator const *>(SimpleABBoxSafetyEstimator::Instance());
  std::vector<std::future<void>> safetyfutures;

  VECGEOM_LOG(info) << " Calculating safeties ... in parallel ";
  for (int t = 0; t < numtasks; ++t) {
    auto fut = std::async([t, numtasks, &safeties, safetyvoxels, &sortedkeys, safetyestimator, vol] {
      // define start and end to work on
      size_t s         = sortedkeys.size();
      size_t chunksize = s / numtasks;
      size_t remainder = s % numtasks;
      int startindex   = t * chunksize;
      int endindex     = (t + 1) * chunksize;
      if (t == numtasks - 1) {
        endindex += remainder;
      }
      const float voxelhalfdiagonal = 0.5 * safetyvoxels->getVoxelDiagonal();

      for (int i = startindex; i < endindex; ++i) {
        auto k                        = sortedkeys[i];
        Vector3D<float> midpointfloat = safetyvoxels->keyToPos(k);
        Vector3D<double> midpointdouble(midpointfloat.x(), midpointfloat.y(), midpointfloat.z());
        auto safety = safetyestimator->TreatSafetyToIn(midpointdouble, vol, 1E20);
        safeties[i] = safety - voxelhalfdiagonal;
      }
    });
    safetyfutures.push_back(std::move(fut));
  }
  std::for_each(safetyfutures.begin(), safetyfutures.end(), [](std::future<void> &fut) { fut.wait(); });
  VECGEOM_LOG(info) << "Generating safeties took " << timer.Stop() << "s";

  auto filename = createName(vol, Nx, Ny, Nz);
  dumpToTFile(filename.c_str(), *points[0], sortedkeys, safeties);

  // finally register safeties in voxel map
  for (size_t i = 0; i < sortedkeys.size(); ++i) {
    auto k      = sortedkeys[i];
    auto safety = safeties[i];
    voxels->addPropertyForKey(k, safety);
  }
  VECGEOM_LOG(info) << " done";

  auto structure     = new VoxelStructure();
  structure->fVoxels = voxels;
  structure->fVol    = vol;

  return structure;
#else

  Vector3D<Precision> lower, upper;
  vol->GetUnplacedVolume()->Extent(lower, upper);
  // these numbers have to be chosen dynamically to best match the situation

  const auto &daughters   = vol->GetDaughters();
  const size_t ndaughters = daughters.size();
  //  a good guess is by the number of daughters and their average extent/dimensions
  VECGEOM_LOG(info) << " Setting up safety voxels for " << vol->GetName() << " with " << ndaughters << " daughters ";

  int Nx = std::max(4., 2 * std::sqrt(1. * ndaughters));
  int Ny = std::max(4., 2 * std::sqrt(1. * ndaughters));
  int Nz = std::max(4., 2 * std::sqrt(1. * ndaughters));
  //  int Nx = 10; //std::max(4., 2*std::sqrt(1.*ndaughters));
  //  int Ny = 10; //std::max(4., 2*std::sqrt(1.*ndaughters));
  //  int Nz = 10; //std::max(4., 2*std::sqrt(1.*ndaughters));

  // Initialize new structure
  auto* safetyvoxels = new FlatVoxelHashMap<int, false>(lower, 1.005 * (upper - lower), Nx, Ny, Nz);

  int numtasks = std::thread::hardware_concurrency();
  int npointstotal{1000 * Nx * Ny * Nz};
  int pointspertask = npointstotal / numtasks;

  std::vector<std::mt19937> engines(numtasks);
  std::vector<SOA3D<float> *> points(numtasks);
  std::vector<std::vector<long> *> keyspertask(numtasks);
  // reserve space and init engines
  for (int t = 0; t < numtasks; ++t) {
    engines[t].seed(t * 13 + 11);
    points[t] = new SOA3D<float>(pointspertask);
  }

  Stopwatch timer;
  timer.Start();
  std::vector<std::future<void>> futures;
  for (int t = 0; t < numtasks; ++t) {
    auto fut = std::async([&engines, &points, &keyspertask, vol, t, safetyvoxels] {
      bool foundPoints = volumeUtilities::FillUncontainedPoints(*vol, engines[t], *points[t]);
      if (foundPoints) {
        keyspertask[t] = new std::vector<long>;
        safetyvoxels->getKeys(*points[t], *keyspertask[t]);
        // delete points[t]; points[t] = nullptr;  // JA 2021.03.04 17:15 CEST ???
      } else {
        keyspertask[t] = nullptr;
        VECGEOM_LOG(info) << " WARNING: Found 0 uncontained points for " << vol->GetName()
                  << " -- expect problems in estimating safety.";
      }
    });
    futures.push_back(std::move(fut));
  }
  std::for_each(futures.begin(), futures.end(), [](std::future<void> &fut) { fut.wait(); });
  auto elapsed = timer.Stop();
  VECGEOM_LOG(info) << "Sampling points and keys took " << elapsed << "s";

  timer.Start();
  // merge all keys
  std::vector<long> allkeys;
  for (int t = 0; t < numtasks; ++t) {
    if (keyspertask[t]) std::copy(keyspertask[t]->begin(), keyspertask[t]->end(), std::back_inserter(allkeys));
  }

  if (allkeys.begin() == allkeys.end() || allkeys.size() <= 1) {
    // No points found -- the daughthers fill the current volume
    return nullptr;
  }

  std::sort(allkeys.begin(), allkeys.end());

  // get rid of duplicates easily since they are sorted
  std::vector<long> sortedkeys;
  for (auto &k : allkeys) {
    if (sortedkeys.size() == 0) {
      sortedkeys.push_back(k);
    } else if (k != sortedkeys.back()) {
      sortedkeys.push_back(k);
    }
  }
  VECGEOM_LOG(info) << "Generating unique keys took " << timer.Stop() << " s";

  VECGEOM_LOG(info) << " We have " << sortedkeys.size() << " sorted unique keys; fraction "
            << sortedkeys.size() / (1. * Nx * Ny * Nz) << " estimated volume "
            << sortedkeys.size() * safetyvoxels->getVoxelVolume();
  size_t minSize = 50;
  if (sortedkeys.size() < minSize) {
    VECGEOM_LOG(info) << " ** Keys are few -- not creating acceleration structure.";
    return nullptr;
  }
  //
  timer.Start();
  // make a vector where to collect the safetycandidates
  std::vector<std::vector<int>> safetycandidates(sortedkeys.size());

  std::vector<std::future<void>> safetyfutures;
#ifdef VOXEL_DEBUG
  VECGEOM_LOG(info) << " Calculating safety candidates ... in parallel";
#endif
  for (int t = 0; t < numtasks; ++t) {
    auto fut = std::async([t, numtasks, &safetycandidates, safetyvoxels, &sortedkeys, vol] {
      // define start and end to work on
      size_t s         = sortedkeys.size();
      size_t chunksize = s / numtasks;
      size_t remainder = s % numtasks;
      int startindex   = t * chunksize;
      int endindex     = (t + 1) * chunksize;
      if (t == numtasks - 1) {
        endindex += remainder;
      }

      int size{0};
      auto abboxcorners = ABBoxManager::Instance().GetABBoxes(vol, size);

      std::vector<Vector3D<float>> voxelsurfacepoints;
      for (int i = startindex; i < endindex; ++i) {
        auto k = sortedkeys[i];

        // ---
        // step 1 is to check intersections of this voxel with all object bounding boxes
        // ---
        Vector3D<float> keylower;
        Vector3D<float> keyupper;
        safetyvoxels->Extent(k, keylower, keyupper);
#ifdef VOXEL_DEBUG
        VECGEOM_LOG(info) << "KEY LOWER EXTENT " << keylower << "\n"
                          << "KEY UPPER EXTENT " << keyupper;
#endif
        // painful but we could speed up with SIMD
        for (int boxindex = 0; boxindex < size; ++boxindex) {
          const auto &boxlower = abboxcorners[2 * boxindex];
          const auto &boxupper = abboxcorners[2 * boxindex + 1];

          if (volumeUtilities::IntersectionExist(keylower, keyupper, boxlower, boxupper)) {
            safetycandidates[i].push_back(boxindex);
          }
        }

        // ---
        // step 2 is to determine which other candidates are possible --- for instance
        // important when this voxel is in empty space
        // (WE NEED TO BE CAREFUL ABOUT TOPOLOGICALLY WEIRD CASES)
        // ---
        std::set<int> othersafetycandidates;
        std::set<int> insidedaughtercandidates;

        voxelsurfacepoints.clear();
        volumeUtilities::GenerateRegularSurfacePointsOnBox(keylower, keyupper, 10, voxelsurfacepoints);
        for (const auto &sp : voxelsurfacepoints) {
          // surface point in double precission (needed for some interfaces)
          Vector3D<Precision> spdouble(sp.x(), sp.y(), sp.z());
#ifdef VOXEL_DEBUG
          VECGEOM_LOG(info) << "CHECKING SURFACE POINT " << sp;
#endif
          const auto inmother = vol->GetUnplacedVolume()->Contains(spdouble);
          if (!inmother) {
            othersafetycandidates.insert(-1);
            continue;
          }

          // we can use the knowledge about intersecting bounding boxes to query
          // daughter insection quickly
          auto daughters     = vol->GetDaughters();
          bool inanydaughter = false;
          for (const auto &boxindex : safetycandidates[i]) {
            const auto &boxlower = abboxcorners[2 * boxindex];
            const auto &boxupper = abboxcorners[2 * boxindex + 1];
            bool inboundingbox{false};
            ABBoxImplementation::ABBoxContainsKernelGeneric(boxlower, boxupper, sp, inboundingbox);
            if (inboundingbox) {
              // ASSUMING BOXINDEX == DAUGHTERINDEX !!
              if (daughters[boxindex]->Contains(spdouble)) {
                inanydaughter = true;
                // VECGEOM_LOG(info) << "used to ignore surface point " << sp << " inside daughter vol " << boxindex;

                insidedaughtercandidates.insert(boxindex);
                // Should only be inside one daughter volume
                break;
              }
            }
          }

          if (!inanydaughter) {
            // get safetytoout as reference length scale
            const auto safetyout    = vol->GetUnplacedVolume()->SafetyToOut(spdouble);
            const auto safetyoutsqr = safetyout * safetyout;
#ifdef VOXEL_DEBUG
            VECGEOM_LOG(info) << "POINT OK; MOTHER SAFETY " << safetyout;
#endif
            // get all intersecting objects within this distance
            // which are not yet part of candidates ... we are using
            // the simple safety estimator for this (could use better algorithms) but in a SIMD way

            int size{0};
            // fetches the SIMDized bounding box representations
            ABBoxManager::ABBoxContainer_v bboxes = ABBoxManager::Instance().GetABBoxes_v(vol, size);

            using IdDistPair_t = ABBoxManager::BoxIdDistancePair_t;
            char stackspace[VECGEOM_MAXDAUGHTERS * sizeof(IdDistPair_t)];
            IdDistPair_t *boxsafetylist = reinterpret_cast<IdDistPair_t *>(&stackspace);

            // calculate squared bounding box safeties in vectorized way which are within range of safety to mother
            auto ncandidates =
                SimpleABBoxSafetyEstimator::GetSafetyCandidates_v(spdouble, bboxes, size, boxsafetylist, safetyoutsqr);

            int bestcandidate = -1; // -1 means mother
            // final safety for this surfacepoint
            float finalsafetysqr = safetyoutsqr;
            for (size_t candidateindex = 0; candidateindex < ncandidates; ++candidateindex) {
              const auto volid          = boxsafetylist[candidateindex].first;
              const auto safetytoboxsqr = boxsafetylist[candidateindex].second;

              const auto candidatesafety = daughters[volid]->SafetyToIn(spdouble);
              const auto candsafetysqr   = candidatesafety * candidatesafety;
#ifdef VOXEL_DEBUG
              VECGEOM_LOG(info) << "DAUGH " << volid << " squared box saf " << safetytoboxsqr
                                << " cands " << candidatesafety;
#endif
              // we take the larger of boxsafety or candidatesafety as the safety for this object
              const auto thiscandidatesafetysqr = std::max<Precision>(candsafetysqr, safetytoboxsqr);

              // if this safety is smaller than the previously known safety
              if (thiscandidatesafetysqr <= finalsafetysqr) {
#ifdef VOXEL_DEBUG
                 VECGEOM_LOG(info) << "Updating best cand from " << bestcandidate << " to " << volid
                                   << " safety-sq = " << thiscandidatesafetysqr << " for sp = " << sp;
              }
#endif
                bestcandidate  = volid;
                finalsafetysqr = thiscandidatesafetysqr;
              }
            }
#ifdef VOXEL_DEBUG
            VECGEOM_LOG(info) << "Inserting best candidate " << bestcandidate << "\n";
#endif
            othersafetycandidates.insert(bestcandidate);
          } // if not in daughter
        }   // loop over surface points of voxel

        for (const auto &other : othersafetycandidates) {
          // we add the other candidates to the list of existing candidates
          auto iter = std::find(safetycandidates[i].begin(), safetycandidates[i].end(), other);
          if (iter == safetycandidates[i].end()) {
            safetycandidates[i].push_back(other);
          }
        }

#ifdef CHECK_IN_DAUGHTER_CANDIDATES
        // Check whether any 'in-daughter' candidates are not yet seen
        for (const auto &other : insidedaughtercandidates) {
          // we add the other candidates to the list of existing candidates
          auto iter = std::find(safetycandidates[i].begin(), safetycandidates[i].end(), other);
          if (iter == safetycandidates[i].end()) {
            VECGEOM_LOG(info) << "used to ignore 'inside daughter' vol " << other << "\n";
            safetycandidates[i].push_back(other);
          }
        }
#endif
        std::sort(safetycandidates[i].begin(), safetycandidates[i].end());
      } // loop over keys/voxels
    });
    safetyfutures.push_back(std::move(fut));
  }
  std::for_each(safetyfutures.begin(), safetyfutures.end(), [](std::future<void> &fut) { fut.wait(); });
  VECGEOM_LOG(info) << "Generating safeties took " << timer.Stop() << "s \n";
  // bool verboseAdd= false;
  // finally register safety or locate candidates in voxel hash map
  for (size_t i = 0; i < sortedkeys.size(); ++i) {
    auto key = sortedkeys[i];
    for (const auto &cand : safetycandidates[i]) {
      // if( verboseAdd ) { VECGEOM_LOG(info) << "Adding cand " << cand << " to key " << key << "\n"; }
      safetyvoxels->addPropertyForKey(key, cand);
    }
  }
  VECGEOM_LOG(info) << " done \n";
  return safetyvoxels;
#endif // extreme lookup
}

FlatVoxelHashMap<int, false> *FlatVoxelManager::BuildLocateVoxels(LogicalVolume const *vol)
{
  Vector3D<Precision> lower, upper;
  vol->GetUnplacedVolume()->Extent(lower, upper);
  // these numbers have to be chosen dynamically to best match the situation

  const auto &daughters   = vol->GetDaughters();
  const size_t ndaughters = daughters.size();
  //  a good guess is by the number of daughters and their average extent/dimensions
  VECGEOM_LOG(info) << "Setting up locate voxels for " << vol->GetName()
                    << " with " << ndaughters << " daughters \n";
  int Nx            = 10; // std::max(4., std::sqrt(1.*ndaughters));
  int Ny            = 10; // std::max(4., std::sqrt(1.*ndaughters));
  int Nz            = 10; // std::max(4., std::sqrt(1.*ndaughters));
  auto locatevoxels = new FlatVoxelHashMap<int, false>(lower, 1.005 * (upper - lower), Nx, Ny, Nz);
  size_t numkeys    = Nx * Ny * Nz;

  int numtasks = std::thread::hardware_concurrency();

  // here we simply iterate over all keys
  Stopwatch timer;
  timer.Start();
  //
  std::vector<long> sortedkeys;
  for (size_t i = 0; i < numkeys; ++i) {
    sortedkeys.push_back(i);
  }
  VECGEOM_LOG(info) << "Generating unique keys took " << timer.Stop() << "s \n";

  //
  timer.Start();
  // make a vector where to collect the locatecandidates
  std::vector<std::vector<int>> locatecandidates(sortedkeys.size());
  std::vector<std::future<void>> futures;

  VECGEOM_LOG(info) << " Calculating locate candidates ... in parallel ";
  for (int t = 0; t < numtasks; ++t) {
    auto fut = std::async([t, numtasks, &locatecandidates, locatevoxels, &sortedkeys, vol] {
      // define start and end to work on
      size_t s         = sortedkeys.size();
      size_t chunksize = s / numtasks;
      size_t remainder = s % numtasks;
      int startindex   = t * chunksize;
      int endindex     = (t + 1) * chunksize;
      if (t == numtasks - 1) {
        endindex += remainder;
      }
      int size{0};
      auto abboxcorners = ABBoxManager::Instance().GetABBoxes(vol, size);

      for (int i = startindex; i < endindex; ++i) {
        auto k = sortedkeys[i];
        Vector3D<float> keylower;
        Vector3D<float> keyupper;
        locatevoxels->Extent(k, keylower, keyupper);

        // painful but we could speed up with SIMD
        for (int boxindex = 0; boxindex < size; ++boxindex) {
          const auto &boxlower = abboxcorners[2 * boxindex];
          const auto &boxupper = abboxcorners[2 * boxindex + 1];

          if (volumeUtilities::IntersectionExist(keylower, keyupper, boxlower, boxupper)) {
            locatecandidates[i].push_back(boxindex);
          }
        }
      } // loop over keys/voxels
    });
    futures.push_back(std::move(fut));
  }
  std::for_each(futures.begin(), futures.end(), [](std::future<void> &fut) { fut.wait(); });
  VECGEOM_LOG(info) << "Generating locate voxels took " << timer.Stop() << "s \n";

  // finally register safety or locate candidates in voxel hash map
  for (size_t i = 0; i < sortedkeys.size(); ++i) {
    auto key = sortedkeys[i];
    for (const auto &cand : locatecandidates[i]) {
      locatevoxels->addPropertyForKey(key, cand);
    }
  }
  // locatevoxels->print();

  return locatevoxels;
}

FlatVoxelManager::VoxelStructure *FlatVoxelManager::BuildStructure(LogicalVolume const *vol)
{
  auto structure                      = new VoxelStructure();
  structure->fVoxelToCandidate        = BuildSafetyVoxels(vol);
  structure->fVoxelToLocateCandidates = BuildLocateVoxels(vol);
  structure->fVol                     = vol;

  return structure;
}

void FlatVoxelManager::RemoveStructure(LogicalVolume const *lvol)
{
  // FIXME: take care of memory deletion within acceleration structure
  if (fStructureHolder[lvol->id()]) delete fStructureHolder[lvol->id()];
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
