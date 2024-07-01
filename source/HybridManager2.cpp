/*
 * HybridManager.cpp
 *
 *  Created on: 03.08.2015
 *      Author: yang.zhang@cern.ch
 */

#include "VecGeom/management/HybridManager2.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/ABBoxManager.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include <map>
#include <vector>
#include <sstream>
#include <queue>
#include <set>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void HybridManager2::InitStructure(LogicalVolume const *lvol)
{
  auto numregisteredlvols = GeoManager::Instance().GetRegisteredVolumesCount();
  if (fStructureHolder.size() != numregisteredlvols) {
    fStructureHolder.resize(numregisteredlvols, nullptr);
  }
  if (fStructureHolder[lvol->id()] != nullptr) {
    RemoveStructure(lvol);
  }
  BuildStructure_v(lvol);
}

/**
 * build bvh bruteforce AND vectorized
 */
void HybridManager2::BuildStructure_v(LogicalVolume const *vol)
{
  // for a logical volume we are referring to the functions that builds everything giving just bounding
  // boxes
  int nDaughters{0};
  // get the boxes (and number of boxes), must be called before the BuildStructure
  // function call since otherwise nDaughters is not guaranteed to be initialized
  auto boxes                  = ABBoxManager<Precision>::Instance().GetABBoxes(vol, nDaughters);
  auto structure              = BuildStructure(boxes, nDaughters);
  fStructureHolder[vol->id()] = structure;
  assert((int)vol->GetDaughters().size() == nDaughters);
  assert(structure == nullptr || structure->fNumberOfOriginalBoxes != 0);
}

/**
 * build bvh bruteforce AND vectorized
 */
HybridManager2::HybridBoxAccelerationStructure *HybridManager2::BuildStructure(
    ABBoxManager<Precision>::ABBoxContainer_t abboxes, size_t numberofdaughters) const
{
  if (numberofdaughters == 0) return nullptr;

  constexpr auto kVS             = vecCore::VectorSize<HybridManager2::Float_v>();
  size_t numberOfFirstLevelNodes = numberofdaughters / kVS + (numberofdaughters % kVS == 0 ? 0 : 1);
  size_t vectorsize =
      numberOfFirstLevelNodes / kVS + (numberOfFirstLevelNodes % kVS == 0 ? 0 : 1) + numberOfFirstLevelNodes;

  std::vector<std::vector<int>> clusters(numberOfFirstLevelNodes);
  SOA3D<Precision> centers(numberOfFirstLevelNodes);
  SOA3D<Precision> allvolumecenters(numberofdaughters);

  InitClustersWithKMeans(abboxes, numberofdaughters, clusters, centers, allvolumecenters);

  // EqualizeClusters(clusters, centers, allvolumecenters, HybridManager2::vecCore::VectorSize<Float_v>());
  HybridBoxAccelerationStructure *structure = new HybridBoxAccelerationStructure();

  using VectorOfInts          = std::vector<int>; // to avoid clang-format error
  structure->fNodeToDaughters = new VectorOfInts[numberOfFirstLevelNodes];
  ABBoxContainer_v boxes_v    = new ABBox_v[vectorsize * 2];

  for (size_t i = 0; i < numberOfFirstLevelNodes; ++i) {
    for (size_t d = 0; d < clusters[i].size(); ++d) {
      int daughterIndex = clusters[i][d];
      structure->fNodeToDaughters[i].push_back(daughterIndex);
    }
  }

  // init boxes_v to -inf
  int vectorindex_v = 0;
  for (size_t i = 0; i < vectorsize * 2; i++) {
    boxes_v[i] = -InfinityLength<typename HybridManager2::Float_v>();
  }

  // init internal nodes for vectorized
  for (size_t i = 0; i < numberOfFirstLevelNodes; ++i) {
    if (i % kVS == 0) {
      vectorindex_v += 2;
    }
    Vector3D<Precision> lowerCornerFirstLevelNode(kInfLength), upperCornerFirstLevelNode(-kInfLength);
    for (size_t d = 0; d < clusters[i].size(); ++d) {
      int daughterIndex               = clusters[i][d];
      Vector3D<Precision> lowerCorner = abboxes[2 * daughterIndex];
      Vector3D<Precision> upperCorner = abboxes[2 * daughterIndex + 1];

      using vecCore::AssignLane;
      AssignLane(boxes_v[vectorindex_v].x(), d, lowerCorner.x());
      AssignLane(boxes_v[vectorindex_v].y(), d, lowerCorner.y());
      AssignLane(boxes_v[vectorindex_v].z(), d, lowerCorner.z());
      AssignLane(boxes_v[vectorindex_v + 1].x(), d, upperCorner.x());
      AssignLane(boxes_v[vectorindex_v + 1].y(), d, upperCorner.y());
      AssignLane(boxes_v[vectorindex_v + 1].z(), d, upperCorner.z());
      for (int axis = 0; axis < 3; axis++) {
        lowerCornerFirstLevelNode[axis] = std::min(lowerCornerFirstLevelNode[axis], lowerCorner[axis]);
        upperCornerFirstLevelNode[axis] = std::max(upperCornerFirstLevelNode[axis], upperCorner[axis]);
      }
    }
    vectorindex_v += 2;
    // insert internal node ABBOX in boxes_v
    int indexForInternalNode  = i / kVS;
    indexForInternalNode      = 2 * (kVS + 1) * indexForInternalNode;
    int offsetForInternalNode = i % kVS;
    using vecCore::AssignLane;
    AssignLane(boxes_v[indexForInternalNode].x(), offsetForInternalNode, lowerCornerFirstLevelNode.x());
    AssignLane(boxes_v[indexForInternalNode].y(), offsetForInternalNode, lowerCornerFirstLevelNode.y());
    AssignLane(boxes_v[indexForInternalNode].z(), offsetForInternalNode, lowerCornerFirstLevelNode.z());
    AssignLane(boxes_v[indexForInternalNode + 1].x(), offsetForInternalNode, upperCornerFirstLevelNode.x());
    AssignLane(boxes_v[indexForInternalNode + 1].y(), offsetForInternalNode, upperCornerFirstLevelNode.y());
    AssignLane(boxes_v[indexForInternalNode + 1].z(), offsetForInternalNode, upperCornerFirstLevelNode.z());
  }
  structure->fNumberOfOriginalBoxes = numberofdaughters;
  structure->fABBoxes_v             = boxes_v;
  return structure;
}

void HybridManager2::RemoveStructure(LogicalVolume const *lvol)
{
  // FIXME: take care of memory deletion within acceleration structure
  if (fStructureHolder[lvol->id()]) delete fStructureHolder[lvol->id()];
}

/**
 * assign daughter volumes to its closest cluster using the cluster centers stored in centers.
 * clusters need to be empty before this function is called
 */
void HybridManager2::AssignVolumesToClusters(std::vector<std::vector<int>> &clusters, SOA3D<Precision> const &centers,
                                             SOA3D<Precision> const &allvolumecenters)
{

  assert(centers.size() == clusters.size());
  int numberOfDaughers = allvolumecenters.size();
  int numberOfClusters = clusters.size();

  Precision minDistance;
  int closestCluster;

  for (int d = 0; d < numberOfDaughers; d++) {
    minDistance    = kInfLength;
    closestCluster = -1;

    Precision dist;
    for (int c = 0; c < numberOfClusters; ++c) {
      dist = (allvolumecenters[d] - centers[c]).Length2();
      if (dist < minDistance) {
        minDistance    = dist;
        closestCluster = c;
      }
    }

    clusters.at(closestCluster).push_back(d);
  }
}

void HybridManager2::RecalculateCentres(SOA3D<Precision> &centers, SOA3D<Precision> const &allvolumecenters,
                                        std::vector<std::vector<int>> const &clusters)
{
  assert(centers.size() == clusters.size());
  auto numberOfClusters = centers.size();
  for (size_t c = 0; c < numberOfClusters; ++c) {
    Vector3D<Precision> newCenter(0);
    for (size_t clustersize = 0; clustersize < clusters[c].size(); ++clustersize) {
      int daughterIndex = clusters[c][clustersize];
      newCenter += allvolumecenters[daughterIndex];
    }
    newCenter /= clusters[c].size();
    centers.set(c, newCenter);
  }
}

template <typename Container_t>
void HybridManager2::InitClustersWithKMeans(ABBoxManager<Precision>::ABBoxContainer_t boxes, int numberOfDaughters,
                                            Container_t &clusters, SOA3D<Precision> &centers,
                                            SOA3D<Precision> &allvolumecenters, int const numberOfIterations) const
{
  int numberOfClusters = clusters.size();

  Vector3D<Precision> meanCenter(0);
  std::set<int> daughterSet;
  for (int i = 0; i < numberOfDaughters; ++i) {
    Vector3D<Precision> center = 0.5 * (boxes[2 * i] + boxes[2 * i + 1]);
    allvolumecenters.set(i, center);
    daughterSet.insert(i);
    meanCenter += allvolumecenters[i];
  }
  meanCenter /= numberOfDaughters;

  size_t clustersize = (numberOfDaughters + numberOfClusters - 1) / numberOfClusters;
  for (int clusterindex = 0; clusterindex < numberOfClusters && !daughterSet.empty(); ++clusterindex) {
    Vector3D<Precision> clusterMean(0);
    while (clusters[clusterindex].size() < clustersize) {
      // parametrized lambda used in std::max_element + std::min_element below
      auto sortlambda = [&](Vector3D<Precision> const &center) {
        return [&, center](int a, int b) {
          return (allvolumecenters[a] - center).Mag2() < (allvolumecenters[b] - center).Mag2();
        };
      };

      int addDaughter = clusters[clusterindex].size() == 0
                            ? *std::max_element(daughterSet.begin(), daughterSet.end(), sortlambda(meanCenter))
                            : *std::min_element(daughterSet.begin(), daughterSet.end(), sortlambda(clusterMean));

      daughterSet.erase(addDaughter);
      clusters[clusterindex].emplace_back(addDaughter);

      if (daughterSet.empty()) break;
      meanCenter  = (meanCenter * (daughterSet.size() + 1) - allvolumecenters[addDaughter]) / daughterSet.size();
      clusterMean = (clusterMean * (clusters[clusterindex].size() - 1) + allvolumecenters[addDaughter]) /
                    clusters[clusterindex].size();
    } // end while
  }
}

template <typename Container_t>
void HybridManager2::EqualizeClusters(Container_t &clusters, SOA3D<Precision> &centers,
                                      SOA3D<Precision> const &allvolumecenters, size_t const maxNodeSize)
{
  // clusters need to be sorted
  size_t numberOfClusters = clusters.size();
  sort(clusters, IsBiggerCluster);
  for (size_t c = 0; c < numberOfClusters; ++c) {
    size_t clustersize = clusters[c].size();
    if (clustersize > maxNodeSize) {
      RecalculateCentres(centers, allvolumecenters, clusters);
      distanceQueue clusterelemToCenterMap; // pairs of index of elem in cluster and distance to cluster center
      for (size_t clusterElem = 0; clusterElem < clustersize; ++clusterElem) {
        Precision distance2 = (centers[c] - allvolumecenters[clusters[c][clusterElem]]).Length2();
        clusterelemToCenterMap.push(std::make_pair(clusters[c][clusterElem], distance2));
      }

      while (clusters[c].size() > maxNodeSize) {
        // int daughterIndex = clusters[c][it];
        int daughterIndex = clusterelemToCenterMap.top().first;
        clusterelemToCenterMap.pop();
        distanceQueue clusterToCenterMap2;
        for (size_t nextclusterIndex = c + 1; nextclusterIndex < numberOfClusters; nextclusterIndex++) {
          if (clusters[nextclusterIndex].size() < maxNodeSize) {
            Precision distanceToOtherCenters = (centers[nextclusterIndex] - allvolumecenters[daughterIndex]).Length2();
            clusterToCenterMap2.push(std::make_pair(nextclusterIndex, -distanceToOtherCenters));
          }
        }
        // find nearest cluster to daughterIndex

        clusters[c].erase(std::find(clusters[c].begin(), clusters[c].end(), daughterIndex));
        clusters[clusterToCenterMap2.top().first].push_back(daughterIndex);
        // RecalculateCentres(centers, allvolumecenters, clusters);
      }

      std::sort(clusters.begin() + c + 1, clusters.end(), IsBiggerCluster);
    }
  }
}

VPlacedVolume const *HybridManager2::PrintHybrid(LogicalVolume const *lvol) const { return 0; }
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
