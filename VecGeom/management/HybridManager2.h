/*
 * HybridManager2.h
 *
 *  Created on: 27.08.2015 by yang.zhang@cern.ch
 *  integrated into main development line by sandro.wenzel@cern.ch 24.11.2015
 *
 */

#ifndef VECGEOM_HYBRIDMANAGER_H
#define VECGEOM_HYBRIDMANAGER_H

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"
#include "VecGeom/base/AlignmentAllocator.h"
#include "VecGeom/management/ABBoxManager.h"

#include <queue>
#include <map>
#include <vector>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// a singleton class which manages a particular helper structure for voxelized navigation
// the helper structure is a hybrid between a flat list of aligned bounding boxed and a pure boundary volume hierarchy
// (BVH)
// and is hence called "HybridManager": TODO: come up with a more appropriate name
class HybridManager2 {

public:
  using Float_v = vecgeom::VectorBackend::Float_v;
  typedef float Real_t;
  typedef Vector3D<Float_v> ABBox_v;

  // scalar or vector vectors
  typedef Vector3D<Precision> ABBox_s;
  // use old style arrays here as std::vector has some problems
  // with Vector3D<kVc::Double_t>
  typedef ABBox_s *ABBoxContainer_t;
  typedef ABBox_v *ABBoxContainer_v;

  // first index is # daughter index, second is step
  typedef std::pair<int, double> BoxIdDistancePair_t;
  using HitContainer_t = std::vector<BoxIdDistancePair_t>;

  // the actual class encapsulating the bounding boxes + tree structure information
  // for a shallow bounding volume hierarchy acceleration structure
  struct HybridBoxAccelerationStructure {
    size_t fNumberOfOriginalBoxes = 0; // the number of objects/original bounding boxes this structure is representing
    ABBoxContainer_v fABBoxes_v   = nullptr;
    std::vector<int> *fNodeToDaughters = nullptr;
  };

private:
  // keeps/registers an acceleration structure for logical volumes
  std::vector<HybridBoxAccelerationStructure const *> fStructureHolder;

public:
  // initialized the helper structure for a given logical volume
  void InitStructure(LogicalVolume const *lvol);

  // initialized the helper structure for the complete geometry
  void InitVoxelStructureForCompleteGeometry()
  {
    std::vector<LogicalVolume const *> logicalvolumes;
    GeoManager::Instance().GetAllLogicalVolumes(logicalvolumes);
    for (auto lvol : logicalvolumes) {
      InitStructure(lvol);
    }
  }

  static HybridManager2 &Instance()
  {
    static HybridManager2 manager;
    return manager;
  }

  // removed/deletes the helper structure for a given logical volume
  void RemoveStructure(LogicalVolume const *lvol);

  template <typename C, typename Compare>
  static void sort(C &v, Compare cmp)
  {
    std::sort(v.begin(), v.end(), cmp);
  }

  // verbose output of helper structure
  VPlacedVolume const *PrintHybrid(LogicalVolume const *) const;

  ABBoxContainer_v GetABBoxes_v(HybridBoxAccelerationStructure const &structure, int &size, int &numberOfNodes) const
  {
    constexpr auto kVS = vecCore::VectorSize<Float_v>();
    VECGEOM_ASSERT(structure.fNumberOfOriginalBoxes != 0);
    int numberOfFirstLevelNodes =
        structure.fNumberOfOriginalBoxes / kVS + (structure.fNumberOfOriginalBoxes % kVS == 0 ? 0 : 1);
    numberOfNodes = numberOfFirstLevelNodes + structure.fNumberOfOriginalBoxes;
    size = numberOfFirstLevelNodes / kVS + (numberOfFirstLevelNodes % kVS == 0 ? 0 : 1) + numberOfFirstLevelNodes;
    VECGEOM_ASSERT(structure.fABBoxes_v != nullptr);
    return structure.fABBoxes_v;
  }

  HybridBoxAccelerationStructure const *GetAccStructure(LogicalVolume const *lvol) const
  {
    return fStructureHolder[lvol->id()];
  }

  // public method allowing to build a hybrid acceleration structure
  // given a vector of aligned bounding boxes
  // can be used by clients (not coupled to LogicalVolumes)
  HybridBoxAccelerationStructure *BuildStructure(ABBoxManager<Precision>::ABBoxContainer_t alignedboxes,
                                                 size_t numberofboxes) const;

private:
  // private methods use

  template <typename Container_t>
  void EqualizeClusters(Container_t &clusters, SOA3D<Precision> &centers, SOA3D<Precision> const &allvolumecenters,
                        size_t const maxNodeSize);

  template <typename Container_t>
  void InitClustersWithKMeans(ABBoxManager<Precision>::ABBoxContainer_t, int, Container_t &, SOA3D<Precision> &,
                              SOA3D<Precision> &, int const numberOfInterations = 50) const;

  void RecalculateCentres(SOA3D<Precision> &centers, SOA3D<Precision> const &allvolumecenters,
                          std::vector<std::vector<int>> const &clusters);
  struct OrderByDistance {
    bool operator()(std::pair<int, Precision> const &a, std::pair<int, Precision> const &b)
    {
      return a.second < b.second;
    }
  };
  typedef std::priority_queue<std::pair<int, Precision>, std::vector<std::pair<int, Precision>>, OrderByDistance>
      distanceQueue;
  void AssignVolumesToClusters(std::vector<std::vector<int>> &clusters, SOA3D<Precision> const &centers,
                               SOA3D<Precision> const &allvolumecenters);

  void BuildStructure_v(LogicalVolume const *vol);

  static bool IsBiggerCluster(std::vector<int> const &first, std::vector<int> const &second)
  {
    return first.size() > second.size();
  }

}; // end class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
