/*
 * ABBoxManager.h
 *
 *  Created on: 24.04.2015
 *      Author: swenzel
 */

#pragma once

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"

#include <map>
#include <vector>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// Singleton class for ABBox manager
// keeps a (centralized) map of volume pointers to vectors of aligned bounding boxes
// the alternative would be to include such a thing into logical volumes
class ABBoxManager {
public:
  typedef float Real_t;
  using Float_v = vecgeom::VectorBackend::Float_v;

  typedef Vector3D<Float_v> ABBox_v;
  // scalar
  typedef Vector3D<Precision> ABBox_s;

  // use old style arrays here as std::vector has some problems
  // with Vector3D<kVc::Double_t>
  typedef ABBox_s *ABBoxContainer_t;
  typedef ABBox_v *ABBoxContainer_v;

  typedef std::pair<unsigned int, double> BoxIdDistancePair_t;

  // build an abstraction of sort to sort vectors and lists portably
  template <typename C, typename Compare>
  static void sort(C &v, Compare cmp)
  {
    std::sort(v.begin(), v.end(), cmp);
  }

  struct HitBoxComparatorFunctor {
    bool operator()(BoxIdDistancePair_t const &left, BoxIdDistancePair_t const &right)
    {
      return left.second < right.second;
    }
  };

  using FP_t = HitBoxComparatorFunctor;

private:
  std::vector<ABBoxContainer_t> fVolToABBoxesMap;
  std::vector<ABBoxContainer_v> fVolToABBoxesMap_v;

public:
  // computes the aligned bounding box for a certain placed volume
  static void ComputeABBox(VPlacedVolume const *pvol, ABBox_s *lower, ABBox_s *upper);
  static void ComputeSplittedABBox(VPlacedVolume const *pvol, std::vector<ABBox_s> &lower, std::vector<ABBox_s> &upper,
                                   int numOfSlices);

  static ABBoxManager &Instance()
  {
    static ABBoxManager instance;
    return instance;
  }

  // initialize ABBoxes for a certain logical volume
  // very first version that just creates as many boxes as there are daughters
  // in reality we might have a lot more boxes than daughters (but not less)
  void InitABBoxes(LogicalVolume const *lvol);

  // doing the same for many logical volumes
  template <typename Container>
  void InitABBoxes(Container const &lvolumes)
  {
    for (auto lvol : lvolumes) {
      InitABBoxes(lvol);
    }
  }

  void InitABBoxesForCompleteGeometry()
  {
    auto &container = GeoManager::Instance().GetLogicalVolumesMap();
    fVolToABBoxesMap.resize(container.size(), nullptr);
    fVolToABBoxesMap_v.resize(container.size(), nullptr);
    std::vector<LogicalVolume const *> logicalvolumes(container.size());
    logicalvolumes.resize(0);
    for (auto p : container) {
      logicalvolumes.push_back(p.second);
    }
    InitABBoxes(logicalvolumes);
  }

  // remove the boxes from the list
  void RemoveABBoxes(LogicalVolume const *lvol);

  // returns the Container for a given logical volume or nullptr if
  // it does not exist
  ABBoxContainer_t GetABBoxes(LogicalVolume const *lvol, int &size)
  {
    size = lvol->GetDaughtersp()->size();
    return fVolToABBoxesMap[lvol->id()];
  }

  // returns the Container for a given logical volume or nullptr if
  // it does not exist
  ABBoxContainer_v GetABBoxes_v(LogicalVolume const *lvol, int &size)
  {
    int ndaughters = lvol->GetDaughtersp()->size();
    int extra      = (ndaughters % vecCore::VectorSize<Float_v>() > 0) ? 1 : 0;
    size           = ndaughters / vecCore::VectorSize<Float_v>() + extra;
    return fVolToABBoxesMap_v[lvol->id()];
  }
};

// Alias for ABBoxManager for forward compatibility with a templated version
using ABBoxManager_t = ABBoxManager;

// output for hitboxes
template <typename stream>
stream &operator<<(stream &s, std::vector<ABBoxManager::BoxIdDistancePair_t> const &list)
{
  for (auto i : list) {
    s << "(" << i.first << "," << i.second << ")"
      << " ";
  }
  return s;
}
}
} // end namespace
