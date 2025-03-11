/*
 * ABBoxManager.h
 *
 *  Created on: 24.04.2015
 *      Author: swenzel
 */

#ifndef ABBOX_MANAGER_H
#define ABBOX_MANAGER_H

#pragma once

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"

#ifdef VECGEOM_USE_SURF
#include "VecGeom/surfaces/SurfData.h" // still need this for the init bvh function, then we can cut this
#include "VecGeom/surfaces/base/CpuTypes.h"
#endif

#include <map>
#include <vector>

namespace vecgeom {

// Singleton class for ABBox manager
// keeps a (centralized) map of volume pointers to vectors of aligned bounding boxes
// the alternative would be to include such a thing into logical volumes
template <typename Real_b>
class ABBoxManager {
public:
  typedef float Real_s;
  using Float_v = vecgeom::VectorBackend::Float_v;

  typedef Vector3D<Float_v> ABBox_v;
  // scalar
  typedef Vector3D<Real_b> ABBox_s;

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

  std::vector<ABBoxContainer_t> fVolToSurfaceABBoxesMap;

private:
  std::vector<ABBoxContainer_t> fVolToABBoxesMap;
  std::vector<ABBoxContainer_v> fVolToABBoxesMap_v;

public:
  // computes the aligned bounding box for a certain placed volume
  static void ComputeABBox(VPlacedVolume const *pvol, ABBox_s *lowerc, ABBox_s *upperc)
  {
    // idea: take the 8 corners of the bounding box in the reference frame of pvol
    // transform those corners and keep track of minimum and maximum extent
    // TODO: could make this code shorter with a more complex Vector3D class
    Vector3D<Precision> lower, upper;
    pvol->GetUnplacedVolume()->Extent(lower, upper);

    auto transformation = pvol->GetTransformation();
    TransformBoundingBox<Transformation3D>(lower, upper, *transformation);
    *lowerc = Vector3D<Precision>(lower.x() - 1E-3, lower.y() - 1E-3, lower.z() - 1E-3);
    *upperc = Vector3D<Precision>(upper.x() + 1E-3, upper.y() + 1E-3, upper.z() + 1E-3);
  }

  /** Splitted Aligned bounding boxes
   *
   *  This function will calculate the "numOfSlices" num of aligned bounding
   *  boxes of "numOfSlices" divisions of Bounding box of Placed Volume
   *
   *  input : 1. *pvol : A pointer to the Placed Volume.
   *  	    2. numOfSlices : that user want
   *
   *  output : lowerc : A STL vector containing the lower extent of the newly
   *  				  calculated "numOfSlices" num of Aligned Bounding boxes
   *
   *           upperc : A STL vector containing the upper extent of the newly
   *  				  calculated "numOfSlices" num of Aligned Bounding boxes
   *
   */
  static void ComputeSplittedABBox(VPlacedVolume const *pvol, std::vector<ABBox_s> &lowerc,
                                   std::vector<ABBox_s> &upperc, int numOfSlices)
  {

    // idea: Split the Placed Bounding Box of volume into the numOfSlices.
    //		  Then pass each placed slice to the ComputABBox function,
    //		  Get the coordinates of lower and upper corner of splittedABBox,
    //		  store these coordinates into the vector of coordinates provided
    //		  by the calling function.

    Vector3D<Precision> tmpLower, tmpUpper;
    pvol->GetUnplacedVolume()->Extent(tmpLower, tmpUpper);
    Vector3D<Precision> delta = tmpUpper - tmpLower;
    // chose the largest dimension for splitting
    int dim = 0;                                        // 0 for x, 1 for y,  2 for z //default considering X is largest
    if (delta.y() > delta.x() && delta.y() > delta.z()) // if y is largest
      dim = 1;
    if (delta.z() > delta.x() && delta.z() > delta.y()) // if z is largest
      dim = 2;

    Precision splitDx = 0., splitDy = 0., splitDz = 0.;
    splitDx = delta.x();
    splitDy = delta.y();
    splitDz = delta.z();

    // Only one will execute, considering slicing only in one dimension
    Precision val = 0.;

    if (dim == 0) {
      splitDx = delta.x() / numOfSlices;
      val     = -delta.x() / 2 + splitDx / 2;
    }
    if (dim == 1) {
      splitDy = delta.y() / numOfSlices;
      val     = -delta.y() / 2 + splitDy / 2;
    }
    if (dim == 2) {
      splitDz = delta.z() / numOfSlices;
      val     = -delta.z() / 2 + splitDz / 2;
    }

    // Precision minx, miny, minz, maxx, maxy, maxz;
    Transformation3D const *transf = pvol->GetTransformation();

    // Actual Stuff of slicing
    for (int i = 0; i < numOfSlices; i++) {
      // TODO :  Try to create sliced placed box.
      // Needs to modifiy translation parameters, without touching rotation
      // parameters

      Transformation3D transf2;
      Vector3D<Precision> transVec(0., 0., 0.);
      if (dim == 0) {
        transVec = transf->InverseTransform(Vector3D<Precision>(val, 0., 0.));
        val += splitDx;
      }
      if (dim == 1) {
        transVec = transf->InverseTransform(Vector3D<Precision>(0., val, 0.));
        val += splitDy;
      }
      if (dim == 2) {
        transVec = transf->InverseTransform(Vector3D<Precision>(0., 0., val));
        val += splitDz;
      }

      transf2.SetTranslation(transVec);
      transf2.SetRotation(transf->Rotation()[0], transf->Rotation()[1], transf->Rotation()[2], transf->Rotation()[3],
                          transf->Rotation()[4], transf->Rotation()[5], transf->Rotation()[6], transf->Rotation()[7],
                          transf->Rotation()[8]);
      transf2.SetProperties();

      Vector3D<Precision> lower1(0., 0., 0.), upper1(0., 0., 0.);
      UnplacedBox newBox2(splitDx / 2., splitDy / 2., splitDz / 2.);
      VPlacedVolume const *newBoxPlaced2 = LogicalVolume("", &newBox2).Place(&transf2);
      ABBoxManager<Precision>::Instance().ComputeABBox(newBoxPlaced2, &lower1, &upper1);
      lowerc.push_back(lower1);
      upperc.push_back(upper1);
    }
  }

  template <typename Transformation>
  static void TransformBoundingBox(Vector3D<Precision> &lower, Vector3D<Precision> &upper, Transformation const &transf)
  {
    auto delta = upper - lower;
    Precision minx, miny, minz, maxx, maxy, maxz;
    minx = kInfLength;
    miny = kInfLength;
    minz = kInfLength;
    maxx = -kInfLength;
    maxy = -kInfLength;
    maxz = -kInfLength;
    for (int x = 0; x <= 1; ++x)
      for (int y = 0; y <= 1; ++y)
        for (int z = 0; z <= 1; ++z) {
          Vector3D<Precision> corner;
          corner.x()                            = lower.x() + x * delta.x();
          corner.y()                            = lower.y() + y * delta.y();
          corner.z()                            = lower.z() + z * delta.z();
          Vector3D<Precision> transformedcorner = transf.InverseTransform(corner);
          minx                                  = std::min(minx, transformedcorner.x());
          miny                                  = std::min(miny, transformedcorner.y());
          minz                                  = std::min(minz, transformedcorner.z());
          maxx                                  = std::max(maxx, transformedcorner.x());
          maxy                                  = std::max(maxy, transformedcorner.y());
          maxz                                  = std::max(maxz, transformedcorner.z());
        }
    lower.Set(minx, miny, minz);
    upper.Set(maxx, maxy, maxz);
  }

  static ABBoxManager<Real_b> &Instance()
  {
    static ABBoxManager<Real_b> instance;
    return instance;
  }

  // initialize ABBoxes for a certain logical volume
  // very first version that just creates as many boxes as there are daughters
  // in reality we might have a lot more boxes than daughters (but not less)
  void InitABBoxes(LogicalVolume const *lvol)
  {
    if (fVolToABBoxesMap[lvol->id()] != nullptr) {
      // remove old boxes first
      RemoveABBoxes(lvol);
    }
    uint ndaughters              = lvol->GetDaughtersp()->size();
    ABBox_s *boxes               = new ABBox_s[2 * ndaughters];
    fVolToABBoxesMap[lvol->id()] = boxes;

    // same for the vector part
    int extra                      = (ndaughters % vecCore::VectorSize<Float_v>() > 0) ? 1 : 0;
    int size                       = 2 * (ndaughters / vecCore::VectorSize<Float_v>() + extra);
    ABBox_v *vectorboxes           = new ABBox_v[size];
    fVolToABBoxesMap_v[lvol->id()] = vectorboxes;

    // calculate boxes by iterating over daughters
    for (uint d = 0; d < ndaughters; ++d) {
      auto pvol = lvol->GetDaughtersp()->operator[](d);
      ComputeABBox(pvol, &boxes[2 * d], &boxes[2 * d + 1]);
    }

    // initialize vector version of Container
    int index                          = 0;
    unsigned int assignedscalarvectors = 0;
    for (uint i = 0; i < ndaughters; i += vecCore::VectorSize<Float_v>()) {
      Vector3D<Float_v> lower;
      Vector3D<Float_v> upper;
      // assign by components ( using generic VecCore API )
      for (uint k = 0; k < vecCore::VectorSize<Float_v>(); ++k) {
        if (2 * (i + k) < 2 * ndaughters) {
          vecCore::Set(lower.x(), k, boxes[2 * (i + k)].x());
          vecCore::Set(lower.y(), k, boxes[2 * (i + k)].y());
          vecCore::Set(lower.z(), k, boxes[2 * (i + k)].z());
          vecCore::Set(upper.x(), k, boxes[2 * (i + k) + 1].x());
          vecCore::Set(upper.y(), k, boxes[2 * (i + k) + 1].y());
          vecCore::Set(upper.z(), k, boxes[2 * (i + k) + 1].z());
          assignedscalarvectors += 2;
        } else {
          // filling in bounding boxes of zero size
          // better to put some irrational number than 0?
          vecCore::Scalar<Float_v> neginf = -InfinityLength<vecCore::Scalar<Float_v>>();
          vecCore::Set(lower.x(), k, neginf);
          vecCore::Set(lower.y(), k, neginf);
          vecCore::Set(lower.z(), k, neginf);
          vecCore::Set(upper.x(), k, neginf);
          vecCore::Set(upper.y(), k, neginf);
          vecCore::Set(upper.z(), k, neginf);
        }
      }
      vectorboxes[index++] = lower;
      vectorboxes[index++] = upper;
    }
    assert(index == size);
    assert(assignedscalarvectors == 2 * ndaughters);
    (void)assignedscalarvectors; // silence compiler warnings
  }

  // doing the same for many logical volumes
  template <typename Container>
  void InitABBoxes(Container const &lvolumes)
  {
    for (auto lvol : lvolumes) {
      InitABBoxes(lvol);
    }
  }

#ifdef VECGEOM_USE_SURF
  static void ComputeSurfaceABBox(vgbrep::FramedSurface<Precision, Transformation3DMP<Precision>> const &framedSurface,
                                  Transformation3D const &volumeTransform, ABBox_s &lowerc, ABBox_s &upperc,
                                  vgbrep::CPUsurfData<Precision> const &cpudata, LogicalVolume const *lvol,
                                  const bool crop)
  {
    Vector3D<Precision> lowert, uppert;

    // bounding box of volume that the surface belongs to
    Vector3D<Precision> lower_vol, upper_vol;
    if (crop) lvol->GetUnplacedVolume()->Extent(lower_vol, upper_vol);

    // Get the frame bounding box
    framedSurface.Extent3D(lowert, uppert, cpudata);
    Vector3D<Precision> lower(lowert[0], lowert[1], lowert[2]);
    Vector3D<Precision> upper(uppert[0], uppert[1], uppert[2]);

    // Apply the local transformation
    TransformBoundingBox<Transformation3DMP<Precision>>(lower, upper, framedSurface.fTrans);

    // Apply the transformation with respect to the mother LV
    TransformBoundingBox<Transformation3D>(lower, upper, volumeTransform);
    if (crop) TransformBoundingBox<Transformation3D>(lower_vol, upper_vol, volumeTransform);

    if (!crop) {
      lowerc.Set(lower.x(), lower.y(), lower.z());
      upperc.Set(upper.x(), upper.y(), upper.z());
    } else {
      lowerc.Set(std::max(lower_vol.x() - 1e-3, lower.x()), std::max(lower_vol.y() - 1e-3, lower.y()),
                 std::max(lower_vol.z() - 1e-3, lower.z()));
      upperc.Set(std::min(upper_vol.x() + 1e-3, upper.x()), std::min(upper_vol.y() + 1e-3, upper.y()),
                 std::min(upper_vol.z() + 1e-3, upper.z()));

      // if surface bounding box is outside of volume bounding box, remove it entirely
      if (lower.x() > upper_vol.x() + vecgeom::kTolerance || upper.x() < lower_vol.x() - vecgeom::kTolerance ||
          lower.y() > upper_vol.y() + vecgeom::kTolerance || upper.y() < lower_vol.y() - vecgeom::kTolerance ||
          lower.z() > upper_vol.z() + vecgeom::kTolerance || upper.z() < lower_vol.z() - vecgeom::kTolerance) {
        lowerc.Set(0., 0., 0.);
        upperc.Set(0., 0., 0.);
      }
    }
  }

  // Initialize AABoxes for the surfaces of a LogicalVolume and those of its daughters
  void InitSurfaceABBoxesVol(LogicalVolume const *lvol, vgbrep::CPUsurfData<Precision> &cpudata, bool crop = false)
  {
    if (fVolToSurfaceABBoxesMap[lvol->id()] != nullptr && !crop) {
      // remove old boxes first
      RemoveSurfaceABBoxes(lvol);
    }

    // Get the shell of the root LV
    auto &rootShell = cpudata.fShells[lvol->id()];
    if (rootShell.fSurfaces.size() == 0) return;

    ABBox_s *boxes;
    if (!crop) {
      // Allocate space for the AABBs (2 corners per surface)
      boxes = new ABBox_s[2 * rootShell.fExitingSurfaces.size() + 2 * rootShell.fEnteringSurfaces.size()];
      fVolToSurfaceABBoxesMap[lvol->id()] = boxes;
    } else {
      boxes = fVolToSurfaceABBoxesMap[lvol->id()];
    }

    auto const identityTransform = new Transformation3D();

    // Create AABBs for the Exiting surfaces of this volume
    for (auto motherSurfIndex = 0u; motherSurfIndex < rootShell.fExitingSurfaces.size(); motherSurfIndex++) {
      // Get the surface
      auto exiting_ind        = rootShell.fExitingSurfaces[motherSurfIndex];
      auto const localSurface = cpudata.fLocalSurfaces[rootShell.fSurfaces[exiting_ind]];

      ComputeSurfaceABBox(localSurface, *identityTransform, boxes[2 * motherSurfIndex], boxes[2 * motherSurfIndex + 1],
                          cpudata, lvol, crop);
    }

    // Now, iterate again over the daughters, and fill the array of AABBs
    // We need to go over the daughters since we need to know their transformation
    // Also initialize the local visible surfaces list in cpudata
    int localSurfIndex = 0;
    for (auto pvol : lvol->GetDaughters()) {
      // Get the shell
      auto shell = cpudata.fShells[pvol->GetLogicalVolume()->id()];
      // Iterate over the local surfaces in this shell
      for (auto i = 0u; i < shell.fExitingSurfaces.size(); i++) {
        auto exiting_ind         = shell.fExitingSurfaces[i];
        auto const &localSurface = cpudata.fLocalSurfaces[shell.fSurfaces[exiting_ind]];
        // Transformation of this daughter volume with respect to its mother
        auto daughterTransform = pvol->GetTransformation();
        ComputeSurfaceABBox(localSurface, *daughterTransform,
                            boxes[2 * (localSurfIndex + rootShell.fExitingSurfaces.size())],
                            boxes[2 * (localSurfIndex + rootShell.fExitingSurfaces.size()) + 1], cpudata,
                            pvol->GetLogicalVolume(), crop);
        localSurfIndex++;
      }
    }
  }

  // Initialize AABoxes for the surfaces of a list of LogicalVolumes and those of their daughters
  template <typename Container>
  void InitSurfaceABBoxes(Container const &lvolumes, vgbrep::CPUsurfData<Precision> &cpudata, bool crop = false)
  {
    for (auto lvol : lvolumes) {
      InitSurfaceABBoxesVol(lvol, cpudata, crop);
    }
  }

  // Initialize ABBoxes for all registered LogicalVolumes
  void InitABBoxesForSurfaces(vgbrep::CPUsurfData<Precision> &cpudata, bool crop = false)
  {
    auto &container = GeoManager::Instance().GetLogicalVolumesMap();
    std::vector<LogicalVolume const *> logicalvolumes;
    if (!crop) {
      fVolToSurfaceABBoxesMap.resize(container.size(), nullptr);
      logicalvolumes.reserve(container.size());
      for (const auto &p : container) {
        logicalvolumes.push_back(p.second);
      }
    } else {
      GeoManager::Instance().GetAllLogicalVolumes(logicalvolumes);
    }
    InitSurfaceABBoxes(logicalvolumes, cpudata, crop);
  }

  void RemoveSurfaceABBoxes(LogicalVolume const *lvol)
  {
    if (fVolToSurfaceABBoxesMap[lvol->id()] != nullptr) delete[] fVolToSurfaceABBoxesMap[lvol->id()];
  }

  // Returns the list of AABBs associated to a LogicalVolume
  ABBoxContainer_t GetSurfaceABBoxes(int ivol, int &size, vgbrep::CPUsurfData<Precision> const &cpudata)
  {
    size = cpudata.fShells[ivol].fExitingSurfaces.size() + cpudata.fShells[ivol].fEnteringSurfaces.size();
    return fVolToSurfaceABBoxesMap[ivol];
  }
#endif

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
  void RemoveABBoxes(LogicalVolume const *lvol)
  {
    if (fVolToABBoxesMap[lvol->id()] != nullptr) delete[] fVolToABBoxesMap[lvol->id()];
  }

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
using ABBoxManager_t = ABBoxManager<vecgeom::Precision>;

// output for hitboxes
template <typename stream, typename Real_b>
stream &operator<<(stream &s, std::vector<std::pair<unsigned int, double>> const &list)
{
  for (auto i : list) {
    s << "(" << i.first << "," << i.second << ")" << " ";
  }
  return s;
}
} // namespace vecgeom

#endif