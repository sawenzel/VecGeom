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

#include "VecGeom/surfaces/SurfData.h"

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

private:
  std::vector<ABBoxContainer_t> fVolToSurfaceABBoxesMap;
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

  template <typename Real_t>
  static void ComputeSurfaceABBox(vgbrep::FramedSurface const &framedSurface,
                                  Transformation3DMP<Real_t> const &surfaceTransform,
                                  Transformation3D const &volumeTransform, ABBox_s &lowerc, ABBox_s &upperc,
                                  vgbrep::SurfData<Real_t> const &surfData)
  {
    Vector3D<Real_t> lowert, uppert;

    // Get the frame bounding box
    framedSurface.Extent3D(lowert, uppert, surfData);
    Vector3D<Precision> lower(lowert[0], lowert[1], lowert[2]);
    Vector3D<Precision> upper(uppert[0], uppert[1], uppert[2]);

    // Apply the local transformation
    TransformBoundingBox<Transformation3DMP<Real_t>>(lower, upper, surfaceTransform);

    // Apply the transformation with respect to the mother LV
    TransformBoundingBox<Transformation3D>(lower, upper, volumeTransform);

    lowerc.Set(lower.x() - 1E-3, lower.y() - 1E-3, lower.z() - 1E-3);
    upperc.Set(upper.x() + 1E-3, upper.y() + 1E-3, upper.z() + 1E-3);

    // lowerc.Set(lower.x(), lower.y(), lower.z());
    // upperc.Set(upper.x(), upper.y(), upper.z());
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

  // Initialize AABoxes for the surfaces of a LogicalVolume and those of its daughters
  template <typename Real_t>
  void InitSurfaceABBoxesVol(LogicalVolume const *lvol, vgbrep::SurfData<Real_t> &surfData)
  {
    if (fVolToSurfaceABBoxesMap[lvol->id()] != nullptr) {
      // remove old boxes first
      RemoveSurfaceABBoxes(lvol);
    }

    // Get the shell of the root LV
    auto &rootShell = surfData.fShells[lvol->id()];

    // Allocate space for the AABBs (2 corners per surface)
    ABBox_s *boxes = new ABBox_s[2 * rootShell.fNExitingSurfaces + 2 * rootShell.fNEnteringSurfaces];
    fVolToSurfaceABBoxesMap[lvol->id()] = boxes;

    auto const identityTransform = new Transformation3D();

    // Create AABBs for the Exiting surfaces of this volume
    for (int motherSurfIndex = 0; motherSurfIndex < rootShell.fNExitingSurfaces; motherSurfIndex++) {
      // Get the surface
      auto exiting_ind        = rootShell.fExitingSurfaces[motherSurfIndex];
      auto const localSurface = surfData.fLocalSurf[rootShell.fSurfaces[exiting_ind]];

      // Local transformation of this surface
      auto const &surfaceTransform = surfData.fLocalTrans[localSurface.fTrans];

      ComputeSurfaceABBox(localSurface, surfaceTransform, *identityTransform, boxes[2 * motherSurfIndex],
                          boxes[2 * motherSurfIndex + 1], surfData);
    }

    // Now, iterate again over the daughters, and fill the array of AABBs
    // We need to go over the daughters since we need to know their transformation
    // Also initialize the local visible surfaces list in surfData
    int localSurfIndex = 0;
    for (auto pvol : lvol->GetDaughters()) {
      // Get the shell
      auto shell = surfData.fShells[pvol->GetLogicalVolume()->id()];
      // Iterate over the local surfaces in this shell
      for (int i = 0; i < shell.fNExitingSurfaces; i++) {
        auto exiting_ind         = shell.fExitingSurfaces[i];
        auto const &localSurface = surfData.fLocalSurf[shell.fSurfaces[exiting_ind]];
        // Local transformation of the surface within this daughter volume
        auto const &surfaceTransform = surfData.fLocalTrans[localSurface.fTrans];
        // Transformation of this daughter volume with respect to its mother
        auto daughterTransform = pvol->GetTransformation();
        ComputeSurfaceABBox<Real_t>(localSurface, surfaceTransform, *daughterTransform,
                                    boxes[2 * (localSurfIndex + rootShell.fNExitingSurfaces)],
                                    boxes[2 * (localSurfIndex + rootShell.fNExitingSurfaces) + 1], surfData);
        localSurfIndex++;
      }
    }
  }

  // Initialize AABoxes for the surfaces of a list of LogicalVolumes and those of their daughters
  template <typename Container, typename Real_t>
  void InitSurfaceABBoxes(Container const &lvolumes, vgbrep::SurfData<Real_t> &surfData)
  {
    for (auto lvol : lvolumes) {
      InitSurfaceABBoxesVol<Real_t>(lvol, surfData);
    }
  }

  // Initialize ABBoxes for all registered LogicalVolumes
  template <typename Real_t>
  void InitABBoxesForSurfaces(vgbrep::SurfData<Real_t> &surfData)
  {
    auto &container = GeoManager::Instance().GetLogicalVolumesMap();
    fVolToSurfaceABBoxesMap.resize(container.size(), nullptr);
    std::vector<LogicalVolume const *> logicalvolumes;
    logicalvolumes.reserve(container.size());
    for (const auto &p : container) {
      logicalvolumes.push_back(p.second);
    }
    InitSurfaceABBoxes(logicalvolumes, surfData);
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
  void RemoveABBoxes(LogicalVolume const *lvol)
  {
    if (fVolToABBoxesMap[lvol->id()] != nullptr) delete[] fVolToABBoxesMap[lvol->id()];
  }

  void RemoveSurfaceABBoxes(LogicalVolume const *lvol)
  {
    if (fVolToSurfaceABBoxesMap[lvol->id()] != nullptr) delete[] fVolToSurfaceABBoxesMap[lvol->id()];
  }

  // Returns the list of AABBs associated to a LogicalVolume
  template <typename Real_t>
  ABBoxContainer_t GetSurfaceABBoxes(int ivol, int &size, vgbrep::SurfData<Real_t> const &surfData)
  {
    size = surfData.fShells[ivol].fNExitingSurfaces + surfData.fShells[ivol].fNEnteringSurfaces;
    return fVolToSurfaceABBoxesMap[ivol];
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
    s << "(" << i.first << "," << i.second << ")"
      << " ";
  }
  return s;
}
} // namespace vecgeom

#endif