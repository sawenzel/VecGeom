/// \file UnplacedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/Array.h"
#include "volumes/Polygon.h"
#include "volumes/UnplacedVolume.h"

#include <ostream>

namespace VECGEOM_NAMESPACE {

class UnplacedPolyhedron : public VUnplacedVolume, public AlignedBase {

private:

  int fSideCount;
  Precision fPhiStart;
  Precision fPhiEnd;
  bool fHasPhi;

public:

#ifndef VECGEOM_NVCC
  UnplacedPolyhedron(
      const int sideCount,
      const Precision phiStart,
      Precision phiTotal,
      const int zPlaneCount,
      const Precision zPlanes[],
      const Precision rInner[],
      const Precision rOuter[]);
#endif

  ~UnplacedPolyhedron();

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  virtual void Print(std::ostream &os) const;

#ifndef VECGEOM_NVCC
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) const { return NULL; }
#else
  __device__
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL) const {
    return NULL;
  }
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const { return NULL; }
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const {
    return NULL;
  }
#endif

private:

#ifndef VECGEOM_NVCC

  class PolyhedronSegment {

  private:

    struct PolyhedronEdge {
      Vector3D<Precision> normal;
      Vector3D<Precision> corner[2];
      Vector3D<Precision> cornerNormal[2];
    };

    struct PolyhedronSide {
      Vector3D<Precision> center, normal;
      Vector3D<Precision> surfRZ, surfPhi;
      PolyhedronEdge *edges[2];
      Vector3D<Precision> edgeNormal[2];
    };

    int fSideCount, fEdgeCount;
    Precision fPhiStart, fPhiEnd, fPhiTotal, fPhiDelta;
    bool fHasPhi;
    Vector2D<Precision> fStart, fEnd;
    Array<PolyhedronSide> fSides;
    Array<PolyhedronEdge> fEdges;
    Precision fRZLength;
    Vector2D<Precision> fPhiLength;
    Precision fEdgeNormal;

  public:

    PolyhedronSegment(const Polygon::const_iterator corner,
                      const int sideCount, const Precision phiStart,
                      const Precision phiTotal);

  };

#endif // VECGEOM_NVCC

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_