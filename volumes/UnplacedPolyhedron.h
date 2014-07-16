/// \file UnplacedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/Array.h"
#include "base/SOA3D.h"
#include "volumes/Polygon.h"
#include "volumes/UnplacedVolume.h"

#include <ostream>

namespace VECGEOM_NAMESPACE {

class UnplacedPolyhedron : public VUnplacedVolume, public AlignedBase {

private:

  struct PolyhedronEdges {
    SOA3D<Precision> normal;
    SOA3D<Precision> corner[2];
    SOA3D<Precision> cornerNormal[2];
    PolyhedronEdges(int edgeCount);
    PolyhedronEdges();
  };

  struct PolyhedronSides {
    SOA3D<Precision> center, normal;
    SOA3D<Precision> surfPhi, surfRZ;
    SOA3D<Precision> edgesNormal[2];
    PolyhedronEdges edges[2];
    Precision rZLength;
    Precision phiLength[2];
    Precision edgeNormal;
    PolyhedronSides(int sideCount);
    PolyhedronSides();
  };

  int fSideCount, fEdgeCount;
  Precision fPhiStart, fPhiEnd, fPhiDelta;
  Precision fEdgeNormal;
  bool fHasPhi;
  Array<PolyhedronSides> fSegments;

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

  VECGEOM_CUDA_HEADER_BOTH
  int GetSideCount() const { return fSideCount; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhiStart() const { return fPhiStart; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhiEnd() const { return fPhiEnd; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhiDelta() const { return fPhiDelta; }

  VECGEOM_CUDA_HEADER_BOTH
  bool HasPhi() const { return fHasPhi; }

  void ConstructSegment(Polygon::const_iterator corner,
                        Array<PolyhedronSides>::iterator segment);

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

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_