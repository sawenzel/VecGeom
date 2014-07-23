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

  typedef Array<Vector3D<Precision> > VecArray_t;

public:

  // struct PolyhedronEdges {
  //   SOA3D<Precision> normal;
  //   SOA3D<Precision> corner[2];
  //   SOA3D<Precision> cornerNormal[2];
  //   PolyhedronEdges(int sideCount);
  //   PolyhedronEdges();
  //   void Allocate(int sideCount);
  // };

  // struct PolyhedronSegment {
  //   SOA3D<Precision> center, normal;
  //   SOA3D<Precision> surfPhi, surfRZ;
  //   SOA3D<Precision> edgeNormal[2];
  //   PolyhedronEdges edge[2];
  //   Precision rZLength;
  //   Precision phiLength[2];
  //   Precision rZPhiNormal;
  //   PolyhedronSegment(int sideCount);
  //   PolyhedronSegment();
  //   void Allocate(int sideCount);
  // };

  struct PolyhedronEdge {
    Vector3D<Precision> normal;
    Vector3D<Precision> corner[2], cornerNormal[2];
  };

  struct PolyhedronSide {
    Vector3D<Precision> center, normal;
    Vector3D<Precision> surfPhi, surfRZ;
    Vector3D<Precision> edgeNormal[2];
    PolyhedronEdge edges[2];
  };

  struct PolyhedronSegment {
    Array<PolyhedronSide> sides;
    Precision rZLength;
    Precision phiLength[2];
    Precision rZPhiNormal;
  };

private:

  int fSideCount;
  Precision fPhiStart, fPhiEnd, fPhiDelta;
  Precision fEdgeNormal;
  bool fHasPhi;
  Array<PolyhedronSegment> fSegments;

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
  PolyhedronSegment const& GetSegment(size_t i) const { return fSegments[i]; }

  VECGEOM_CUDA_HEADER_BOTH
  Array<PolyhedronSegment> const& GetSegments() const { return fSegments; }

  VECGEOM_CUDA_HEADER_BOTH
  bool HasPhi() const { return fHasPhi; }

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  virtual void Print(std::ostream &os) const;

  VECGEOM_CUDA_HEADER_DEVICE
  VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
      const int id,
#endif
      VPlacedVolume *const placement) const;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const {
    Assert(0, "NYI");
    return NULL;
  }
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const {
    Assert(0, "NYI");
    return NULL;
  }
#endif

private:

  void ConstructSegment(Polygon::const_iterator corner,
                        Array<PolyhedronSegment>::iterator segment);

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_