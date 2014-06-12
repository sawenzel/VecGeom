/// \file UnplacedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/Array.h"
#include "volumes/Polygon.h"
#include "volumes/UnplacedVolume.h"

namespace VECGEOM_NAMESPACE {

class UnplacedPolyhedron : public VUnplacedVolume, public AlignedBase {

private:

  int fSideCount;
  Precision fPhiStart;
  Precision fPhiTotal;
  bool fHasPhi;

public:

#ifndef VECGEOM_NVCC
  UnplacedPolyhedron(
      const int sideCount,
      const Precision phiStart,
      const Precision phiTotal,
      const int zPlaneCount,
      const Precision zPlanes[],
      const Precision rInner[],
      const Precision rOuter[]);
#endif

  ~UnplacedPolyhedron();

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