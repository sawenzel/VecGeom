/// \file PolyhedronFace.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_POLYHEDRONFACE_H_
#define VECGEOM_VOLUMES_POLYHEDRONFACE_H_

#include "base/Global.h"

#include "base/Array.h"
#include "base/Vector2D.h"
#include "base/Vector3D.h"
#include "volumes/Face.h"
#include "volumes/Polygon.h"

namespace VECGEOM_NAMESPACE {

class PolyhedronFace : public Face {

private:

  Precision fSurfaceArea;
  Precision fPhiCached;
  Vector3D<Precision> fPhiCachedPoint;
  Vector2D<Precision> fR, fZ;

public:

  PolyhedronFace(Polygon::PolygonIterator cornerTail);

};

} // End global namespace

#endif // VECGEOM_VOLUMES_POLYHEDRONFACE_H_